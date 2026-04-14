#!/usr/bin/env python3
"""
Tradutor de Imagens - Web App
Flask server com interface visual para traduzir imagens de produtos.
"""
from __future__ import annotations

import os
import sys
import base64
import csv
import io
import re
import time
import uuid
import json
import shutil
import tempfile
import zipfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests as http_requests
from bs4 import BeautifulSoup
import sqlite3
import hashlib
import functools
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    print("Erro: pip install openai")
    sys.exit(1)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Aviso: anthropic não instalado. Tradução via Claude indisponível.")

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Aviso: google-genai não instalado. Substituição de produto indisponível.")

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False
    print("Aviso: cloudscraper não instalado. Bypass anti-bot indisponível.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
app.config["PREFERRED_URL_SCHEME"] = "https"
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "imagetools-default-secret-key-2024-prod")

# Fix for running behind reverse proxy (Railway) — ensures request.host_url uses https
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Cloud: usar volume persistente /data se disponível, senão /tmp
if os.environ.get("RAILWAY_ENVIRONMENT"):
    # Railway volume mount (must be configured in Railway dashboard)
    PERSIST_DIR = Path(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/data"))
    if not PERSIST_DIR.exists():
        PERSIST_DIR = Path(tempfile.gettempdir())
    UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "img-uploads"
    OUTPUT_FOLDER = PERSIST_DIR / "img-output"
    DB_PATH = PERSIST_DIR / "imagetools.db"
else:
    UPLOAD_FOLDER = Path(__file__).parent / "uploads"
    OUTPUT_FOLDER = Path(__file__).parent / "output"
    DB_PATH = Path(__file__).parent / "imagetools.db"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)


# ─── Database ─────────────────────────────────────────────

def get_db():
    """Get a database connection (thread-local)."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS shopify_stores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            store_name TEXT NOT NULL DEFAULT '',
            store_url TEXT NOT NULL,
            access_token TEXT NOT NULL DEFAULT '',
            client_id TEXT NOT NULL DEFAULT '',
            client_secret TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_id TEXT NOT NULL UNIQUE,
            source_type TEXT NOT NULL DEFAULT '',
            source_label TEXT NOT NULL DEFAULT '',
            product_count INTEGER NOT NULL DEFAULT 0,
            image_count INTEGER NOT NULL DEFAULT 0,
            languages TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'translating',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS team_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL,
            member_id INTEGER,
            email TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'agent',
            status TEXT NOT NULL DEFAULT 'pending',
            invite_token TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (member_id) REFERENCES users(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            owner_id INTEGER NOT NULL,
            call_type TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            tokens_input INTEGER NOT NULL DEFAULT 0,
            tokens_output INTEGER NOT NULL DEFAULT 0,
            images_count INTEGER NOT NULL DEFAULT 0,
            estimated_cost REAL NOT NULL DEFAULT 0.0,
            job_id TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)
    # Add columns if they don't exist (migration for existing DBs)
    for col_sql in [
        "ALTER TABLE shopify_stores ADD COLUMN client_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE shopify_stores ADD COLUMN client_secret TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE users ADD COLUMN owner_id INTEGER REFERENCES users(id)",
        "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'owner'",
    ]:
        try:
            conn.execute(col_sql)
        except Exception:
            pass
    conn.commit()
    conn.close()


init_db()


# ─── Email Notifications ────────────────────────────────

SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "") or SMTP_USER
APP_URL = os.environ.get("APP_URL", "https://web-production-fff15.up.railway.app")


def send_invite_email(to_email: str, owner_name: str, invite_token: str) -> bool:
    """Send team invite email via SMTP. Returns True on success."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        logger.warning("[Email] SMTP not configured — skipping email send")
        return False

    signup_url = f"{APP_URL.rstrip('/')}/?invite={invite_token}"

    subject = f"{owner_name} te convidou para a equipe no Image Tools"
    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 520px; margin: 0 auto; padding: 2rem;">
        <h2 style="color: #111827; font-size: 1.3rem;">Convite para Equipe</h2>
        <p style="color: #374151; font-size: 0.95rem; line-height: 1.6;">
            <strong>{owner_name}</strong> te convidou para fazer parte da equipe no <strong>Image Tools</strong>.
        </p>
        <p style="color: #374151; font-size: 0.95rem; line-height: 1.6;">
            Clique no botão abaixo para criar sua conta e se juntar à equipe:
        </p>
        <div style="text-align: center; margin: 1.5rem 0;">
            <a href="{signup_url}" style="display: inline-block; background: #6366f1; color: #fff; padding: 0.75rem 2rem; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 0.95rem;">
                Aceitar Convite
            </a>
        </div>
        <p style="color: #6b7280; font-size: 0.85rem;">
            Ou acesse diretamente: <a href="{signup_url}" style="color: #6366f1;">{signup_url}</a>
        </p>
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0;">
        <p style="color: #9ca3af; font-size: 0.8rem;">
            Se você não esperava este convite, pode ignorar este email.
        </p>
    </div>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            if SMTP_PORT != 25:
                server.starttls()
                server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())
        logger.info(f"[Email] Invite sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"[Email] Failed to send invite to {to_email}: {e}")
        return False


# ─── API Usage Tracking ─────────────────────────────────

# Cost estimates per model (USD)
API_COSTS = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "gpt-image-1": {"per_image": 0.02},
    "gemini-3-pro-image": {"per_image": 0.03},
    "gemini-3.1-flash-image-preview": {"per_image": 0.015},
    "gemini-2.5-flash-image": {"per_image": 0.01},
    "gemini-2.0-flash-exp-image-generation": {"per_image": 0.01},
}


def get_owner_id(user_id: int) -> int:
    """Get the team owner ID for a user (self if owner, or their owner_id)."""
    conn = get_db()
    user = conn.execute("SELECT id, owner_id FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if user and user["owner_id"]:
        return user["owner_id"]
    return user_id


def track_api_usage(user_id: int, call_type: str, model: str = "",
                    tokens_input: int = 0, tokens_output: int = 0,
                    images_count: int = 0, job_id: str = ""):
    """Record an API call for usage tracking."""
    owner_id = get_owner_id(user_id)
    cost = 0.0
    model_costs = API_COSTS.get(model, {})
    if tokens_input or tokens_output:
        cost = tokens_input * model_costs.get("input", 0) + tokens_output * model_costs.get("output", 0)
    if images_count:
        cost += images_count * model_costs.get("per_image", 0.02)

    conn = get_db()
    conn.execute(
        """INSERT INTO api_usage (user_id, owner_id, call_type, model, tokens_input, tokens_output,
           images_count, estimated_cost, job_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, owner_id, call_type, model, tokens_input, tokens_output, images_count, cost, job_id)
    )
    conn.commit()
    conn.close()


def save_job_record(user_id: int, job_id: str, source_type: str, source_label: str,
                    product_count: int, image_count: int, languages: str = "", status: str = "created"):
    """Save or update a job record in the database."""
    conn = get_db()
    existing = conn.execute("SELECT id FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if existing:
        conn.execute(
            "UPDATE jobs SET languages = ?, status = ? WHERE job_id = ?",
            (languages, status, job_id)
        )
    else:
        conn.execute(
            "INSERT INTO jobs (user_id, job_id, source_type, source_label, product_count, image_count, languages, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, job_id, source_type, source_label, product_count, image_count, languages, status)
        )
    conn.commit()
    conn.close()


def get_current_user():
    """Return current user dict or None."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    conn = get_db()
    user = conn.execute("SELECT id, nome, email FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if user:
        return dict(user)
    return None


def login_required_api(f):
    """Decorator for API routes that require login. Returns JSON error."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"erro": "Faça login para acessar esta funcionalidade"}), 401
        return f(*args, **kwargs)
    return decorated


def _cleanup_old_files():
    """Remove pastas de output com mais de 1 hora."""
    if not OUTPUT_FOLDER.exists():
        return
    now = time.time()
    for d in OUTPUT_FOLDER.iterdir():
        if d.is_dir() and (now - d.stat().st_mtime) > 3600:
            shutil.rmtree(d, ignore_errors=True)

IDIOMAS = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "de": "German",
    "nl": "Dutch",
}

PROMPT_TEMPLATE = """Translate ALL text in this image from Portuguese to {idioma_nome}.
{marca_instrucao}
IMPORTANT: The target language is {idioma_nome}. ALL translated text MUST be in {idioma_nome}, not in English or any other language.

RULES:
- Keep the EXACT same layout, positions, fonts, colors, sizes, and design
- Only change the text content — nothing else in the image should change
- Keep measurement units (cm, mm, kg, etc.) unchanged
- Translate ALL text labels, titles, descriptions, and disclaimers to {idioma_nome}
- Maintain the same visual hierarchy, spacing, and alignment
- The output image must look identical to the original except for the translated text
- Keep any icons, logos, and decorative elements exactly as they are
- Unless a brand replacement is specified above, keep all brand names unchanged"""


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY não configurada")
    return OpenAI(api_key=api_key)


def get_anthropic_client():
    if not ANTHROPIC_AVAILABLE:
        raise ValueError("anthropic SDK não instalado")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY não configurada")
    return anthropic.Anthropic(api_key=api_key)


def converter_imagem(imagem_bytes: bytes, formato: str = "webp") -> bytes:
    """Converte imagem para o formato escolhido (webp ou jpeg)."""
    img = Image.open(io.BytesIO(imagem_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buffer = io.BytesIO()
    if formato == "webp":
        img.save(buffer, format="WebP", quality=85, method=6)
    else:
        img.save(buffer, format="JPEG", quality=85, optimize=True, progressive=True, subsampling=0)
    return buffer.getvalue()


def _resize_to_original(img_bytes: bytes, target_size: tuple) -> bytes:
    """Redimensiona imagem traduzida para as dimensões originais (evita corte)."""
    img = Image.open(io.BytesIO(img_bytes))
    if img.size != target_size:
        logger.info(f"[Resize] {img.size} → {target_size}")
        img = img.resize(target_size, Image.LANCZOS)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _get_aspect_ratio_category(width: int, height: int) -> str:
    """Classify image aspect ratio. Returns 'chatgpt' for 1:1, 2:3, 3:2; 'gemini' for others."""
    if width == 0 or height == 0:
        return "chatgpt"

    ratio = width / height
    # Define tolerance for matching ratios
    TOLERANCE = 0.08

    targets = {
        1.0: "1:1",       # 1:1
        2 / 3: "2:3",     # 2:3 (portrait)
        3 / 2: "3:2",     # 3:2 (landscape)
    }

    for target_ratio, label in targets.items():
        if abs(ratio - target_ratio) < TOLERANCE:
            logger.info(f"[Ratio] {width}×{height} → {label} (ratio={ratio:.3f}) → ChatGPT")
            return "chatgpt"

    logger.info(f"[Ratio] {width}×{height} → ratio={ratio:.3f} → Gemini")
    return "gemini"


def traduzir_imagem_gemini(imagem_bytes: bytes, mime_type: str,
                            idioma_nome: str, marca_de: str = "", marca_para: str = "",
                            original_size: tuple = None) -> Optional[bytes]:
    """Traduz imagem via Google Gemini (Imagen) para proporções não-padrão."""
    if not GEMINI_AVAILABLE:
        logger.error("[Gemini Tradução] google-genai não instalado")
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("[Gemini Tradução] GEMINI_API_KEY não configurada")
        return None

    marca_instrucao = ""
    if marca_de and marca_para:
        marca_instrucao = f'\nBRAND REPLACEMENT: Replace the brand "{marca_de}" with "{marca_para}" everywhere it appears (text, logos, labels). Keep the same style and position.\n'

    prompt = PROMPT_TEMPLATE.format(idioma_nome=idioma_nome, marca_instrucao=marca_instrucao)

    client = genai.Client(api_key=api_key)
    img = Image.open(io.BytesIO(imagem_bytes))

    modelos = [
        "gemini-3-pro-image",
        "gemini-3.1-flash-image-preview",
        "gemini-2.5-flash-image",
        "gemini-2.0-flash-exp-image-generation",
    ]

    for modelo in modelos:
        try:
            logger.info(f"[Gemini Tradução] Tentando modelo: {modelo}")
            response = client.models.generate_content(
                model=modelo,
                contents=[prompt, img],
                config=genai_types.GenerateContentConfig(
                    response_modalities=["Text", "Image"]
                )
            )

            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        logger.info(f"[Gemini Tradução] Imagem gerada com sucesso via {modelo}")
                        result_bytes = part.inline_data.data
                        if original_size:
                            return _resize_to_original(result_bytes, original_size)
                        return result_bytes

        except Exception as e:
            logger.error(f"[Gemini Tradução] {modelo} falhou: {type(e).__name__}: {e}")
            continue

    logger.error("[Gemini Tradução] Todos os modelos falharam")
    return None


def traduzir_imagem(client: OpenAI, imagem_bytes: bytes, mime_type: str,
                     idioma_nome: str, marca_de: str = "", marca_para: str = "") -> Optional[bytes]:
    """Traduz imagem — Gemini (modelo mais avançado) como principal, ChatGPT como fallback."""

    # Capturar dimensões originais para preservar após tradução
    original_img = Image.open(io.BytesIO(imagem_bytes))
    original_size = original_img.size  # (width, height)
    logger.info(f"[Tradução] Imagem original: {original_size[0]}×{original_size[1]}")

    # Tentar Gemini primeiro (melhor qualidade, preserva aspect ratio)
    result = traduzir_imagem_gemini(
        imagem_bytes, mime_type, idioma_nome, marca_de, marca_para, original_size
    )
    if result:
        return result
    # Fallback para ChatGPT se Gemini falhar
    logger.warning("[Tradução] Gemini falhou, tentando ChatGPT como fallback...")

    marca_instrucao = ""
    if marca_de and marca_para:
        marca_instrucao = f'\nBRAND REPLACEMENT: Replace the brand "{marca_de}" with "{marca_para}" everywhere it appears (text, logos, labels). Keep the same style and position.\n'

    prompt = PROMPT_TEMPLATE.format(idioma_nome=idioma_nome, marca_instrucao=marca_instrucao)
    imagem_b64 = base64.standard_b64encode(imagem_bytes).decode("utf-8")

    # Método 1: Responses API
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:{mime_type};base64,{imagem_b64}"},
                    {"type": "input_text", "text": prompt},
                ],
            }],
            tools=[{"type": "image_generation", "size": "auto", "quality": "high"}],
        )

        for item in response.output:
            if hasattr(item, "result") and hasattr(item.result, "__iter__"):
                for result in item.result:
                    if hasattr(result, "image") and result.image:
                        img_b64 = result.image.get("b64_json") if isinstance(result.image, dict) else getattr(result.image, "b64_json", None)
                        if img_b64:
                            return _resize_to_original(base64.standard_b64decode(img_b64), original_size)
            if hasattr(item, "image") and item.image:
                img_b64 = item.image.get("b64_json") if isinstance(item.image, dict) else getattr(item.image, "b64_json", None)
                if img_b64:
                    return _resize_to_original(base64.standard_b64decode(img_b64), original_size)

        resp_dict = response.model_dump() if hasattr(response, "model_dump") else {}
        for output_item in resp_dict.get("output", []):
            if output_item.get("type") == "image_generation_call":
                result = output_item.get("result", {})
                if isinstance(result, dict) and "b64_json" in result:
                    return _resize_to_original(base64.standard_b64decode(result["b64_json"]), original_size)
            for key in ("result", "content", "image"):
                val = output_item.get(key)
                if isinstance(val, dict) and "b64_json" in val:
                    return _resize_to_original(base64.standard_b64decode(val["b64_json"]), original_size)
                if isinstance(val, list):
                    for v in val:
                        if isinstance(v, dict) and "b64_json" in v:
                            return _resize_to_original(base64.standard_b64decode(v["b64_json"]), original_size)

    except Exception as e:
        logger.error(f"[Tradução] Responses API falhou: {type(e).__name__}: {e}")

    # Método 2: Images Edit API
    try:
        img_io = io.BytesIO(imagem_bytes)
        img_io.name = "image.png"
        response = client.images.edit(
            model="gpt-image-1",
            image=img_io,
            prompt=prompt,
            size="auto",
        )
        if response.data and len(response.data) > 0:
            img_data = response.data[0]
            if hasattr(img_data, "b64_json") and img_data.b64_json:
                return _resize_to_original(base64.standard_b64decode(img_data.b64_json), original_size)
    except Exception as e:
        logger.error(f"[Tradução] Images Edit API falhou: {type(e).__name__}: {e}")

    logger.error("[Tradução] Todos os métodos falharam")
    return None


# ─── CSV Shopify Helpers ──────────────────────────────────

def parse_shopify_csv(file_stream) -> list[dict]:
    """Parse Shopify CSV para lista de dicts."""
    text = file_stream.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def extract_images_from_csv(rows: list[dict]) -> list[dict]:
    """Extrai todas as URLs de imagens do CSV (gallery, variant, description HTML)."""
    seen_urls = set()
    images = []

    for idx, row in enumerate(rows):
        handle = row.get("Handle", "product")

        # Image Src (gallery)
        img_src = (row.get("Image Src") or "").strip()
        if img_src and img_src not in seen_urls:
            seen_urls.add(img_src)
            images.append({
                "url": img_src,
                "source": "gallery",
                "handle": handle,
                "row_index": idx,
                "filename": img_src.split("/")[-1].split("?")[0],
            })

        # Variant Image
        var_img = (row.get("Variant Image") or "").strip()
        if var_img and var_img not in seen_urls:
            seen_urls.add(var_img)
            images.append({
                "url": var_img,
                "source": "variant",
                "handle": handle,
                "row_index": idx,
                "filename": var_img.split("/")[-1].split("?")[0],
            })

        # Body (HTML) — extract <img> tags
        body_html = (row.get("Body (HTML)") or "").strip()
        if body_html:
            soup = BeautifulSoup(body_html, "html.parser")
            for img_tag in soup.find_all("img"):
                src = (img_tag.get("src") or "").strip()
                if src and src.startswith("http") and src not in seen_urls:
                    seen_urls.add(src)
                    images.append({
                        "url": src,
                        "source": "description",
                        "handle": handle,
                        "row_index": idx,
                        "filename": src.split("/")[-1].split("?")[0],
                    })

    return images


def download_image(url: str) -> Optional[bytes]:
    """Download imagem de URL (Shopify CDN etc)."""
    try:
        resp = http_requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.error(f"[CSV] Erro ao baixar {url}: {e}")
        return None


def detect_text_in_image(client: OpenAI, img_bytes: bytes, mime_type: str = "image/png") -> dict:
    """Usa GPT-4o Vision para detectar se imagem tem texto traduzível."""
    img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}", "detail": "low"}},
                    {"type": "text", "text": (
                        "Analyze this image. Does it contain text that could be translated to another language? "
                        "Text includes: titles, descriptions, specifications, measurements, labels, disclaimers, etc. "
                        "Do NOT count brand names/logos or simple numbers as translatable text. "
                        "Respond with ONLY this JSON (no markdown): "
                        '{"has_text": true/false, "description": "brief description in Portuguese"}'
                    )},
                ],
            }],
            max_tokens=150,
        )
        text = response.choices[0].message.content.strip()
        # Remove markdown wrapping if present
        text = re.sub(r"^```json?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception as e:
        logger.error(f"[CSV] Erro na detecção de texto: {e}")
        return {"has_text": True, "description": "Não foi possível analisar (assumindo que tem texto)"}


def traduzir_texto_produto(client, title: str, body_html: str,
                            seo_title: str, seo_desc: str, tags: str,
                            idioma_nome: str, marca_de: str = "", marca_para: str = "",
                            options: list = None) -> dict:
    """Traduz campos textuais de um produto via Claude Sonnet (fallback GPT-4o)."""
    marca_instrucao = ""
    if marca_de and marca_para:
        marca_instrucao = f'\nBRAND REPLACEMENT: Replace ALL occurrences of the brand "{marca_de}" with "{marca_para}" in every field.\n'

    options_input = ""
    options_output = ""
    if options:
        options_json = json.dumps(options, ensure_ascii=False)
        options_input = f"\noptions: {options_json}"
        options_output = ', "options": [{"name": "translated name", "values": ["translated value1", "translated value2"]}]'

    prompt = f"""You are a top-tier e-commerce copywriter who writes ONLY in native, idiomatic {idioma_nome}. Translate this Shopify product listing.
{marca_instrucao}
CRITICAL — NATURAL LANGUAGE:
- You MUST write as if the text was originally created in {idioma_nome} by a native speaker
- NEVER use rare/uncommon words. Use everyday language that online shoppers actually use
- Adapt Portuguese idioms to equivalent {idioma_nome} expressions — do NOT translate literally
- The result must sound like professional {idioma_nome} e-commerce copy, NOT a translation

COMMON MISTAKES TO AVOID (Portuguese → Bad English → Good English):
- "eternizar" → NOT "eternalize" → USE "cherish forever" / "keep close forever" / "treasure"
- "emocionar" / "presente para emocionar" → NOT "move emotions" / "heartwarming gift" → USE "a touching gift" / "touch your heart"
- "carinho" → NOT "affection" → USE "love" / "warmth"
- "jóia" → NOT "jewel" → USE "piece of jewelry" / "jewelry"
- "não estar mais presente fisicamente" → NOT "no longer physically present" → USE "no longer with us"
- "Código de Defesa do Consumidor" → NOT "Consumer Protection Code" → ADAPT to "{idioma_nome} market equivalent" (e.g. "our satisfaction guarantee")
- "nos emocionamos" → NOT "we get emotional" → USE "we're deeply moved" / "it touches our hearts"

COMMON MISTAKES TO AVOID (Portuguese → Bad French → Good French):
- "eternizar" → NOT "immortaliser" → USE "graver à jamais" / "chérir pour toujours" / "garder précieusement"
- "emocionar" → NOT "émotionner" → USE "toucher le cœur" / "émouvoir"
- "presente para emocionar" → NOT "cadeau émouvant" → USE "un cadeau qui touche le cœur"
- "antecipar" → NOT "anticiper" → USE "préparer" / "prévoir"
- "colocar um sorriso no rosto" → NOT "mettre un sourire sur le visage" → USE "illuminer votre visage" / "vous faire sourire"
- "amores" (família) → NOT "amours" → USE "êtres chers" / "proches"
- "não descolore / não perde a cor" → NOT "ne décolore pas / ne perd pas sa couleur" → USE "ne s'oxyde pas" / "ne ternit pas"
- "melhor escolha para dar de presente" → NOT "meilleur choix pour offrir un cadeau" → USE "meilleur choix pour faire plaisir"
- "voltamos às nossas raízes" (emocional) → NOT "revenir à nos racines" → USE "replonger dans nos plus beaux souvenirs"
- "Código de Defesa do Consumidor" → NOT "Code de Défense du Consommateur" → USE "notre garantie satisfaction" / "conformément à notre garantie"

TECHNICAL RULES:
- Keep ALL HTML tags and attributes exactly as they are (only translate the visible text between tags)
- Keep brand names unchanged (unless brand replacement is specified above)
- Keep measurement units (cm, mm, kg, etc.) unchanged
- Keep product codes/SKUs unchanged
- Keep emojis in the same positions
- If a field is empty, return it as empty string
- Translate option names AND option values (e.g. "Cor" → "Color", "Ouro" → "Gold", "Prata" → "Silver")
- Return ONLY valid JSON (no markdown, no ```), with these exact keys:

{{"title": "translated title", "body_html": "translated HTML", "seo_title": "translated SEO title", "seo_description": "translated SEO description", "tags": "translated tags"{options_output}}}

Input:
title: {title}
body_html: {body_html}
seo_title: {seo_title}
seo_description: {seo_desc}
tags: {tags}{options_input}"""

    try:
        # Use Claude Sonnet as primary translator
        if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
            anthropic_client = get_anthropic_client()
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            # Track usage
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            if "user_id" in session:
                track_api_usage(
                    session["user_id"], "text_translation", "claude-sonnet-4-6",
                    tokens_input=input_tokens,
                    tokens_output=output_tokens,
                )

            text = response.content[0].text.strip()
        else:
            # Fallback to GPT-4o
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )
            usage = getattr(response, "usage", None)
            if usage and "user_id" in session:
                track_api_usage(
                    session["user_id"], "text_translation", "gpt-4o",
                    tokens_input=getattr(usage, "prompt_tokens", 0),
                    tokens_output=getattr(usage, "completion_tokens", 0),
                )
            text = response.choices[0].message.content.strip()

        text = re.sub(r"^```json?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception as e:
        logger.error(f"[CSV] Erro ao traduzir texto: {e}")
        return {}


def generate_translated_csv(rows: list[dict], mapping: dict, lang: str,
                             text_translations: dict = None) -> str:
    """Gera CSV com URLs substituídas e textos traduzidos."""
    import copy
    new_rows = copy.deepcopy(rows)
    fieldnames = list(rows[0].keys()) if rows else []

    for row in new_rows:
        handle = row.get("Handle", "")

        # Substituir Image Src
        img_src = (row.get("Image Src") or "").strip()
        if img_src in mapping:
            row["Image Src"] = mapping[img_src]

        # Substituir Variant Image
        var_img = (row.get("Variant Image") or "").strip()
        if var_img in mapping:
            row["Variant Image"] = mapping[var_img]

        # Substituir <img src> no Body HTML
        body_html = (row.get("Body (HTML)") or "").strip()
        if body_html:
            for original_url, new_path in mapping.items():
                if original_url in body_html:
                    body_html = body_html.replace(original_url, new_path)
            row["Body (HTML)"] = body_html

        # Aplicar traduções de texto (se disponíveis)
        if text_translations and handle in text_translations:
            tt = text_translations[handle]
            # Title: só aplicar na row principal (que tem Title preenchido)
            if row.get("Title") and tt.get("title"):
                row["Title"] = tt["title"]
            # Body HTML: só na row principal (que tem body preenchido)
            if row.get("Body (HTML)") and tt.get("body_html"):
                # Preservar substituições de URL já feitas
                translated_body = tt["body_html"]
                for original_url, new_path in mapping.items():
                    if original_url in translated_body:
                        translated_body = translated_body.replace(original_url, new_path)
                row["Body (HTML)"] = translated_body
            if tt.get("seo_title") and "SEO Title" in row:
                row["SEO Title"] = tt["seo_title"]
            if tt.get("seo_description") and "SEO Description" in row:
                row["SEO Description"] = tt["seo_description"]
            if tt.get("tags") and "Tags" in row and row.get("Tags"):
                row["Tags"] = tt["tags"]
            # Image Alt Text: aplicar em todas as rows do handle
            if tt.get("image_alt_text") and "Image Alt Text" in row and row.get("Image Alt Text"):
                row["Image Alt Text"] = tt["image_alt_text"]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(new_rows)
    return output.getvalue()


# ─── Rotas ─────────────────────────────────────────────────

@app.route("/")
def index():
    user = get_current_user()
    return render_template("index.html", idiomas=IDIOMAS, user=user)


@app.route("/extension/download")
def download_extension():
    """Package the Chrome extension as a zip file for download."""
    ext_dir = Path(__file__).parent / "extension"
    if not ext_dir.exists():
        return jsonify({"erro": "Extensão não encontrada"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in ext_dir.iterdir():
            if file.is_file():
                zf.write(file, arcname=file.name)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name="image-tools-extension.zip",
    )


# ─── Auth Routes ─────────────────────────────────────────

@app.route("/auth/register", methods=["POST"])
def auth_register():
    """Create a new user account."""
    data = request.get_json()
    nome = (data.get("nome") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "")

    if not nome or not email or not password:
        return jsonify({"erro": "Preencha todos os campos"}), 400

    if len(password) < 6:
        return jsonify({"erro": "Senha deve ter pelo menos 6 caracteres"}), 400

    if "@" not in email or "." not in email:
        return jsonify({"erro": "Email inválido"}), 400

    conn = get_db()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            conn.close()
            return jsonify({"erro": "Este email já está cadastrado"}), 409

        password_hash = generate_password_hash(password)
        cursor = conn.execute(
            "INSERT INTO users (nome, email, password_hash) VALUES (?, ?, ?)",
            (nome, email, password_hash)
        )
        user_id = cursor.lastrowid

        # Check if this email was invited to a team
        invite = conn.execute(
            "SELECT id, owner_id FROM team_members WHERE email = ? AND status = 'pending'",
            (email,)
        ).fetchone()
        if invite:
            conn.execute("UPDATE users SET owner_id = ?, role = 'agent' WHERE id = ?", (invite["owner_id"], user_id))
            conn.execute("UPDATE team_members SET member_id = ?, status = 'active' WHERE id = ?", (user_id, invite["id"]))

        conn.commit()
        session["user_id"] = user_id
        conn.close()

        return jsonify({"ok": True, "user": {"id": user_id, "nome": nome, "email": email}})
    except Exception as e:
        conn.close()
        logger.error(f"[Auth] Erro no registro: {e}")
        return jsonify({"erro": "Erro ao criar conta"}), 500


@app.route("/auth/invite/<token>")
def auth_invite_check(token):
    """Validate an invite token and return invite info."""
    conn = get_db()
    invite = conn.execute("""
        SELECT tm.email, tm.status, u.nome as owner_name
        FROM team_members tm
        JOIN users u ON u.id = tm.owner_id
        WHERE tm.invite_token = ?
    """, (token,)).fetchone()
    conn.close()

    if not invite:
        return jsonify({"erro": "Convite inválido ou expirado"}), 404
    if invite["status"] == "active":
        return jsonify({"erro": "Este convite já foi aceito"}), 400

    return jsonify({
        "ok": True,
        "email": invite["email"],
        "owner_name": invite["owner_name"],
    })


@app.route("/auth/login", methods=["POST"])
def auth_login():
    """Log in with email and password."""
    data = request.get_json()
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "")

    if not email or not password:
        return jsonify({"erro": "Preencha email e senha"}), 400

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"erro": "Email ou senha incorretos"}), 401

    session["user_id"] = user["id"]
    return jsonify({"ok": True, "user": {"id": user["id"], "nome": user["nome"], "email": user["email"]}})


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    """Log out current user."""
    session.clear()
    return jsonify({"ok": True})


@app.route("/auth/me")
def auth_me():
    """Return current user info + linked Shopify stores."""
    user = get_current_user()
    if not user:
        return jsonify({"logged_in": False})

    conn = get_db()
    # Agents see their owner's stores too
    owner_id = get_owner_id(user["id"])
    stores = conn.execute(
        "SELECT id, store_name, store_url FROM shopify_stores WHERE user_id = ? ORDER BY updated_at DESC",
        (owner_id,)
    ).fetchall()

    # Get user role
    user_row = conn.execute("SELECT role, owner_id FROM users WHERE id = ?", (user["id"],)).fetchone()
    conn.close()

    user_data = dict(user)
    user_data["role"] = user_row["role"] if user_row else "owner"
    user_data["is_agent"] = bool(user_row and user_row["owner_id"])

    return jsonify({
        "logged_in": True,
        "user": user_data,
        "stores": [dict(s) for s in stores],
    })


@app.route("/auth/shopify-store", methods=["POST"])
@login_required_api
def auth_save_shopify_store():
    """Save or update a Shopify store linked to the current user."""
    data = request.get_json()
    store_url = (data.get("store_url") or "").strip()
    token = (data.get("token") or "").strip()
    store_name = (data.get("store_name") or "").strip()

    if not store_url or not token:
        return jsonify({"erro": "URL e token são obrigatórios"}), 400

    store_url = store_url.replace("https://", "").replace("http://", "").rstrip("/")
    if not store_url.endswith(".myshopify.com"):
        if "." not in store_url:
            store_url = store_url + ".myshopify.com"

    # Validate token by making a test API call
    try:
        result = shopify_api_get(store_url, token, "shop.json")
        shop_info = result.get("shop", {})
        if not store_name:
            store_name = shop_info.get("name", store_url)
    except http_requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            return jsonify({"erro": "Token de acesso inválido. Verifique suas credenciais."}), 401
        elif e.response is not None and e.response.status_code == 404:
            return jsonify({"erro": "Loja não encontrada. Verifique a URL."}), 404
        return jsonify({"erro": f"Erro ao validar conexão: {e}"}), 500
    except Exception as e:
        logger.error(f"[Shopify] Erro ao validar loja: {e}")
        return jsonify({"erro": f"Erro ao validar: {e}"}), 500

    owner_id = get_owner_id(session["user_id"])
    conn = get_db()

    # Check if this store already exists for this user/owner
    existing = conn.execute(
        "SELECT id FROM shopify_stores WHERE user_id = ? AND store_url = ?",
        (owner_id, store_url)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE shopify_stores SET access_token = ?, store_name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (token, store_name or store_url, existing["id"])
        )
        store_id = existing["id"]
    else:
        cursor = conn.execute(
            "INSERT INTO shopify_stores (user_id, store_name, store_url, access_token) VALUES (?, ?, ?, ?)",
            (owner_id, store_name or store_url, store_url, token)
        )
        store_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return jsonify({"ok": True, "store_id": store_id})


# ─── Shopify OAuth ─────────────────────────────────────────

SHOPIFY_OAUTH_SCOPES = "read_products,write_products,read_themes,write_themes,read_product_listings,write_product_listings,read_publications,write_publications,read_content,write_content"


@app.route("/shopify/oauth/start", methods=["POST"])
@login_required_api
def shopify_oauth_start():
    """Initiate Shopify OAuth flow."""
    data = request.get_json()
    store_url = (data.get("store_url") or "").strip()
    client_id = (data.get("client_id") or "").strip()
    client_secret = (data.get("client_secret") or "").strip()
    store_name = (data.get("store_name") or "").strip()

    if not store_url or not client_id or not client_secret:
        return jsonify({"erro": "URL da loja, Client ID e Chave Secreta são obrigatórios"}), 400

    store_url = store_url.replace("https://", "").replace("http://", "").rstrip("/")
    if not store_url.endswith(".myshopify.com"):
        if "." not in store_url:
            store_url = store_url + ".myshopify.com"

    # Generate a unique state to prevent CSRF
    state = str(uuid.uuid4())

    # Save OAuth state in session
    session["shopify_oauth"] = {
        "store_url": store_url,
        "client_id": client_id,
        "client_secret": client_secret,
        "store_name": store_name,
        "state": state,
    }

    # Build the callback URL
    callback_url = request.host_url.rstrip("/") + "/shopify/oauth/callback"
    callback_url = callback_url.replace("http://", "https://")

    # Build Shopify OAuth authorization URL
    auth_url = (
        f"https://{store_url}/admin/oauth/authorize"
        f"?client_id={client_id}"
        f"&scope={SHOPIFY_OAUTH_SCOPES}"
        f"&redirect_uri={callback_url}"
        f"&state={state}"
    )

    return jsonify({"auth_url": auth_url})


@app.route("/shopify/oauth/callback")
def shopify_oauth_callback():
    """Handle Shopify OAuth callback — exchange code for access token."""
    code = request.args.get("code")
    state = request.args.get("state")
    shop = request.args.get("shop", "")

    oauth_data = session.get("shopify_oauth")
    if not oauth_data:
        return "<h2>Erro: sessão OAuth expirada. Volte ao app e tente novamente.</h2>", 400

    # Validate state to prevent CSRF
    if state != oauth_data.get("state"):
        return "<h2>Erro: estado OAuth inválido.</h2>", 400

    if not code:
        return "<h2>Erro: código de autorização não recebido.</h2>", 400

    store_url = oauth_data["store_url"]
    client_id = oauth_data["client_id"]
    client_secret = oauth_data["client_secret"]
    store_name = oauth_data.get("store_name", "")

    # Exchange code for access token
    try:
        token_url = f"https://{store_url}/admin/oauth/access_token"
        resp = http_requests.post(token_url, json={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
        }, timeout=30)
        resp.raise_for_status()
        token_data = resp.json()
        access_token = token_data.get("access_token", "")
        granted_scopes = token_data.get("scope", "")
        logger.info(f"[Shopify OAuth] Token obtido. Escopos concedidos: {granted_scopes}")
    except Exception as e:
        logger.error(f"[Shopify OAuth] Erro ao trocar código: {e}")
        return f"<h2>Erro ao obter token: {e}</h2>", 500

    if not access_token:
        return "<h2>Erro: token de acesso não recebido.</h2>", 500

    # Get shop name from API
    if not store_name:
        try:
            shop_data = shopify_api_get(store_url, access_token, "shop.json")
            store_name = shop_data.get("shop", {}).get("name", store_url)
        except Exception:
            store_name = store_url

    # Save the store with credentials
    owner_id = get_owner_id(session["user_id"])
    conn = get_db()

    existing = conn.execute(
        "SELECT id FROM shopify_stores WHERE user_id = ? AND store_url = ?",
        (owner_id, store_url)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE shopify_stores SET access_token = ?, store_name = ?, client_id = ?, client_secret = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (access_token, store_name, client_id, client_secret, existing["id"])
        )
    else:
        conn.execute(
            "INSERT INTO shopify_stores (user_id, store_name, store_url, access_token, client_id, client_secret) VALUES (?, ?, ?, ?, ?, ?)",
            (owner_id, store_name, store_url, access_token, client_id, client_secret)
        )

    conn.commit()
    conn.close()

    # Clear OAuth session data
    session.pop("shopify_oauth", None)

    # Redirect back to the app with success
    return redirect("/?shopify_connected=1")


@app.route("/auth/shopify-store/<int:store_id>")
@login_required_api
def auth_get_shopify_store(store_id):
    """Get Shopify store credentials (for auto-fill)."""
    owner_id = get_owner_id(session["user_id"])
    conn = get_db()
    store = conn.execute(
        "SELECT id, store_name, store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
        (store_id, owner_id)
    ).fetchone()
    conn.close()

    if not store:
        return jsonify({"erro": "Loja não encontrada"}), 404

    return jsonify({"store": dict(store)})


@app.route("/auth/shopify-store/<int:store_id>/reconnect", methods=["POST"])
@login_required_api
def auth_reconnect_shopify_store(store_id):
    """Reconnect a store via OAuth using saved credentials."""
    owner_id = get_owner_id(session["user_id"])
    conn = get_db()
    store = conn.execute(
        "SELECT id, store_url, client_id, client_secret, store_name FROM shopify_stores WHERE id = ? AND user_id = ?",
        (store_id, owner_id)
    ).fetchone()
    conn.close()

    if not store:
        return jsonify({"erro": "Loja não encontrada"}), 404

    store_url = store["store_url"]
    client_id = store["client_id"]
    client_secret = store["client_secret"]

    if not client_id or not client_secret:
        return jsonify({"erro": "Credenciais OAuth não encontradas. Remova e adicione a loja novamente."}), 400

    state = str(uuid.uuid4())
    session["shopify_oauth"] = {
        "store_url": store_url,
        "client_id": client_id,
        "client_secret": client_secret,
        "store_name": store["store_name"] or "",
        "state": state,
    }

    callback_url = request.host_url.rstrip("/") + "/shopify/oauth/callback"
    callback_url = callback_url.replace("http://", "https://")
    auth_url = (
        f"https://{store_url}/admin/oauth/authorize"
        f"?client_id={client_id}"
        f"&scope={SHOPIFY_OAUTH_SCOPES}"
        f"&redirect_uri={callback_url}"
        f"&state={state}"
    )

    return jsonify({"auth_url": auth_url})


@app.route("/auth/shopify-store/<int:store_id>/test", methods=["POST"])
@login_required_api
def auth_test_shopify_store(store_id):
    """Test store connection by making a simple API call."""
    owner_id = get_owner_id(session["user_id"])
    conn = get_db()
    store = conn.execute(
        "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
        (store_id, owner_id)
    ).fetchone()
    conn.close()

    if not store:
        return jsonify({"erro": "Loja não encontrada"}), 404

    store_url = store["store_url"]
    token = store["access_token"]

    if not token:
        return jsonify({"ok": False, "erro": "Token de acesso não configurado. Reconecte a loja."})

    try:
        result = shopify_api_get(store_url, token, "shop.json")
        shop = result.get("shop", {})
        # Also test products scope
        products_result = shopify_api_get(store_url, token, "products/count.json")
        count = products_result.get("count", 0)
        return jsonify({
            "ok": True,
            "shop_name": shop.get("name", ""),
            "plan": shop.get("plan_display_name", ""),
            "product_count": count,
            "api_version": SHOPIFY_API_VERSION,
        })
    except http_requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        body = e.response.text[:300] if e.response is not None else ""
        return jsonify({"ok": False, "erro": f"HTTP {status}: {body}"})
    except Exception as e:
        return jsonify({"ok": False, "erro": str(e)})


@app.route("/auth/shopify-store/<int:store_id>", methods=["DELETE"])
@login_required_api
def auth_delete_shopify_store(store_id):
    """Delete a linked Shopify store."""
    owner_id = get_owner_id(session["user_id"])
    conn = get_db()
    conn.execute(
        "DELETE FROM shopify_stores WHERE id = ? AND user_id = ?",
        (store_id, owner_id)
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# ─── Team Management ─────────────────────────────────────

@app.route("/team/invite", methods=["POST"])
@login_required_api
def team_invite():
    """Invite a team member by email."""
    user_id = session["user_id"]
    data = request.get_json()
    email = (data.get("email") or "").strip().lower()

    if not email:
        return jsonify({"erro": "Email é obrigatório"}), 400

    # Only owners can invite
    conn = get_db()
    user = conn.execute("SELECT id, owner_id FROM users WHERE id = ?", (user_id,)).fetchone()
    if user and user["owner_id"]:
        conn.close()
        return jsonify({"erro": "Apenas o dono da conta pode convidar membros"}), 403

    # Check if already invited
    existing = conn.execute(
        "SELECT id, status FROM team_members WHERE owner_id = ? AND email = ?",
        (user_id, email)
    ).fetchone()
    if existing:
        conn.close()
        return jsonify({"erro": "Este email já foi convidado"}), 400

    invite_token = str(uuid.uuid4())[:12]

    # Get owner name for the email
    owner_info = conn.execute("SELECT nome FROM users WHERE id = ?", (user_id,)).fetchone()
    owner_name = owner_info["nome"] if owner_info else "Sua equipe"

    # Check if user already exists
    existing_user = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    member_id = existing_user["id"] if existing_user else None
    status = "pending"

    if member_id:
        # User exists — link directly
        conn.execute("UPDATE users SET owner_id = ?, role = 'agent' WHERE id = ?", (user_id, member_id))
        status = "active"

    conn.execute(
        "INSERT INTO team_members (owner_id, member_id, email, role, status, invite_token) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, member_id, email, "agent", status, invite_token)
    )
    conn.commit()
    conn.close()

    # Send invite email (for pending invites)
    email_sent = False
    invite_url = f"{APP_URL.rstrip('/')}/?invite={invite_token}"
    if status == "pending":
        email_sent = send_invite_email(email, owner_name, invite_token)

    return jsonify({
        "ok": True,
        "status": status,
        "invite_token": invite_token,
        "email_sent": email_sent,
        "invite_url": invite_url if status == "pending" else None,
    })


@app.route("/team/members")
@login_required_api
def team_list():
    """List team members for the current owner."""
    user_id = session["user_id"]
    owner_id = get_owner_id(user_id)

    conn = get_db()
    members = conn.execute("""
        SELECT tm.id, tm.email, tm.role, tm.status, tm.created_at,
               tm.invite_token, u.nome as member_name
        FROM team_members tm
        LEFT JOIN users u ON u.id = tm.member_id
        WHERE tm.owner_id = ?
        ORDER BY tm.created_at DESC
    """, (owner_id,)).fetchall()

    # Also get owner info
    owner = conn.execute("SELECT id, nome, email FROM users WHERE id = ?", (owner_id,)).fetchone()
    conn.close()

    return jsonify({
        "owner": dict(owner) if owner else {},
        "members": [dict(m) for m in members],
    })


@app.route("/team/remove/<int:member_id>", methods=["DELETE"])
@login_required_api
def team_remove(member_id):
    """Remove a team member."""
    user_id = session["user_id"]
    conn = get_db()

    # Only owner can remove
    user = conn.execute("SELECT id, owner_id FROM users WHERE id = ?", (user_id,)).fetchone()
    if user and user["owner_id"]:
        conn.close()
        return jsonify({"erro": "Apenas o dono da conta pode remover membros"}), 403

    tm = conn.execute(
        "SELECT member_id FROM team_members WHERE id = ? AND owner_id = ?",
        (member_id, user_id)
    ).fetchone()

    if tm and tm["member_id"]:
        conn.execute("UPDATE users SET owner_id = NULL, role = 'owner' WHERE id = ?", (tm["member_id"],))

    conn.execute("DELETE FROM team_members WHERE id = ? AND owner_id = ?", (member_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# ─── API Usage Dashboard ─────────────────────────────────

@app.route("/usage/summary")
@login_required_api
def usage_summary():
    """Get API usage summary for the current account."""
    user_id = session["user_id"]
    owner_id = get_owner_id(user_id)

    conn = get_db()

    # Today
    today = conn.execute("""
        SELECT call_type, SUM(tokens_input) as ti, SUM(tokens_output) as to_,
               SUM(images_count) as imgs, SUM(estimated_cost) as cost, COUNT(*) as calls
        FROM api_usage WHERE owner_id = ? AND date(created_at) = date('now')
        GROUP BY call_type
    """, (owner_id,)).fetchall()

    # This month
    month = conn.execute("""
        SELECT call_type, SUM(tokens_input) as ti, SUM(tokens_output) as to_,
               SUM(images_count) as imgs, SUM(estimated_cost) as cost, COUNT(*) as calls
        FROM api_usage WHERE owner_id = ? AND created_at >= date('now', 'start of month')
        GROUP BY call_type
    """, (owner_id,)).fetchall()

    # All time
    total = conn.execute("""
        SELECT call_type, SUM(tokens_input) as ti, SUM(tokens_output) as to_,
               SUM(images_count) as imgs, SUM(estimated_cost) as cost, COUNT(*) as calls
        FROM api_usage WHERE owner_id = ?
        GROUP BY call_type
    """, (owner_id,)).fetchall()

    # Per user breakdown
    per_user = conn.execute("""
        SELECT u.nome, u.email, au.call_type,
               SUM(au.tokens_input) as ti, SUM(au.tokens_output) as to_,
               SUM(au.images_count) as imgs, SUM(au.estimated_cost) as cost, COUNT(*) as calls
        FROM api_usage au JOIN users u ON u.id = au.user_id
        WHERE au.owner_id = ?
        GROUP BY au.user_id, au.call_type
        ORDER BY cost DESC
    """, (owner_id,)).fetchall()

    conn.close()

    def rows_to_dict(rows):
        result = {"text_translation": {}, "image_translation": {}, "text_detection": {}}
        for r in rows:
            ct = r["call_type"]
            result[ct] = {
                "tokens_input": r["ti"] or 0,
                "tokens_output": r["to_"] or 0,
                "images": r["imgs"] or 0,
                "cost": round(r["cost"] or 0, 4),
                "calls": r["calls"] or 0,
            }
        return result

    return jsonify({
        "today": rows_to_dict(today),
        "month": rows_to_dict(month),
        "total": rows_to_dict(total),
        "per_user": [dict(r) for r in per_user],
    })


@app.route("/traduzir", methods=["POST"])
def traduzir():
    """Recebe uma ou mais imagens + config e retorna imagens traduzidas."""
    _cleanup_old_files()

    # Suporte a múltiplas imagens: campo "imagens" (novo) ou "imagem" (legado)
    arquivos = request.files.getlist("imagens")
    if not arquivos or all(a.filename == "" for a in arquivos):
        # Fallback para campo singular (compatibilidade)
        if "imagem" in request.files and request.files["imagem"].filename:
            arquivos = [request.files["imagem"]]
        else:
            return jsonify({"erro": "Nenhuma imagem enviada"}), 400

    # Ler configurações
    idiomas_selecionados = request.form.getlist("idiomas")
    marca_de = request.form.get("marca_de", "").strip()
    marca_para = request.form.get("marca_para", "").strip()
    formato_saida = request.form.get("formato", "webp")  # "webp" ou "jpeg"

    if not idiomas_selecionados:
        return jsonify({"erro": "Selecione pelo menos um idioma"}), 400

    # Salvar original
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_FOLDER / job_id
    job_dir.mkdir(exist_ok=True)

    # Traduzir
    try:
        client = get_client()
    except ValueError as e:
        return jsonify({"erro": str(e)}), 500

    resultados = []
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

    for arquivo in arquivos:
        imagem_bytes = arquivo.read()
        ext = Path(arquivo.filename).suffix.lower()
        mime_type = mime_types.get(ext, "image/png")
        nome_base = Path(arquivo.filename).stem

        for codigo in idiomas_selecionados:
            nome_idioma = IDIOMAS.get(codigo, codigo)

            resultado = traduzir_imagem(
                client, imagem_bytes, mime_type,
                nome_idioma, marca_de, marca_para
            )

            if resultado:
                resultado_conv = converter_imagem(resultado, formato_saida)
                ext_saida = "webp" if formato_saida == "webp" else "jpg"
                mime_saida = "image/webp" if formato_saida == "webp" else "image/jpeg"
                nome_saida = f"{nome_base}_{codigo}.{ext_saida}"
                caminho_saida = job_dir / nome_saida

                with open(caminho_saida, "wb") as f:
                    f.write(resultado_conv)

                preview_b64 = base64.standard_b64encode(resultado_conv).decode("utf-8")

                resultados.append({
                    "idioma": codigo,
                    "idioma_nome": nome_idioma,
                    "imagem_original": arquivo.filename,
                    "arquivo": nome_saida,
                    "formato": formato_saida.upper(),
                    "tamanho_kb": round(len(resultado_conv) / 1024),
                    "preview": f"data:{mime_saida};base64,{preview_b64}",
                    "download_url": f"/download/{job_id}/{nome_saida}",
                })
            else:
                resultados.append({
                    "idioma": codigo,
                    "idioma_nome": nome_idioma,
                    "imagem_original": arquivo.filename,
                    "erro": "Falha na tradução",
                })

            # Rate limiting
            time.sleep(2)

    return jsonify({"job_id": job_id, "resultados": resultados})


SUBSTITUIR_PROMPT = """FIRST IMAGE: A cropped close-up of a product/accessory. Note every design detail: shape, color, material, pattern, frame style, lens color, logos, decorative elements.

SECOND IMAGE: A person wearing a similar type of product. This is the photo to edit.

TASK: Replace ONLY the product/accessory on the person in the SECOND image with an exact replica of the product from the FIRST image.

ABSOLUTE RULES:
- The person must remain 100% identical: same face, jawline, eyes, nose, skin tone, hair, expression, pose, clothes
- The background and lighting must not change at all
- Do NOT blend or mix facial features from the first image into the result
- The replaced product must faithfully match the first image's design details
- Keep the same camera angle, framing, and composition from the SECOND image

Output a single photorealistic image."""


def _detectar_e_recortar_produto(client, img_pil: Image.Image) -> Image.Image:
    """Usa Gemini para detectar e recortar apenas o produto/acessório da imagem."""
    try:
        print("[Gemini] Detectando produto na Image A...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "Look at this image. Find the main product/accessory the person is wearing "
                "(e.g. sunglasses, watch, hat, necklace, earrings, bag, etc). "
                "Return ONLY the bounding box as 4 numbers in this exact format: x1,y1,x2,y2 "
                "where values are percentages (0-100) of image width/height. "
                "x1,y1 = top-left corner. x2,y2 = bottom-right corner. "
                "Example: 25,10,75,45",
                img_pil,
            ],
        )

        coords_text = response.text.strip()
        print(f"[Gemini] Coordenadas detectadas: {coords_text}")

        # Extrair números do texto
        import re
        nums = re.findall(r"[\d.]+", coords_text)
        if len(nums) >= 4:
            x1_pct, y1_pct, x2_pct, y2_pct = float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])

            w, h = img_pil.size
            # Converter percentagens para pixels
            x1 = int(w * x1_pct / 100)
            y1 = int(h * y1_pct / 100)
            x2 = int(w * x2_pct / 100)
            y2 = int(h * y2_pct / 100)

            # Margem de 15%
            margin_x = int((x2 - x1) * 0.15)
            margin_y = int((y2 - y1) * 0.15)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)

            cropped = img_pil.crop((x1, y1, x2, y2))
            print(f"[Gemini] Produto recortado: {cropped.size} (de {img_pil.size})")
            return cropped
        else:
            print(f"[Gemini] Não foi possível extrair coordenadas, usando imagem completa")
            return img_pil

    except Exception as e:
        print(f"[Gemini] Erro ao detectar produto: {e}, usando imagem completa")
        return img_pil


def substituir_produto_gemini(bytes_cena: bytes, bytes_produto: bytes, instrucoes_extras: str = "") -> Optional[bytes]:
    """Substitui produto na imagem usando Google Gemini.
    bytes_cena = Image A (produto de referência)
    bytes_produto = Image B (cena/pessoa para editar)
    """
    if not GEMINI_AVAILABLE:
        raise ValueError("google-genai não instalado. Execute: pip install google-genai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não configurada")

    client = genai.Client(api_key=api_key)

    # Image A = produto de referência → recortar apenas o produto
    img_ref_completa = Image.open(io.BytesIO(bytes_cena))
    img_produto_crop = _detectar_e_recortar_produto(client, img_ref_completa)

    # Image B = cena/pessoa para editar → manter completa
    img_cena = Image.open(io.BytesIO(bytes_produto))

    prompt = SUBSTITUIR_PROMPT
    if instrucoes_extras:
        prompt += f"\n\nADDITIONAL INSTRUCTIONS: {instrucoes_extras}"

    modelos = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
    ]

    ultimo_erro = ""
    for modelo in modelos:
        try:
            print(f"[Gemini] Tentando modelo: {modelo}")
            response = client.models.generate_content(
                model=modelo,
                contents=[prompt, img_produto_crop, img_cena],
                config=genai_types.GenerateContentConfig(
                    response_modalities=["Text", "Image"]
                )
            )

            print(f"[Gemini] Resposta recebida de {modelo}")
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        print(f"[Gemini] Imagem gerada com sucesso via {modelo}")
                        return part.inline_data.data
                    if hasattr(part, "text") and part.text:
                        print(f"[Gemini] Texto na resposta: {part.text[:200]}")
            else:
                print(f"[Gemini] Sem candidates na resposta")
                if hasattr(response, "prompt_feedback"):
                    print(f"[Gemini] Feedback: {response.prompt_feedback}")

        except Exception as e:
            erro_str = str(e)
            print(f"[Gemini] Erro com {modelo}: {type(e).__name__}: {erro_str[:300]}")
            if "RESOURCE_EXHAUSTED" in erro_str or "429" in erro_str:
                ultimo_erro = "quota_exceeded"
            else:
                ultimo_erro = erro_str[:200]

    if ultimo_erro == "quota_exceeded":
        raise ValueError("Quota da API Gemini esgotada. Ative o billing em https://ai.google.dev ou aguarde o reset diário.")

    return None


@app.route("/substituir", methods=["POST"])
def substituir():
    """Recebe 2 imagens e retorna produto substituído via Gemini."""
    _cleanup_old_files()

    if "imagem_cena" not in request.files or "imagem_produto" not in request.files:
        return jsonify({"erro": "Envie as duas imagens (cena + produto)"}), 400

    arquivo_cena = request.files["imagem_cena"]
    arquivo_produto = request.files["imagem_produto"]

    if arquivo_cena.filename == "" or arquivo_produto.filename == "":
        return jsonify({"erro": "Selecione ambas as imagens"}), 400

    instrucoes_extras = request.form.get("instrucoes", "").strip()
    formato_saida = request.form.get("formato", "webp")

    bytes_cena = arquivo_cena.read()
    bytes_produto = arquivo_produto.read()

    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_FOLDER / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        resultado = substituir_produto_gemini(bytes_cena, bytes_produto, instrucoes_extras)
    except ValueError as e:
        return jsonify({"erro": str(e)}), 500

    if resultado:
        resultado_conv = converter_imagem(resultado, formato_saida)
        ext_saida = "webp" if formato_saida == "webp" else "jpg"
        mime_saida = "image/webp" if formato_saida == "webp" else "image/jpeg"
        nome_saida = f"substituido_{job_id}.{ext_saida}"
        caminho_saida = job_dir / nome_saida

        with open(caminho_saida, "wb") as f:
            f.write(resultado_conv)

        preview_b64 = base64.standard_b64encode(resultado_conv).decode("utf-8")

        return jsonify({
            "job_id": job_id,
            "preview": f"data:{mime_saida};base64,{preview_b64}",
            "download_url": f"/download/{job_id}/{nome_saida}",
            "tamanho_kb": round(len(resultado_conv) / 1024),
            "formato": formato_saida.upper(),
        })
    else:
        return jsonify({"erro": "Falha na substituição. Tente novamente."}), 500


@app.route("/download/<job_id>/<nome_arquivo>")
def download(job_id, nome_arquivo):
    """Download de imagem traduzida."""
    caminho = OUTPUT_FOLDER / job_id / nome_arquivo
    if not caminho.exists():
        return jsonify({"erro": "Arquivo não encontrado"}), 404
    return send_file(str(caminho), as_attachment=True, download_name=nome_arquivo)


# ─── Shopify API Helpers ─────────────────────────────────

SHOPIFY_API_VERSION = "2026-04"

# Language code → Shopify locale mapping
LANG_TO_SHOPIFY_LOCALE = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it",
    "nl": "nl", "pt": "pt-BR", "pt-br": "pt-BR", "ja": "ja", "ko": "ko",
    "zh": "zh-CN", "ar": "ar", "ru": "ru", "pl": "pl", "sv": "sv",
    "da": "da", "no": "nb", "fi": "fi", "tr": "tr", "he": "he",
    "th": "th", "cs": "cs", "hu": "hu", "ro": "ro", "uk": "uk",
}


def shopify_graphql(store_url: str, token: str, query: str, variables: dict = None) -> dict:
    """Execute a GraphQL query against Shopify Admin API."""
    url = f"https://{store_url}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
    headers = {"X-Shopify-Access-Token": token, "Content-Type": "application/json"}
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = http_requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Surface top-level GraphQL errors (e.g. missing scope, throttling) so callers
    # don't silently see an empty result and report "nothing to translate".
    if isinstance(data, dict) and data.get("errors"):
        msgs = "; ".join(e.get("message", "") for e in data["errors"] if isinstance(e, dict))
        logger.error(f"[Shopify GraphQL] {msgs} | query={query[:80]}...")
        raise RuntimeError(f"Shopify GraphQL: {msgs}")
    return data


def shopify_register_translation(store_url: str, token: str, resource_gid: str, locale: str, translations: list[dict]) -> dict:
    """Register translations for a Shopify resource using the Translations API.

    translations: list of {"key": "title", "value": "Translated Title", "translatableContentDigest": "..."}
    """
    mutation = """
    mutation translationsRegister($resourceId: ID!, $translations: [TranslationInput!]!) {
        translationsRegister(resourceId: $resourceId, translations: $translations) {
            userErrors {
                field
                message
            }
            translations {
                key
                value
                locale
            }
        }
    }
    """
    variables = {
        "resourceId": resource_gid,
        "translations": [
            {
                "key": t["key"],
                "value": t["value"],
                "locale": locale,
                "translatableContentDigest": t["digest"],
            }
            for t in translations
        ],
    }
    return shopify_graphql(store_url, token, mutation, variables)


def shopify_get_translatable_content(store_url: str, token: str, resource_gid: str) -> list[dict]:
    """Get translatable content and digests for a resource."""
    query = """
    query translatableResource($resourceId: ID!) {
        translatableResource(resourceId: $resourceId) {
            resourceId
            translatableContent {
                key
                value
                digest
                locale
            }
        }
    }
    """
    result = shopify_graphql(store_url, token, query, {"resourceId": resource_gid})
    resource = result.get("data", {}).get("translatableResource")
    if resource:
        return resource.get("translatableContent", [])
    return []


def shopify_api_get(store_url: str, token: str, endpoint: str, params: dict = None) -> dict:
    """GET request to Shopify Admin API."""
    url = f"https://{store_url}/admin/api/{SHOPIFY_API_VERSION}/{endpoint}"
    headers = {"X-Shopify-Access-Token": token, "Content-Type": "application/json"}
    resp = http_requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def shopify_api_put(store_url: str, token: str, endpoint: str, data: dict) -> dict:
    """PUT request to Shopify Admin API."""
    url = f"https://{store_url}/admin/api/{SHOPIFY_API_VERSION}/{endpoint}"
    headers = {"X-Shopify-Access-Token": token, "Content-Type": "application/json"}
    resp = http_requests.put(url, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json()


def shopify_products_to_csv_rows(products: list[dict]) -> list[dict]:
    """Convert Shopify API product objects to CSV-like row dicts (same format as parse_shopify_csv)."""
    rows = []
    for product in products:
        handle = product.get("handle", "")
        title = product.get("title", "")
        body_html = product.get("body_html", "")
        tags = product.get("tags", "")
        seo_title = product.get("metafields_global_title_tag", "") or ""
        seo_desc = product.get("metafields_global_description_tag", "") or ""

        # Extract SEO from metafields if available
        if not seo_title:
            seo_title = product.get("title", "")

        variants = product.get("variants", [])
        images = product.get("images", [])

        # First row: main product info + first image + first variant
        first_variant = variants[0] if variants else {}
        first_image = images[0] if images else {}

        base_row = {
            "Handle": handle,
            "Title": title,
            "Body (HTML)": body_html,
            "Tags": tags,
            "SEO Title": seo_title,
            "SEO Description": seo_desc,
            "Image Src": first_image.get("src", ""),
            "Image Alt Text": first_image.get("alt", "") or "",
            "Image Position": str(first_image.get("position", "1")),
            "Variant Image": "",
            "Variant SKU": first_variant.get("sku", ""),
            "Variant Price": str(first_variant.get("price", "")),
            "Variant Inventory Qty": str(first_variant.get("inventory_quantity", "")),
            "Option1 Name": (product.get("options", [{}])[0].get("name", "") if product.get("options") else ""),
            "Option1 Value": first_variant.get("option1", "") or "",
            "_shopify_product_id": str(product.get("id", "")),
        }

        # Variant image
        if first_variant.get("image_id") and images:
            for img in images:
                if img.get("id") == first_variant.get("image_id"):
                    base_row["Variant Image"] = img.get("src", "")
                    break

        rows.append(base_row)

        # Additional images (rows with only Handle + Image Src)
        for img in images[1:]:
            rows.append({
                "Handle": handle,
                "Title": "",
                "Body (HTML)": "",
                "Tags": "",
                "SEO Title": "",
                "SEO Description": "",
                "Image Src": img.get("src", ""),
                "Image Alt Text": img.get("alt", "") or "",
                "Image Position": str(img.get("position", "")),
                "Variant Image": "",
                "Variant SKU": "",
                "Variant Price": "",
                "Variant Inventory Qty": "",
                "Option1 Name": "",
                "Option1 Value": "",
                "_shopify_product_id": str(product.get("id", "")),
            })

        # Additional variants (rows with Handle + variant info)
        for variant in variants[1:]:
            var_row = {
                "Handle": handle,
                "Title": "",
                "Body (HTML)": "",
                "Tags": "",
                "SEO Title": "",
                "SEO Description": "",
                "Image Src": "",
                "Image Alt Text": "",
                "Image Position": "",
                "Variant Image": "",
                "Variant SKU": variant.get("sku", ""),
                "Variant Price": str(variant.get("price", "")),
                "Variant Inventory Qty": str(variant.get("inventory_quantity", "")),
                "Option1 Name": "",
                "Option1 Value": variant.get("option1", "") or "",
                "_shopify_product_id": str(product.get("id", "")),
            }
            # Check if variant has its own image
            if variant.get("image_id") and images:
                for img in images:
                    if img.get("id") == variant.get("image_id"):
                        var_row["Variant Image"] = img.get("src", "")
                        break
            rows.append(var_row)

    return rows


# ─── Generic Product Scraper ─────────────────────────────

def _scrape_product_from_page(page_url: str) -> Optional[dict]:
    """Scrape product data from any e-commerce page using structured data and HTML parsing."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    try:
        from urllib.parse import urljoin, urlparse

        # Use cloudscraper to bypass anti-bot challenges (Cloudflare, etc.)
        if CLOUDSCRAPER_AVAILABLE:
            sess = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows"})
        else:
            sess = http_requests.Session()

        resp = sess.get(page_url, headers=headers, timeout=30, allow_redirects=True)

        # Handle challenge pages (POST-based anti-bot) as fallback
        if "challenge" in resp.text.lower() and resp.status_code == 200 and len(resp.text) < 3000:
            soup_challenge = BeautifulSoup(resp.text, "html.parser")
            form = soup_challenge.find("form")
            if form:
                action = form.get("action", page_url)
                if not action.startswith("http"):
                    action = urljoin(page_url, action)
                form_data = {}
                for inp in form.find_all("input"):
                    name = inp.get("name")
                    if name:
                        form_data[name] = inp.get("value", "")
                if "client_data" in form_data:
                    form_data["client_data"] = json.dumps({"time": 500, "ua": headers["User-Agent"], "w": 1920, "h": 1080, "lang": "pt-BR", "cores": 8})
                resp = sess.post(action, data=form_data, headers=headers, timeout=30, allow_redirects=True)

        if resp.status_code != 200:
            logger.warning(f"[Scraper] {page_url} returned {resp.status_code}")
            return None

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        product = {}

        # 1. Try JSON-LD structured data (most reliable)
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                ld = json.loads(script.string)
                # Handle @graph arrays
                items = ld if isinstance(ld, list) else ld.get("@graph", [ld])
                for item in items:
                    if item.get("@type") in ("Product", "IndividualProduct"):
                        product["title"] = item.get("name", "")
                        product["body_html"] = item.get("description", "")
                        imgs = item.get("image", [])
                        if isinstance(imgs, str):
                            imgs = [imgs]
                        elif isinstance(imgs, dict):
                            imgs = [imgs.get("url", "")]
                        product["images"] = [{"src": img} for img in imgs if img]

                        # Variants from offers
                        offers = item.get("offers", {})
                        if isinstance(offers, dict):
                            offers = [offers]
                        if isinstance(offers, list):
                            product["variants"] = []
                            for offer in offers:
                                variant = {
                                    "price": str(offer.get("price", "")),
                                    "option1": offer.get("name", ""),
                                }
                                product["variants"].append(variant)
                        break
            except (json.JSONDecodeError, TypeError):
                continue

        # 2. Fallback: Open Graph / meta tags
        if not product.get("title"):
            og_title = soup.find("meta", property="og:title")
            product["title"] = og_title["content"] if og_title and og_title.get("content") else ""
        if not product.get("title"):
            title_tag = soup.find("title")
            product["title"] = title_tag.text.strip() if title_tag else ""

        if not product.get("body_html"):
            og_desc = soup.find("meta", property="og:description")
            product["body_html"] = og_desc["content"] if og_desc and og_desc.get("content") else ""
        if not product.get("body_html"):
            meta_desc = soup.find("meta", attrs={"name": "description"})
            product["body_html"] = meta_desc["content"] if meta_desc and meta_desc.get("content") else ""

        # Try to find rich description in common e-commerce selectors
        if not product.get("body_html") or len(product.get("body_html", "")) < 50:
            desc_selectors = [
                ".product-description", ".product__description", "#product-description",
                "[data-product-description]", ".description", ".product-single__description",
                ".woocommerce-product-details__short-description", ".product_description",
                ".product-info-description", "#tab-description",
            ]
            for sel in desc_selectors:
                desc_el = soup.select_one(sel)
                if desc_el and len(desc_el.get_text(strip=True)) > 20:
                    product["body_html"] = str(desc_el)
                    break

        # Images from OG or page
        if not product.get("images"):
            og_img = soup.find("meta", property="og:image")
            if og_img and og_img.get("content"):
                product["images"] = [{"src": og_img["content"]}]

        # Find more product images
        if not product.get("images") or len(product.get("images", [])) < 2:
            existing = {img["src"] for img in product.get("images", [])}
            img_selectors = [
                ".product-gallery img", ".product__media img", ".product-images img",
                ".product-single__photo img", "[data-product-image]", ".product-image img",
                ".woocommerce-product-gallery img", ".product-thumbs img",
            ]
            for sel in img_selectors:
                for img_el in soup.select(sel):
                    src = img_el.get("src") or img_el.get("data-src") or img_el.get("data-lazy-src") or ""
                    if src and src not in existing and not src.endswith(".svg"):
                        if not src.startswith("http"):
                            src = urljoin(page_url, src)
                        product.setdefault("images", []).append({"src": src})
                        existing.add(src)

        # Generate a handle from the URL
        parsed = urlparse(page_url)
        path = parsed.path.strip("/").split("/")[-1] if parsed.path else ""
        product["handle"] = path or product.get("title", "").lower().replace(" ", "-")[:80]

        # Set defaults
        product.setdefault("tags", "")
        product.setdefault("vendor", parsed.netloc)
        product.setdefault("product_type", "")
        product.setdefault("variants", [{"price": "", "option1": "Default"}])

        if not product.get("title"):
            return None

        logger.info(f"[Scraper] Extracted product: {product['title'][:60]} with {len(product.get('images', []))} images")
        return product

    except Exception as e:
        logger.error(f"[Scraper] Failed to scrape {page_url}: {e}")
        return None


def _is_shopify_store(domain: str) -> bool:
    """Quick check if a domain is a Shopify store."""
    if ".myshopify.com" in domain:
        return True
    try:
        resp = http_requests.get(f"https://{domain}/products.json?limit=1", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def _try_woocommerce_api(domain: str, product_slug: str = "") -> list[dict]:
    """Try to fetch products from WooCommerce Store API (no auth needed)."""
    products = []
    try:
        if product_slug:
            url = f"https://{domain}/wp-json/wc/store/v1/products?slug={product_slug}"
        else:
            url = f"https://{domain}/wp-json/wc/store/v1/products?per_page=100"
        resp = http_requests.get(url, timeout=15)
        if resp.status_code != 200:
            return []

        wc_products = resp.json()
        if not isinstance(wc_products, list):
            return []

        for wc in wc_products:
            product = {
                "title": wc.get("name", ""),
                "handle": wc.get("slug", ""),
                "body_html": wc.get("description", "") or wc.get("short_description", ""),
                "tags": ", ".join(t.get("name", "") for t in wc.get("tags", [])),
                "vendor": domain,
                "product_type": ", ".join(c.get("name", "") for c in wc.get("categories", [])),
                "images": [{"src": img.get("src", "")} for img in wc.get("images", []) if img.get("src")],
                "variants": [],
            }
            # Price
            prices = wc.get("prices", {})
            price = prices.get("price", "0")
            # WooCommerce returns price in cents
            try:
                price_val = str(int(price) / 100) if price.isdigit() and int(price) > 100 else price
            except (ValueError, TypeError):
                price_val = price
            product["variants"].append({
                "price": price_val,
                "option1": "Default",
            })
            products.append(product)
    except Exception as e:
        logger.debug(f"[WooCommerce] API failed for {domain}: {e}")
    return products


# ─── Rotas Shopify API ───────────────────────────────────

def _clean_store_url(url: str) -> str:
    """Normalize a Shopify store URL."""
    url = url.replace("https://", "").replace("http://", "").rstrip("/")
    # Remove /products, /collections etc paths
    url = url.split("/")[0]
    if not url.endswith(".myshopify.com"):
        if "." not in url:
            url = url + ".myshopify.com"
    return url


def _extract_product_handle(raw_url: str) -> Optional[str]:
    """Extract a product handle from a Shopify product URL, or return None."""
    raw_url = raw_url.replace("https://", "").replace("http://", "").rstrip("/")
    # Match patterns like: store.com/products/handle or store.com/collections/xxx/products/handle
    m = re.search(r"/products/([^/?#]+)", "/" + raw_url)
    if m:
        return m.group(1)
    # Match direct path like: store.com/product-slug (no /products/ prefix)
    # Only if there's a single path segment (not /collections/ etc.)
    parts = raw_url.split("/", 1)
    if len(parts) == 2 and parts[1]:
        path = parts[1].strip("/")
        # Single segment or ends with a hash-like suffix — likely a product page
        if "/" not in path and path and not path.startswith(("collections", "pages", "blogs", "cart", "account", "admin", "search")):
            return path
    return None


@app.route("/shopify/copiar", methods=["POST"])
@login_required_api
def shopify_copiar():
    """Scrape products from a public e-commerce store and prepare for translation."""
    data = request.get_json()
    source_url = (data.get("source_url") or "").strip()
    dest_store_id = data.get("dest_store_id")

    if not source_url:
        return jsonify({"erro": "URL da loja fonte é obrigatória"}), 400

    # Normalize URL
    if not source_url.startswith("http"):
        source_url = "https://" + source_url

    # Detect if the URL points to a specific product
    product_handle = _extract_product_handle(source_url)
    store_domain = _clean_store_url(source_url)

    # Check if it's a Shopify store
    is_shopify = _is_shopify_store(store_domain)

    # Load destination store credentials
    dest_store_url = ""
    dest_token = ""
    owner_id = get_owner_id(session["user_id"])
    if dest_store_id:
        conn = get_db()
        store = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
            (dest_store_id, owner_id)
        ).fetchone()
        conn.close()
        if store:
            dest_store_url = store["store_url"]
            dest_token = store["access_token"]

    # Fallback: if no dest store selected, try first saved store
    if not dest_store_url or not dest_token:
        conn = get_db()
        fallback = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE user_id = ? AND access_token IS NOT NULL AND access_token != '' ORDER BY updated_at DESC LIMIT 1",
            (owner_id,)
        ).fetchone()
        conn.close()
        if fallback:
            dest_store_url = fallback["store_url"]
            dest_token = fallback["access_token"]

    # Fetch products
    all_products = []

    if is_shopify:
        # ── Shopify store: use JSON API ──
        try:
            if product_handle:
                url = f"https://{store_domain}/products/{product_handle}.json"
                resp = http_requests.get(url, timeout=30)
                if resp.status_code == 200:
                    product = resp.json().get("product")
                    if product:
                        all_products.append(product)
                else:
                    logger.info(f"[Copiar] /products/{product_handle}.json returned {resp.status_code}, trying page scrape")
                    page_url = f"https://{store_domain}/{product_handle}"
                    page_resp = http_requests.get(page_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                    if page_resp.status_code == 200:
                        page_text = page_resp.text
                        handle_match = re.search(r'/products/([a-zA-Z0-9_-]+)(?:\.js|\.json|")', page_text)
                        if handle_match:
                            real_handle = handle_match.group(1)
                            url2 = f"https://{store_domain}/products/{real_handle}.json"
                            resp2 = http_requests.get(url2, timeout=30)
                            if resp2.status_code == 200:
                                product = resp2.json().get("product")
                                if product:
                                    all_products.append(product)
                        if not all_products:
                            js_url = f"https://{store_domain}/products/{product_handle}.js"
                            js_resp = http_requests.get(js_url, timeout=30)
                            if js_resp.status_code == 200:
                                product = js_resp.json()
                                if product and product.get("title"):
                                    all_products.append(product)
            else:
                page = 1
                while True:
                    url = f"https://{store_domain}/products.json?limit=250&page={page}"
                    resp = http_requests.get(url, timeout=30)
                    resp.raise_for_status()
                    products = resp.json().get("products", [])
                    all_products.extend(products)
                    if len(products) < 250:
                        break
                    page += 1
                    if page > 10:
                        break
        except Exception as e:
            logger.warning(f"[Copiar] Shopify API failed: {e}")

    # ── Try WooCommerce Store API ──
    if not all_products:
        logger.info(f"[Copiar] Trying WooCommerce API for {store_domain}")
        wc_products = _try_woocommerce_api(store_domain, product_handle or "")
        all_products.extend(wc_products)

    # ── Fallback: Generic HTML scraping for any platform ──
    if not all_products:
        logger.info(f"[Copiar] Trying generic scraper for {source_url}")
        scraped = _scrape_product_from_page(source_url)
        if scraped:
            all_products.append(scraped)

    if not all_products:
        return jsonify({
            "erro": "Não foi possível extrair o produto automaticamente. Esta loja tem proteção anti-bot. Use a entrada manual abaixo.",
            "manual_input": True,
        }), 404

    # Convert to CSV-like rows
    rows = shopify_products_to_csv_rows(all_products)
    images = extract_images_from_csv(rows)

    # Create job directory
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_FOLDER / job_id
    job_dir.mkdir(exist_ok=True)

    handles = set(r.get("Handle", "") for r in rows if r.get("Handle"))
    product_count = len(handles)

    # Download images and detect text
    try:
        client = get_client()
    except ValueError as e:
        return jsonify({"erro": str(e)}), 500

    result_images = []
    for i, img_info in enumerate(images):
        logger.info(f"[Shopify Copiar] Baixando imagem {i+1}/{len(images)}: {img_info['filename']}")

        img_bytes = download_image(img_info["url"])
        if not img_bytes:
            result_images.append({
                "index": i, "url": img_info["url"], "source": img_info["source"],
                "handle": img_info["handle"], "filename": img_info["filename"],
                "has_text": False, "description": "Erro ao baixar imagem", "error": True,
            })
            continue

        img_path = job_dir / f"original_{i}_{img_info['filename']}"
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        ext = Path(img_info["filename"]).suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "image/png")
        detection = detect_text_in_image(client, img_bytes, mime)

        try:
            thumb = Image.open(io.BytesIO(img_bytes))
            thumb.thumbnail((120, 120))
            if thumb.mode in ("RGBA", "P"):
                thumb = thumb.convert("RGB")
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=70)
            thumb_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            thumb_b64 = ""

        result_images.append({
            "index": i, "url": img_info["url"], "source": img_info["source"],
            "handle": img_info["handle"], "filename": img_info["filename"],
            "has_text": detection.get("has_text", False),
            "description": detection.get("description", ""),
            "thumbnail": f"data:image/jpeg;base64,{thumb_b64}" if thumb_b64 else "",
        })
        time.sleep(0.5)

    # Save state — destination store for publish, source products for data
    state = {
        "rows": rows,
        "images": [img for img in images],
        "product_count": product_count,
        "source_url": store_domain,
        "shopify_store_url": dest_store_url,
        "shopify_token": dest_token,
        "shopify_products": all_products,
        "dest_store_id": dest_store_id,
    }
    with open(job_dir / "state.json", "w") as f:
        json.dump(state, f, ensure_ascii=False)

    # Save job record
    if "user_id" in session:
        source_label = store_domain
        if product_handle:
            source_label = f"{store_domain} / {product_handle}"
        save_job_record(session["user_id"], job_id, "shopify_copy", source_label, product_count, len(images))

    return jsonify({
        "job_id": job_id,
        "product_count": product_count,
        "total_images": len(images),
        "images": result_images,
        "source": "shopify_public",
    })


@app.route("/shopify/manual-product", methods=["POST"])
@login_required_api
def shopify_manual_product():
    """Handle manually entered product data when auto-scraping fails."""
    data = request.get_json()
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    image_urls = data.get("image_urls") or []
    dest_store_id = data.get("dest_store_id")

    if not title:
        return jsonify({"erro": "Nome do produto é obrigatório"}), 400

    # Load destination store credentials
    dest_store_url = ""
    dest_token = ""
    owner_id = get_owner_id(session["user_id"])
    if dest_store_id:
        conn = get_db()
        store = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
            (dest_store_id, owner_id)
        ).fetchone()
        conn.close()
        if store:
            dest_store_url = store["store_url"]
            dest_token = store["access_token"]

    if not dest_store_url or not dest_token:
        conn = get_db()
        fallback = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE user_id = ? AND access_token IS NOT NULL AND access_token != '' ORDER BY updated_at DESC LIMIT 1",
            (owner_id,)
        ).fetchone()
        conn.close()
        if fallback:
            dest_store_url = fallback["store_url"]
            dest_token = fallback["access_token"]

    # Build a Shopify-like product structure
    handle = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    product = {
        "title": title,
        "body_html": description,
        "handle": handle,
        "images": [{"src": url} for url in image_urls],
        "variants": [],
        "options": [],
    }

    # Convert to CSV-like rows
    rows = shopify_products_to_csv_rows([product])
    images = extract_images_from_csv(rows)

    # Create job directory
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_FOLDER / job_id
    job_dir.mkdir(exist_ok=True)

    product_count = 1

    # Download images and detect text
    try:
        client = get_client()
    except ValueError as e:
        return jsonify({"erro": str(e)}), 500

    result_images = []
    for i, img_info in enumerate(images):
        logger.info(f"[Manual Product] Downloading image {i+1}/{len(images)}: {img_info['filename']}")

        img_bytes = download_image(img_info["url"])
        if not img_bytes:
            result_images.append({
                "index": i, "url": img_info["url"], "source": img_info["source"],
                "handle": img_info["handle"], "filename": img_info["filename"],
                "has_text": False, "description": "Erro ao baixar imagem", "error": True,
            })
            continue

        img_path = job_dir / f"original_{i}_{img_info['filename']}"
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        ext = Path(img_info["filename"]).suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "image/png")
        detection = detect_text_in_image(client, img_bytes, mime)

        try:
            thumb = Image.open(io.BytesIO(img_bytes))
            thumb.thumbnail((120, 120))
            if thumb.mode in ("RGBA", "P"):
                thumb = thumb.convert("RGB")
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=70)
            thumb_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            thumb_b64 = ""

        result_images.append({
            "index": i, "url": img_info["url"], "source": img_info["source"],
            "handle": img_info["handle"], "filename": img_info["filename"],
            "has_text": detection.get("has_text", False),
            "description": detection.get("description", ""),
            "thumbnail": f"data:image/jpeg;base64,{thumb_b64}" if thumb_b64 else "",
        })
        time.sleep(0.5)

    # Save state
    state = {
        "rows": rows,
        "images": [img for img in images],
        "product_count": product_count,
        "source_url": "manual",
        "shopify_store_url": dest_store_url,
        "shopify_token": dest_token,
        "shopify_products": [product],
        "dest_store_id": dest_store_id,
    }
    with open(job_dir / "state.json", "w") as f:
        json.dump(state, f, ensure_ascii=False)

    if "user_id" in session:
        save_job_record(session["user_id"], job_id, "manual_product", title, product_count, len(images))

    return jsonify({
        "job_id": job_id,
        "product_count": product_count,
        "total_images": len(images),
        "images": result_images,
        "source": "manual",
    })


@app.route("/shopify/conectar", methods=["POST"])
@login_required_api
def shopify_conectar():
    """Validate Shopify connection and fetch products count."""
    data = request.get_json()
    store_url = (data.get("store_url") or "").strip()
    token = (data.get("token") or "").strip()

    if not store_url or not token:
        return jsonify({"erro": "URL da loja e token são obrigatórios"}), 400

    # Clean store URL
    store_url = store_url.replace("https://", "").replace("http://", "").rstrip("/")
    if not store_url.endswith(".myshopify.com"):
        # Try to add .myshopify.com if not present
        if "." not in store_url:
            store_url = store_url + ".myshopify.com"

    try:
        result = shopify_api_get(store_url, token, "products/count.json")
        count = result.get("count", 0)
        return jsonify({"ok": True, "store_url": store_url, "product_count": count})
    except http_requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            return jsonify({"erro": "Token de acesso inválido. Verifique suas credenciais."}), 401
        elif e.response is not None and e.response.status_code == 404:
            return jsonify({"erro": "Loja não encontrada. Verifique a URL."}), 404
        return jsonify({"erro": f"Erro ao conectar: {e}"}), 500
    except Exception as e:
        logger.error(f"[Shopify] Erro ao conectar: {e}")
        return jsonify({"erro": f"Erro ao conectar: {e}"}), 500


@app.route("/shopify/extrair", methods=["POST"])
@login_required_api
def shopify_extrair():
    """Fetch all products from Shopify store and return as CSV-like data for analysis."""
    data = request.get_json()
    store_url = (data.get("store_url") or "").strip()
    token = (data.get("token") or "").strip()

    if not store_url or not token:
        return jsonify({"erro": "URL da loja e token são obrigatórios"}), 400

    store_url = store_url.replace("https://", "").replace("http://", "").rstrip("/")
    if not store_url.endswith(".myshopify.com"):
        if "." not in store_url:
            store_url = store_url + ".myshopify.com"

    try:
        # Fetch all products (paginated)
        all_products = []
        page_info = None
        while True:
            params = {"limit": 250}
            if page_info:
                params["page_info"] = page_info

            result = shopify_api_get(store_url, token, "products.json", params)
            products = result.get("products", [])
            all_products.extend(products)

            if len(products) < 250:
                break

            # Simple pagination - stop after 250 for safety (adjust if needed)
            break

        if not all_products:
            return jsonify({"erro": "Nenhum produto encontrado na loja"}), 400

        # Convert to CSV-like rows
        rows = shopify_products_to_csv_rows(all_products)

        # Extract images (same as CSV flow)
        images = extract_images_from_csv(rows)

        # Create job directory
        job_id = str(uuid.uuid4())[:8]
        job_dir = OUTPUT_FOLDER / job_id
        job_dir.mkdir(exist_ok=True)

        # Count unique products
        handles = set(r.get("Handle", "") for r in rows if r.get("Handle"))
        product_count = len(handles)

        # Download images and detect text (same as CSV flow)
        try:
            client = get_client()
        except ValueError as e:
            return jsonify({"erro": str(e)}), 500

        result_images = []
        for i, img_info in enumerate(images):
            logger.info(f"[Shopify] Baixando imagem {i+1}/{len(images)}: {img_info['filename']}")

            img_bytes = download_image(img_info["url"])
            if not img_bytes:
                result_images.append({
                    "index": i,
                    "url": img_info["url"],
                    "source": img_info["source"],
                    "handle": img_info["handle"],
                    "filename": img_info["filename"],
                    "has_text": False,
                    "description": "Erro ao baixar imagem",
                    "error": True,
                })
                continue

            img_path = job_dir / f"original_{i}_{img_info['filename']}"
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            ext = Path(img_info["filename"]).suffix.lower()
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "image/png")

            detection = detect_text_in_image(client, img_bytes, mime)

            try:
                thumb = Image.open(io.BytesIO(img_bytes))
                thumb.thumbnail((120, 120))
                if thumb.mode in ("RGBA", "P"):
                    thumb = thumb.convert("RGB")
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG", quality=70)
                thumb_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                thumb_b64 = ""

            result_images.append({
                "index": i,
                "url": img_info["url"],
                "source": img_info["source"],
                "handle": img_info["handle"],
                "filename": img_info["filename"],
                "has_text": detection.get("has_text", False),
                "description": detection.get("description", ""),
                "thumbnail": f"data:image/jpeg;base64,{thumb_b64}" if thumb_b64 else "",
            })

            time.sleep(0.5)

        # Save state (include store credentials for later publish)
        state = {
            "rows": rows,
            "images": [img for img in images],
            "product_count": product_count,
            "shopify_store_url": store_url,
            "shopify_token": token,
            "shopify_products": all_products,
        }
        with open(job_dir / "state.json", "w") as f:
            json.dump(state, f, ensure_ascii=False)

        # Save job record
        if "user_id" in session:
            save_job_record(session["user_id"], job_id, "shopify_api", store_url, product_count, len(images))

        return jsonify({
            "job_id": job_id,
            "product_count": product_count,
            "total_images": len(images),
            "images": result_images,
            "source": "shopify_api",
        })

    except http_requests.exceptions.HTTPError as e:
        logger.error(f"[Shopify] HTTP erro: {e}")
        return jsonify({"erro": f"Erro na API Shopify: {e}"}), 500
    except Exception as e:
        logger.error(f"[Shopify] Erro ao extrair: {e}")
        return jsonify({"erro": f"Erro: {e}"}), 500


@app.route("/shopify/colecoes", methods=["POST"])
def shopify_colecoes():
    """Fetch collections from Shopify store (both custom and smart)."""
    data = request.get_json()
    store_url = (data.get("store_url") or "").strip()
    token = (data.get("token") or "").strip()
    store_id = data.get("store_id")

    # Allow lookup by saved store ID
    if store_id and "user_id" in session:
        owner_id = get_owner_id(session["user_id"])
        conn = get_db()
        store = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
            (store_id, owner_id)
        ).fetchone()
        conn.close()
        if store:
            store_url = store["store_url"]
            token = store["access_token"]

    if not store_url or not token:
        return jsonify({"erro": "Credenciais não fornecidas"}), 400

    store_url = store_url.replace("https://", "").replace("http://", "").rstrip("/")

    collections = []
    try:
        # Custom collections
        result = shopify_api_get(store_url, token, "custom_collections.json", {"limit": 250})
        for c in result.get("custom_collections", []):
            collections.append({"id": c["id"], "title": c["title"], "type": "custom"})
    except Exception as e:
        logger.warning(f"[Shopify] Erro ao buscar custom collections: {e}")

    try:
        # Smart collections
        result = shopify_api_get(store_url, token, "smart_collections.json", {"limit": 250})
        for c in result.get("smart_collections", []):
            collections.append({"id": c["id"], "title": c["title"], "type": "smart"})
    except Exception as e:
        logger.warning(f"[Shopify] Erro ao buscar smart collections: {e}")

    return jsonify({"collections": collections})


@app.route("/shopify/templates", methods=["POST"])
def shopify_templates():
    """Fetch available theme templates from Shopify store."""
    data = request.get_json()
    store_url = (data.get("store_url") or "").strip()
    token = (data.get("token") or "").strip()
    store_id = data.get("store_id")

    # Allow lookup by saved store ID
    if store_id and "user_id" in session:
        owner_id = get_owner_id(session["user_id"])
        conn = get_db()
        store = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
            (store_id, owner_id)
        ).fetchone()
        conn.close()
        if store:
            store_url = store["store_url"]
            token = store["access_token"]

    if not store_url or not token:
        return jsonify({"erro": "Credenciais não fornecidas"}), 400

    store_url = store_url.replace("https://", "").replace("http://", "").rstrip("/")

    templates = []
    try:
        # Get the main (published) theme
        result = shopify_api_get(store_url, token, "themes.json")
        main_theme = None
        for theme in result.get("themes", []):
            if theme.get("role") == "main":
                main_theme = theme
                break

        if main_theme:
            # Get assets from the main theme to find product templates
            theme_id = main_theme["id"]
            assets_result = shopify_api_get(store_url, token, f"themes/{theme_id}/assets.json")
            for asset in assets_result.get("assets", []):
                key = asset.get("key", "")
                # Look for product template files
                if key.startswith("templates/product.") and key != "templates/product.json":
                    # Extract suffix: "templates/product.custom.json" -> "custom"
                    suffix = key.replace("templates/product.", "").replace(".json", "").replace(".liquid", "")
                    if suffix:
                        templates.append(suffix)
                # Also check sections/templates folder pattern
                elif key.startswith("sections/product-template") or key.startswith("templates/product/"):
                    suffix = key.split("/")[-1].replace(".json", "").replace(".liquid", "").replace("product-template-", "").replace("product.", "")
                    if suffix and suffix not in templates:
                        templates.append(suffix)
    except Exception as e:
        logger.warning(f"[Shopify] Erro ao buscar templates: {e}")

    return jsonify({"templates": templates})


@app.route("/shopify/preview-products", methods=["POST"])
def shopify_preview_products():
    """Return translated product data for review/editing before publish."""
    data = request.get_json()
    job_id = data.get("job_id", "")
    lang = data.get("lang", "")

    if not job_id or not lang:
        return jsonify({"erro": "job_id e lang são obrigatórios"}), 400

    job_dir = OUTPUT_FOLDER / job_id
    state_path = job_dir / "state.json"

    if not state_path.exists():
        return jsonify({"erro": "Job não encontrado"}), 404

    with open(state_path) as f:
        state = json.load(f)

    shopify_products = state.get("shopify_products", [])

    # Read translated CSV
    csv_path = job_dir / lang / f"products_export_{lang}.csv"
    text_translations = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                handle = row.get("Handle", "")
                if handle and handle not in text_translations:
                    text_translations[handle] = {
                        "title": row.get("Title", ""),
                        "body_html": row.get("Body (HTML)", ""),
                        "tags": row.get("Tags", ""),
                    }

    # Build preview list
    products = []
    for product in shopify_products:
        handle = product.get("handle", "")
        product_id = product.get("id")
        if not product_id:
            continue

        tt = text_translations.get(handle, {})
        products.append({
            "id": product_id,
            "handle": handle,
            "original_title": product.get("title", ""),
            "original_body_html": product.get("body_html", "") or "",
            "translated_title": tt.get("title", "") or product.get("title", ""),
            "translated_body_html": tt.get("body_html", "") or product.get("body_html", "") or "",
            "translated_tags": tt.get("tags", "") or product.get("tags", ""),
            "template_suffix": product.get("template_suffix", "") or "",
        })

    return jsonify({"products": products})


@app.route("/shopify/publicar", methods=["POST"])
@login_required_api
def shopify_publicar():
    """Publish translated content back to Shopify store (update or create new)."""
    data = request.get_json()
    job_id = data.get("job_id", "")
    lang = data.get("lang", "")
    collection_id = data.get("collection_id", "")
    template_suffix = data.get("template_suffix", "")
    product_edits = data.get("product_edits", {})  # {handle: {title, body_html}}
    dest_store_id = data.get("dest_store_id")  # Override destination store

    if not job_id or not lang:
        return jsonify({"erro": "job_id e lang são obrigatórios"}), 400

    job_dir = OUTPUT_FOLDER / job_id
    state_path = job_dir / "state.json"

    if not state_path.exists():
        return jsonify({"erro": "Job não encontrado"}), 404

    with open(state_path) as f:
        state = json.load(f)

    store_url = state.get("shopify_store_url", "")
    token = state.get("shopify_token", "")
    shopify_products = state.get("shopify_products", [])

    # Override with destination store from user's saved stores
    owner_id = get_owner_id(session["user_id"])
    if dest_store_id:
        conn = get_db()
        dest = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
            (dest_store_id, owner_id)
        ).fetchone()
        conn.close()
        if dest:
            store_url = dest["store_url"]
            token = dest["access_token"]

    # Fallback: if no credentials yet, try first saved store for this owner
    if not store_url or not token:
        conn = get_db()
        fallback = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE user_id = ? AND access_token IS NOT NULL AND access_token != '' ORDER BY updated_at DESC LIMIT 1",
            (owner_id,)
        ).fetchone()
        conn.close()
        if fallback:
            store_url = fallback["store_url"]
            token = fallback["access_token"]
            logger.info(f"[Publish] Fallback to store {store_url} for owner {owner_id}")

    if not store_url or not token:
        return jsonify({"erro": "Selecione uma loja destino para publicar."}), 400

    # Determine if we're creating new products (copied from another store) or updating
    is_copy = bool(state.get("source_url"))

    if not shopify_products:
        return jsonify({"erro": "Produtos Shopify não encontrados no estado do job."}), 400

    nome_idioma = IDIOMAS.get(lang, lang)

    # Read translated CSV to get text translations
    csv_path = job_dir / lang / f"products_export_{lang}.csv"
    text_translations = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                handle = row.get("Handle", "")
                if handle and handle not in text_translations:
                    text_translations[handle] = {
                        "title": row.get("Title", ""),
                        "body_html": row.get("Body (HTML)", ""),
                        "seo_title": row.get("SEO Title", ""),
                        "seo_description": row.get("SEO Description", ""),
                        "tags": row.get("Tags", ""),
                    }

    # Load image mapping for this language (original_url -> relative_path)
    img_mapping = {}
    lang_dir = job_dir / lang
    mapping_path = lang_dir / "img_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            img_mapping = json.load(f)

    # Load translated options for this language
    options_translations = {}
    options_path = lang_dir / "options_translations.json"
    if options_path.exists():
        with open(options_path) as f:
            options_translations = json.load(f)

    # Identify description images per handle from state
    state_images = state.get("images", [])
    desc_urls_by_handle = {}  # {handle: set(original_urls)}
    for img in state_images:
        if img.get("source") == "description":
            h = img.get("handle", "")
            desc_urls_by_handle.setdefault(h, set()).add(img["url"])

    # Upload translated images and create/update products
    updated = 0
    errors = []
    product_ids_for_collection = []

    # Group products by handle (avoid duplicates)
    seen_handles = set()

    for product in shopify_products:
        handle = product.get("handle", "")
        if not handle or handle in seen_handles:
            continue
        seen_handles.add(handle)

        product_data = {}

        # Base: use original product data for creating new products
        if is_copy:
            product_data["title"] = product.get("title", "")
            product_data["body_html"] = product.get("body_html", "")
            product_data["tags"] = product.get("tags", "")
            product_data["product_type"] = product.get("product_type", "")
            product_data["vendor"] = product.get("vendor", "")
            # Copy variants (price, SKU, etc.)
            original_variants = product.get("variants", [])
            if original_variants:
                product_data["variants"] = []
                for v in original_variants:
                    variant = {}
                    for key in ("price", "compare_at_price", "sku", "weight", "weight_unit",
                                "inventory_quantity", "option1", "option2", "option3",
                                "requires_shipping", "taxable"):
                        if v.get(key) is not None:
                            variant[key] = v[key]
                    product_data["variants"].append(variant)
            # Copy options (Size, Color, etc.)
            original_options = product.get("options", [])
            if original_options:
                product_data["options"] = [{"name": o.get("name", ""), "values": o.get("values", [])} for o in original_options]

        # Apply text translations (base from CSV — preserves HTML structure + images)
        if handle in text_translations:
            tt = text_translations[handle]
            if tt.get("title"):
                product_data["title"] = tt["title"]
            if tt.get("body_html"):
                product_data["body_html"] = tt["body_html"]
            if tt.get("tags"):
                product_data["tags"] = tt["tags"]

        # Apply translated options and variant values
        if handle in options_translations:
            translated_opts = options_translations[handle]
            if translated_opts and isinstance(translated_opts, list):
                # Build value mapping: original_value → translated_value
                value_map = {}
                original_options = product.get("options", [])
                for i, orig_opt in enumerate(original_options):
                    if i < len(translated_opts):
                        trans_opt = translated_opts[i]
                        orig_values = orig_opt.get("values", [])
                        trans_values = trans_opt.get("values", [])
                        for j, ov in enumerate(orig_values):
                            if j < len(trans_values):
                                value_map[ov] = trans_values[j]

                # Apply translated options
                product_data["options"] = translated_opts

                # Apply translated values to variants
                if "variants" in product_data:
                    for variant in product_data["variants"]:
                        for opt_key in ("option1", "option2", "option3"):
                            if variant.get(opt_key) and variant[opt_key] in value_map:
                                variant[opt_key] = value_map[variant[opt_key]]

        # Apply manual edits (override CSV translations)
        if handle in product_edits:
            edits = product_edits[handle]
            if edits.get("title"):
                product_data["title"] = edits["title"]
            if edits.get("body_html"):
                product_data["body_html"] = edits["body_html"]

        # Generate translated handle from title for copy mode
        if is_copy and product_data.get("title"):
            import unicodedata
            slug = product_data["title"].lower().strip()
            slug = unicodedata.normalize("NFKD", slug).encode("ascii", "ignore").decode("ascii")
            slug = re.sub(r"[^a-z0-9\s-]", "", slug)
            slug = re.sub(r"[\s_]+", "-", slug).strip("-")
            slug = re.sub(r"-{2,}", "-", slug)
            if slug:
                product_data["handle"] = slug

        # Apply template suffix
        if template_suffix:
            product_data["template_suffix"] = template_suffix

        # Collect translated images — separate gallery from description
        gallery_images = []
        desc_image_data = []  # [{attachment, filename}] — description only
        desc_filenames = set()
        desc_orig_urls = desc_urls_by_handle.get(handle, set())

        if lang_dir.exists():
            for img_file in lang_dir.iterdir():
                if img_file.is_file() and img_file.suffix in (".webp", ".jpg", ".jpeg", ".png") and handle in img_file.stem:
                    with open(img_file, "rb") as f:
                        img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

                    img_entry = {"attachment": img_b64, "filename": img_file.name}

                    # Check if this is a description image
                    is_desc = False
                    for orig_url, rel_path in img_mapping.items():
                        if orig_url in desc_orig_urls and Path(rel_path).name == img_file.name:
                            is_desc = True
                            desc_filenames.add(img_file.name)
                            break

                    if is_desc:
                        desc_image_data.append(img_entry)
                    else:
                        gallery_images.append(img_entry)

        # For copy: include original gallery images when none were translated
        if is_copy and not gallery_images:
            for img in product.get("images", []):
                src = img.get("src", "")
                if src:
                    gallery_images.append({"src": src})

        # Only gallery/cover images go in product_data["images"]
        if gallery_images:
            product_data["images"] = gallery_images

        if not product_data:
            continue

        try:
            if is_copy:
                # CREATE new product on destination store
                url = f"https://{store_url}/admin/api/{SHOPIFY_API_VERSION}/products.json"
                headers = {"X-Shopify-Access-Token": token, "Content-Type": "application/json"}
                resp = http_requests.post(url, headers=headers, json={"product": product_data}, timeout=60)
                if resp.status_code >= 400:
                    err_body = resp.text[:500]
                    logger.error(f"[Shopify] Erro {resp.status_code} ao criar {handle}: {err_body}")
                resp.raise_for_status()
                new_product = resp.json().get("product", {})
                new_id = new_product.get("id")
                new_handle = new_product.get("handle", "")
                if new_id:
                    product_ids_for_collection.append(new_id)
                    # Persist new product id back to state so subsequent
                    # /shopify/traduzir can find it by handle.
                    for sp in shopify_products:
                        if sp.get("handle") == handle and not sp.get("id"):
                            sp["id"] = new_id
                            if new_handle:
                                sp["shopify_handle"] = new_handle
                            break

                # ── Upload description images separately & fix body_html ──
                if new_id and desc_image_data:
                    cdn_map = {}  # filename → CDN URL

                    # Upload each description image via product images API
                    imgs_url = f"https://{store_url}/admin/api/{SHOPIFY_API_VERSION}/products/{new_id}/images.json"
                    for desc_img in desc_image_data:
                        try:
                            img_resp = http_requests.post(
                                imgs_url, headers=headers,
                                json={"image": desc_img}, timeout=30
                            )
                            if img_resp.status_code < 400:
                                img_result = img_resp.json().get("image", {})
                                cdn_src = img_result.get("src", "")
                                if cdn_src:
                                    cdn_fname = cdn_src.split("/")[-1].split("?")[0]
                                    cdn_map[desc_img["filename"]] = cdn_src
                        except Exception as img_err:
                            logger.warning(f"[Shopify] Falha ao enviar img desc {desc_img['filename']}: {img_err}")

                    if cdn_map:
                        body = product_data.get("body_html", "") or ""
                        body_updated = False

                        # Replace relative paths (from CSV) with CDN URLs
                        for orig_url, rel_path in img_mapping.items():
                            if orig_url in desc_orig_urls:
                                fname = Path(rel_path).name
                                cdn_url = cdn_map.get(fname)
                                if cdn_url:
                                    if rel_path in body:
                                        body = body.replace(rel_path, cdn_url)
                                        body_updated = True
                                    elif fname in body:
                                        body = body.replace(fname, cdn_url)
                                        body_updated = True

                        # If no replacements (user edited text → images lost), append at end
                        if not body_updated:
                            img_tags = []
                            for orig_url, rel_path in img_mapping.items():
                                if orig_url in desc_orig_urls:
                                    fname = Path(rel_path).name
                                    cdn_url = cdn_map.get(fname)
                                    if cdn_url:
                                        img_tags.append(f'<p><img src="{cdn_url}" alt=""></p>')
                            if img_tags:
                                body = body + "\n" + "\n".join(img_tags)
                                body_updated = True

                        if body_updated:
                            put_url = f"https://{store_url}/admin/api/{SHOPIFY_API_VERSION}/products/{new_id}.json"
                            http_requests.put(put_url, headers=headers,
                                              json={"product": {"id": new_id, "body_html": body}}, timeout=30)
                            logger.info(f"[Shopify] body_html de {handle} atualizado com {len(cdn_map)} imagens de descrição")

                updated += 1
                logger.info(f"[Shopify] Produto {handle} criado com sucesso na loja destino")
            else:
                # UPDATE existing product
                product_id = product.get("id")
                if not product_id:
                    continue
                shopify_api_put(store_url, token, f"products/{product_id}.json", {"product": product_data})
                updated += 1
                product_ids_for_collection.append(product_id)
                logger.info(f"[Shopify] Produto {handle} ({product_id}) atualizado com sucesso")
        except http_requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            body = e.response.text[:300] if e.response is not None else ""
            logger.error(f"[Shopify] HTTP {status} ao publicar {handle}: {body}")
            # Show user-friendly error with details
            detail = body
            try:
                err_json = e.response.json() if e.response is not None else {}
                if "errors" in err_json:
                    detail = str(err_json["errors"])
            except Exception:
                pass
            errors.append(f"{handle}: HTTP {status} - {detail[:200]}")
        except Exception as e:
            logger.error(f"[Shopify] Erro ao publicar {handle}: {e}")
            errors.append(f"{handle}: {str(e)}")

    # Add products to collection if specified
    collection_added = 0
    if collection_id and product_ids_for_collection:
        for pid in product_ids_for_collection:
            try:
                url = f"https://{store_url}/admin/api/{SHOPIFY_API_VERSION}/collects.json"
                headers = {"X-Shopify-Access-Token": token, "Content-Type": "application/json"}
                collect_data = {"collect": {"product_id": pid, "collection_id": int(collection_id)}}
                resp = http_requests.post(url, headers=headers, json=collect_data, timeout=30)
                resp.raise_for_status()
                collection_added += 1
            except Exception as e:
                logger.warning(f"[Shopify] Erro ao adicionar produto {pid} à coleção: {e}")

    # Persist state with newly created product IDs so /shopify/traduzir
    # can find them without relying on handle lookup.
    try:
        state["shopify_products"] = shopify_products
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[Shopify] Falha ao salvar state.json com IDs: {e}")

    return jsonify({
        "ok": True,
        "updated": updated,
        "collection_added": collection_added,
        "errors": errors,
        "language": nome_idioma,
    })


@app.route("/shopify/traduzir", methods=["POST"])
@login_required_api
def shopify_traduzir():
    """Push translations via Shopify Translations API (Translate & Adapt integration)."""
    data = request.get_json()
    job_id = data.get("job_id", "")
    lang = data.get("lang", "")
    product_edits = data.get("product_edits", {})
    dest_store_id = data.get("dest_store_id")

    if not job_id or not lang:
        return jsonify({"erro": "job_id e lang são obrigatórios"}), 400

    # Map language code to Shopify locale
    shopify_locale = LANG_TO_SHOPIFY_LOCALE.get(lang, lang)

    job_dir = OUTPUT_FOLDER / job_id
    state_path = job_dir / "state.json"

    if not state_path.exists():
        return jsonify({"erro": "Job não encontrado"}), 404

    with open(state_path) as f:
        state = json.load(f)

    store_url = state.get("shopify_store_url", "")
    token = state.get("shopify_token", "")
    shopify_products = state.get("shopify_products", [])

    # Resolve store credentials (same logic as publish)
    owner_id = get_owner_id(session["user_id"])
    if dest_store_id:
        conn = get_db()
        dest = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
            (dest_store_id, owner_id)
        ).fetchone()
        conn.close()
        if dest:
            store_url = dest["store_url"]
            token = dest["access_token"]

    if not store_url or not token:
        conn = get_db()
        fallback = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE user_id = ? AND access_token IS NOT NULL AND access_token != '' ORDER BY updated_at DESC LIMIT 1",
            (owner_id,)
        ).fetchone()
        conn.close()
        if fallback:
            store_url = fallback["store_url"]
            token = fallback["access_token"]

    if not store_url or not token:
        return jsonify({"erro": "Selecione uma loja destino."}), 400

    if not shopify_products:
        return jsonify({"erro": "Produtos Shopify não encontrados no estado do job."}), 400

    # Read translated CSV
    csv_path = job_dir / lang / f"products_export_{lang}.csv"
    text_translations = {}
    csv_missing = not csv_path.exists()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                handle = row.get("Handle", "")
                if handle and handle not in text_translations:
                    text_translations[handle] = {
                        "title": row.get("Title", ""),
                        "body_html": row.get("Body (HTML)", ""),
                        "seo_title": row.get("SEO Title", ""),
                        "seo_description": row.get("SEO Description", ""),
                        "tags": row.get("Tags", ""),
                    }
    else:
        logger.warning(f"[Translate] CSV traduzido não encontrado em {csv_path}. Rode a tradução de textos primeiro.")

    # Load translated options
    lang_dir = job_dir / lang
    options_translations = {}
    options_path = lang_dir / "options_translations.json"
    if options_path.exists():
        with open(options_path) as f:
            options_translations = json.load(f)

    translated = 0
    errors = []
    seen_handles = set()
    not_found = []
    no_translations = []  # handles sem texto traduzido para registrar

    for product in shopify_products:
        handle = product.get("handle", "")
        product_id = product.get("id")
        if not handle or handle in seen_handles:
            continue
        seen_handles.add(handle)

        # If product has no Shopify ID (scraped externally), try to find it by handle in dest store
        if not product_id:
            try:
                lookup_url = f"https://{store_url}/admin/api/{SHOPIFY_API_VERSION}/products.json?handle={handle}"
                lookup_resp = http_requests.get(
                    lookup_url,
                    headers={"X-Shopify-Access-Token": token},
                    timeout=30,
                )
                if lookup_resp.status_code == 200:
                    found = lookup_resp.json().get("products", [])
                    if found:
                        product_id = found[0].get("id")
                        logger.info(f"[Translate] Produto '{handle}' encontrado na loja destino: id={product_id}")
            except Exception as e:
                logger.warning(f"[Translate] Erro ao buscar '{handle}' na loja destino: {e}")

        if not product_id:
            not_found.append(handle)
            continue

        resource_gid = f"gid://shopify/Product/{product_id}"

        try:
            # Get translatable content with digests
            content_list = shopify_get_translatable_content(store_url, token, resource_gid)
            if not content_list:
                logger.warning(f"[Translate] No translatable content for {handle} ({product_id})")
                errors.append(f"{handle}: Sem conteúdo traduzível")
                continue

            # Build digest map: key → digest
            digest_map = {}
            for c in content_list:
                digest_map[c["key"]] = c["digest"]

            # Get translated values
            tt = text_translations.get(handle, {})
            edits = product_edits.get(handle, {})

            # Build translation entries
            translations_to_register = []

            # Title
            title_val = edits.get("title") or tt.get("title", "")
            if title_val and "title" in digest_map:
                translations_to_register.append({
                    "key": "title",
                    "value": title_val,
                    "digest": digest_map["title"],
                })

            # Body HTML
            body_val = edits.get("body_html") or tt.get("body_html", "")
            if body_val and "body_html" in digest_map:
                translations_to_register.append({
                    "key": "body_html",
                    "value": body_val,
                    "digest": digest_map["body_html"],
                })

            # Meta title (SEO)
            seo_title = tt.get("seo_title", "")
            if seo_title and "meta_title" in digest_map:
                translations_to_register.append({
                    "key": "meta_title",
                    "value": seo_title,
                    "digest": digest_map["meta_title"],
                })

            # Meta description (SEO)
            seo_desc = tt.get("seo_description", "")
            if seo_desc and "meta_description" in digest_map:
                translations_to_register.append({
                    "key": "meta_description",
                    "value": seo_desc,
                    "digest": digest_map["meta_description"],
                })

            if not translations_to_register:
                no_translations.append(handle)
                logger.warning(
                    f"[Translate] {handle}: nenhum campo para registrar "
                    f"(csv_has_handle={handle in text_translations}, csv_missing={csv_missing})"
                )
                continue

            # Register translations
            result = shopify_register_translation(
                store_url, token, resource_gid, shopify_locale, translations_to_register
            )

            user_errors = result.get("data", {}).get("translationsRegister", {}).get("userErrors", [])
            if user_errors:
                err_msgs = "; ".join(e.get("message", "") for e in user_errors)
                logger.warning(f"[Translate] Errors for {handle}: {err_msgs}")
                errors.append(f"{handle}: {err_msgs}")
            else:
                translated += 1
                logger.info(f"[Translate] {handle} traduzido para {shopify_locale} ({len(translations_to_register)} campos)")

            # Translate variant options
            if handle in options_translations:
                translated_opts = options_translations[handle]
                original_options = product.get("options", [])

                for i, orig_opt in enumerate(original_options):
                    if i >= len(translated_opts):
                        break
                    trans_opt = translated_opts[i]
                    option_id = orig_opt.get("id")
                    if not option_id:
                        continue

                    option_gid = f"gid://shopify/ProductOption/{option_id}"
                    try:
                        option_content = shopify_get_translatable_content(store_url, token, option_gid)
                        option_digest_map = {c["key"]: c["digest"] for c in option_content}

                        option_translations = []
                        trans_name = trans_opt.get("name", "")
                        if trans_name and "name" in option_digest_map:
                            option_translations.append({
                                "key": "name",
                                "value": trans_name,
                                "digest": option_digest_map["name"],
                            })

                        if option_translations:
                            shopify_register_translation(
                                store_url, token, option_gid, shopify_locale, option_translations
                            )

                        # Translate option values
                        orig_values = orig_opt.get("values", [])
                        trans_values = trans_opt.get("values", [])
                        for j, ov in enumerate(orig_values):
                            if j >= len(trans_values):
                                break
                            # ProductOptionValue GIDs require querying — use product variant approach
                            # Option values are translated through ProductOptionValue resources
                    except Exception as opt_err:
                        logger.warning(f"[Translate] Option {option_id} error: {opt_err}")

        except Exception as e:
            logger.error(f"[Translate] Erro em {handle}: {e}")
            errors.append(f"{handle}: {str(e)}")

    nome_idioma = IDIOMAS.get(lang, lang)
    return jsonify({
        "ok": True,
        "translated": translated,
        "errors": errors,
        "not_found": not_found,
        "no_translations": no_translations,
        "csv_missing": csv_missing,
        "language": nome_idioma,
        "locale": shopify_locale,
    })


# ─── Traduzir Produtos Existentes (da loja do usuário) ───

@app.route("/shopify/list-my-products", methods=["POST"])
@login_required_api
def shopify_list_my_products():
    """Lista produtos de uma loja conectada do usuário."""
    data = request.get_json() or {}
    store_id = data.get("store_id")

    if not store_id:
        return jsonify({"erro": "store_id obrigatório"}), 400

    owner_id = get_owner_id(session["user_id"])
    conn = get_db()
    store = conn.execute(
        "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
        (store_id, owner_id)
    ).fetchone()
    conn.close()

    if not store or not store["access_token"]:
        return jsonify({"erro": "Loja não encontrada ou sem token"}), 404

    store_url = store["store_url"]
    token = store["access_token"]

    # Fetch products via Admin API
    products = []
    try:
        page_info = None
        while True:
            if page_info:
                url = f"https://{store_url}/admin/api/{SHOPIFY_API_VERSION}/products.json?limit=250&page_info={page_info}"
            else:
                url = f"https://{store_url}/admin/api/{SHOPIFY_API_VERSION}/products.json?limit=250"

            resp = http_requests.get(url, headers={"X-Shopify-Access-Token": token}, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"[ListMyProducts] status {resp.status_code}: {resp.text[:200]}")
                break

            batch = resp.json().get("products", [])
            for p in batch:
                products.append({
                    "id": p.get("id"),
                    "title": p.get("title", ""),
                    "handle": p.get("handle", ""),
                    "body_html": p.get("body_html", "") or "",
                    "image": (p.get("image") or {}).get("src", "") if p.get("image") else "",
                    "options": p.get("options", []),
                    "tags": p.get("tags", "") or "",
                })

            # Parse Link header for pagination
            link = resp.headers.get("Link", "")
            next_match = re.search(r'<[^>]*page_info=([^&>]+)[^>]*>;\s*rel="next"', link)
            if next_match:
                page_info = next_match.group(1)
            else:
                break

            if len(products) > 2000:
                break
    except Exception as e:
        logger.error(f"[ListMyProducts] erro: {e}")
        return jsonify({"erro": f"Erro ao buscar produtos: {str(e)}"}), 500

    return jsonify({"products": products, "count": len(products)})


@app.route("/shopify/traduzir-existentes", methods=["POST"])
@login_required_api
def shopify_traduzir_existentes():
    """Traduz produtos existentes da loja do usuário e aplica via Translate & Adapt."""
    data = request.get_json() or {}
    store_id = data.get("store_id")
    product_ids = data.get("product_ids", [])
    langs = data.get("langs", [])
    marca_de = (data.get("marca_de") or "").strip()
    marca_para = (data.get("marca_para") or "").strip()

    if not store_id or not product_ids or not langs:
        return jsonify({"erro": "store_id, product_ids e langs obrigatórios"}), 400

    owner_id = get_owner_id(session["user_id"])
    conn = get_db()
    store = conn.execute(
        "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
        (store_id, owner_id)
    ).fetchone()
    conn.close()

    if not store or not store["access_token"]:
        return jsonify({"erro": "Loja não encontrada"}), 404

    store_url = store["store_url"]
    token = store["access_token"]

    try:
        client = get_client()
    except ValueError as e:
        return jsonify({"erro": str(e)}), 500

    results_by_lang = {}
    total_translated = 0

    # Pre-flight: verify the access token has translation scope by
    # querying translatableResource on the first product. If this fails,
    # abort early with a clear message instead of retrying per-product.
    if product_ids:
        test_gid = f"gid://shopify/Product/{product_ids[0]}"
        try:
            shopify_get_translatable_content(store_url, token, test_gid)
        except RuntimeError as e:
            msg = str(e)
            if "access denied" in msg.lower() or "not authorized" in msg.lower() or "scope" in msg.lower():
                return jsonify({
                    "erro": (
                        "O token da loja destino não tem permissão para gerenciar traduções. "
                        "Reinstale o app custom adicionando os scopes 'read_translations' e "
                        "'write_translations' nas configurações da sua Shopify custom app."
                    ),
                    "detalhe": msg,
                }), 403
            return jsonify({"erro": f"Erro ao acessar API de traduções: {msg}"}), 500

    # Fetch full product data once
    products_data = {}
    for pid in product_ids:
        try:
            url = f"https://{store_url}/admin/api/{SHOPIFY_API_VERSION}/products/{pid}.json"
            resp = http_requests.get(url, headers={"X-Shopify-Access-Token": token}, timeout=30)
            if resp.status_code == 200:
                p = resp.json().get("product", {})
                products_data[pid] = p
        except Exception as e:
            logger.warning(f"[TraduzirExistentes] erro ao buscar produto {pid}: {e}")

    for lang in langs:
        shopify_locale = LANG_TO_SHOPIFY_LOCALE.get(lang, lang)
        idioma_nome = IDIOMAS.get(lang, lang)
        lang_results = {"translated": 0, "errors": []}

        for pid, product in products_data.items():
            handle = product.get("handle", str(pid))
            try:
                title = product.get("title", "")
                body_html = product.get("body_html", "") or ""

                # SEO metafields — need separate query
                seo_title = ""
                seo_desc = ""
                # Options
                options = product.get("options", [])
                options_for_translation = [
                    {"name": o.get("name", ""), "values": o.get("values", [])}
                    for o in options
                ]

                # Translate via Claude Sonnet
                traducao = traduzir_texto_produto(
                    client, title, body_html, seo_title, seo_desc,
                    product.get("tags", "") or "",
                    idioma_nome, marca_de, marca_para,
                    options=options_for_translation if options_for_translation else None,
                )

                # Register translations
                resource_gid = f"gid://shopify/Product/{pid}"
                content_list = shopify_get_translatable_content(store_url, token, resource_gid)
                if not content_list:
                    lang_results["errors"].append(f"{handle}: sem conteúdo traduzível")
                    continue

                digest_map = {c["key"]: c["digest"] for c in content_list}
                translations_to_register = []

                tt_title = traducao.get("title", "")
                if tt_title and "title" in digest_map:
                    translations_to_register.append({
                        "key": "title", "value": tt_title, "digest": digest_map["title"],
                    })

                tt_body = traducao.get("body_html", "")
                if tt_body and "body_html" in digest_map:
                    translations_to_register.append({
                        "key": "body_html", "value": tt_body, "digest": digest_map["body_html"],
                    })

                tt_seo_title = traducao.get("seo_title", "")
                if tt_seo_title and "meta_title" in digest_map:
                    translations_to_register.append({
                        "key": "meta_title", "value": tt_seo_title, "digest": digest_map["meta_title"],
                    })

                tt_seo_desc = traducao.get("seo_description", "")
                if tt_seo_desc and "meta_description" in digest_map:
                    translations_to_register.append({
                        "key": "meta_description", "value": tt_seo_desc, "digest": digest_map["meta_description"],
                    })

                if translations_to_register:
                    result = shopify_register_translation(
                        store_url, token, resource_gid, shopify_locale, translations_to_register
                    )
                    user_errors = result.get("data", {}).get("translationsRegister", {}).get("userErrors", [])
                    if user_errors:
                        err_msgs = "; ".join(e.get("message", "") for e in user_errors)
                        lang_results["errors"].append(f"{handle}: {err_msgs}")
                    else:
                        lang_results["translated"] += 1
                        total_translated += 1

                # Translate options
                trans_options = traducao.get("options", [])
                for i, orig_opt in enumerate(options):
                    if i >= len(trans_options):
                        break
                    t_opt = trans_options[i]
                    opt_id = orig_opt.get("id")
                    if not opt_id:
                        continue
                    opt_gid = f"gid://shopify/ProductOption/{opt_id}"
                    try:
                        opt_content = shopify_get_translatable_content(store_url, token, opt_gid)
                        opt_digest_map = {c["key"]: c["digest"] for c in opt_content}
                        opt_trans = []
                        t_name = t_opt.get("name", "")
                        if t_name and "name" in opt_digest_map:
                            opt_trans.append({
                                "key": "name", "value": t_name, "digest": opt_digest_map["name"],
                            })
                        if opt_trans:
                            shopify_register_translation(
                                store_url, token, opt_gid, shopify_locale, opt_trans
                            )
                    except Exception as opt_err:
                        logger.warning(f"[TraduzirExistentes] option {opt_id}: {opt_err}")

            except Exception as e:
                logger.error(f"[TraduzirExistentes] {handle} ({lang}): {e}")
                lang_results["errors"].append(f"{handle}: {str(e)}")

        results_by_lang[lang] = lang_results

    return jsonify({
        "ok": True,
        "total_translated": total_translated,
        "by_language": results_by_lang,
    })


# ─── Rotas CSV Shopify ────────────────────────────────────

@app.route("/csv/analisar", methods=["POST"])
def csv_analisar():
    """Recebe CSV Shopify, extrai imagens e detecta texto."""
    _cleanup_old_files()

    if "csv_file" not in request.files:
        return jsonify({"erro": "Nenhum arquivo CSV enviado"}), 400

    csv_file = request.files["csv_file"]
    if not csv_file.filename or not csv_file.filename.lower().endswith(".csv"):
        return jsonify({"erro": "Envie um arquivo .csv"}), 400

    # Parse CSV
    try:
        rows = parse_shopify_csv(csv_file)
    except Exception as e:
        logger.error(f"[CSV] Erro ao parsear CSV: {e}")
        return jsonify({"erro": f"Erro ao ler CSV: {e}"}), 400

    if not rows:
        return jsonify({"erro": "CSV vazio ou sem dados"}), 400

    # Verificar colunas obrigatórias
    required = {"Handle", "Image Src"}
    if not required.issubset(set(rows[0].keys())):
        return jsonify({"erro": f"CSV não tem colunas obrigatórias: {required}"}), 400

    # Extrair imagens
    images = extract_images_from_csv(rows)
    if not images:
        return jsonify({"erro": "Nenhuma imagem encontrada no CSV"}), 400

    # Criar job directory
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_FOLDER / job_id
    job_dir.mkdir(exist_ok=True)

    # Contar produtos únicos
    handles = set(r.get("Handle", "") for r in rows if r.get("Handle"))
    product_count = len(handles)

    # Download imagens e detectar texto
    try:
        client = get_client()
    except ValueError as e:
        return jsonify({"erro": str(e)}), 500

    result_images = []
    for i, img_info in enumerate(images):
        logger.info(f"[CSV] Baixando imagem {i+1}/{len(images)}: {img_info['filename']}")

        img_bytes = download_image(img_info["url"])
        if not img_bytes:
            result_images.append({
                "index": i,
                "url": img_info["url"],
                "source": img_info["source"],
                "handle": img_info["handle"],
                "filename": img_info["filename"],
                "has_text": False,
                "description": "Erro ao baixar imagem",
                "error": True,
            })
            continue

        # Salvar imagem em disco
        img_path = job_dir / f"original_{i}_{img_info['filename']}"
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        # Detectar texto
        ext = Path(img_info["filename"]).suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "image/png")

        detection = detect_text_in_image(client, img_bytes, mime)

        # Thumbnail (base64, pequeno)
        try:
            thumb = Image.open(io.BytesIO(img_bytes))
            thumb.thumbnail((120, 120))
            if thumb.mode in ("RGBA", "P"):
                thumb = thumb.convert("RGB")
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=70)
            thumb_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            thumb_b64 = ""

        result_images.append({
            "index": i,
            "url": img_info["url"],
            "source": img_info["source"],
            "handle": img_info["handle"],
            "filename": img_info["filename"],
            "has_text": detection.get("has_text", False),
            "description": detection.get("description", ""),
            "thumbnail": f"data:image/jpeg;base64,{thumb_b64}" if thumb_b64 else "",
        })

        time.sleep(0.5)  # Rate limit para Vision API

    # Salvar estado
    state = {
        "rows": rows,
        "images": [img for img in images],
        "product_count": product_count,
    }
    with open(job_dir / "state.json", "w") as f:
        json.dump(state, f, ensure_ascii=False)

    # Save job record
    if "user_id" in session:
        save_job_record(session["user_id"], job_id, "csv_upload", f"{product_count} produtos", product_count, len(images))

    return jsonify({
        "job_id": job_id,
        "product_count": product_count,
        "total_images": len(images),
        "images": result_images,
    })


@app.route("/csv/traduzir/stream")
def csv_traduzir_stream():
    """SSE endpoint para traduzir imagens e/ou texto do CSV — 1 idioma por vez."""
    job_id = request.args.get("job_id", "")
    selected = request.args.get("images", "")  # "0,2,5" ou "none"
    lang = request.args.get("lang", "")  # "en" (1 idioma apenas)
    marca_de = request.args.get("marca_de", "")
    marca_para = request.args.get("marca_para", "")
    traduzir_texto = request.args.get("traduzir_texto", "false") == "true"
    formato_saida = request.args.get("formato", "webp")

    if not job_id or not lang:
        def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Parâmetros inválidos'})}\n\n"
        return Response(error_gen(), mimetype="text/event-stream")

    job_dir = OUTPUT_FOLDER / job_id
    state_path = job_dir / "state.json"

    if not state_path.exists():
        def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Job não encontrado'})}\n\n"
        return Response(error_gen(), mimetype="text/event-stream")

    selected_indices = [int(x) for x in selected.split(",") if x.strip().isdigit()] if selected and selected != "none" else []
    nome_idioma = IDIOMAS.get(lang, lang)

    # Número de traduções paralelas (3 é conservador e evita rate limit)
    MAX_PARALLEL = 3

    def generate():
        try:
            with open(state_path) as f:
                state = json.load(f)

            rows = state["rows"]
            images_info = state["images"]
            client = get_client()

            # Calcular total de tarefas para este idioma
            img_tasks = len(selected_indices)
            unique_handles = []
            if traduzir_texto:
                seen = set()
                for r in rows:
                    h = r.get("Handle", "")
                    if h and h not in seen and (r.get("Title") or r.get("Body (HTML)")):
                        seen.add(h)
                        unique_handles.append(h)
            text_tasks = len(unique_handles) if traduzir_texto else 0
            total = img_tasks + text_tasks
            current = 0
            mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

            mapping = {}  # {original_url: new_relative_path}
            text_trans = {}  # {handle: {title, body_html, ...}}

            # ── ETAPA 1: Traduzir imagens em PARALELO ──
            # Preparar tarefas
            img_tasks_list = []
            for img_idx in selected_indices:
                if img_idx >= len(images_info):
                    continue
                img_info = images_info[img_idx]
                img_filename = img_info.get("filename", f"img_{img_idx}")
                original_path = job_dir / f"original_{img_idx}_{img_filename}"
                if not original_path.exists():
                    current += 1
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Imagem não encontrada: {img_filename}', 'current': current, 'total': total})}\n\n"
                    continue
                img_tasks_list.append((img_idx, img_info, img_filename, original_path))

            if img_tasks_list:
                yield f"data: {json.dumps({'type': 'progress', 'current': current, 'total': total, 'image': f'Traduzindo {len(img_tasks_list)} imagens em paralelo...', 'language': nome_idioma})}\n\n"

                # Fila thread-safe para receber resultados
                result_queue = queue.Queue()
                lang_dir = job_dir / lang
                lang_dir.mkdir(exist_ok=True)

                def _translate_image(task):
                    """Worker thread: traduz 1 imagem e coloca resultado na fila."""
                    img_idx, img_info, img_filename, original_path = task
                    try:
                        with open(original_path, "rb") as f:
                            img_bytes = f.read()
                        ext = Path(img_filename).suffix.lower()
                        mime = mime_types.get(ext, "image/png")
                        handle = img_info.get("handle", "product")

                        resultado = traduzir_imagem(client, img_bytes, mime, nome_idioma, marca_de, marca_para)

                        if resultado:
                            resultado_conv = converter_imagem(resultado, formato_saida)
                            ext_saida = "webp" if formato_saida == "webp" else "jpg"
                            nome_base = Path(img_filename).stem
                            nome_saida = f"{handle}_{nome_base}_{lang}.{ext_saida}"
                            caminho_saida = lang_dir / nome_saida

                            with open(caminho_saida, "wb") as fout:
                                fout.write(resultado_conv)

                            # Thumbnail
                            try:
                                thumb = Image.open(io.BytesIO(resultado_conv))
                                thumb.thumbnail((150, 150))
                                buf = io.BytesIO()
                                thumb.save(buf, format="JPEG", quality=75)
                                preview_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
                            except Exception:
                                preview_b64 = ""

                            result_queue.put({
                                "ok": True,
                                "img_filename": img_filename,
                                "url": img_info["url"],
                                "nome_saida": nome_saida,
                                "preview_b64": preview_b64,
                            })
                        else:
                            result_queue.put({
                                "ok": False,
                                "img_filename": img_filename,
                            })
                    except Exception as e:
                        logger.error(f"[CSV] Thread erro {img_filename}: {e}")
                        result_queue.put({
                            "ok": False,
                            "img_filename": img_filename,
                        })

                # Lançar threads em paralelo
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
                    futures = [executor.submit(_translate_image, task) for task in img_tasks_list]

                    # Coletar resultados à medida que completam
                    done_count = 0
                    total_img = len(img_tasks_list)
                    while done_count < total_img:
                        # Heartbeat enquanto espera
                        try:
                            result = result_queue.get(timeout=5)
                        except queue.Empty:
                            yield ": keepalive\n\n"
                            continue

                        done_count += 1
                        current += 1

                        if result["ok"]:
                            mapping[result["url"]] = f"{lang}/{result['nome_saida']}"
                            preview = f"data:image/jpeg;base64,{result['preview_b64']}" if result["preview_b64"] else ""
                            # Track image translation usage
                            if "user_id" in session:
                                track_api_usage(
                                    session["user_id"], "image_translation",
                                    model=result.get("model_used", "gemini"),
                                    images_count=1, job_id=job_id,
                                )
                            yield f"data: {json.dumps({'type': 'image_done', 'current': current, 'total': total, 'image': result['img_filename'], 'language': nome_idioma, 'lang_code': lang, 'output_file': result['nome_saida'], 'preview': preview})}\n\n"
                        else:
                            fname = result["img_filename"]
                            yield f"data: {json.dumps({'type': 'error', 'message': f'Falha ao traduzir {fname} → {nome_idioma}', 'current': current, 'total': total})}\n\n"

            # ── ETAPA 2: Traduzir texto dos produtos ──
            if traduzir_texto and unique_handles:
                product_data = {}
                for r in rows:
                    h = r.get("Handle", "")
                    if h and h not in product_data:
                        product_data[h] = {
                            "title": r.get("Title", ""),
                            "body_html": r.get("Body (HTML)", ""),
                            "seo_title": r.get("SEO Title", ""),
                            "seo_description": r.get("SEO Description", ""),
                            "tags": r.get("Tags", ""),
                        }

                # Build handle → options map from shopify_products
                shopify_products = state.get("shopify_products", [])
                options_by_handle = {}
                for sp in shopify_products:
                    h = sp.get("handle", "")
                    opts = sp.get("options", [])
                    if h and opts:
                        options_by_handle[h] = [{"name": o.get("name", ""), "values": o.get("values", [])} for o in opts]

                for handle in unique_handles:
                    pd = product_data.get(handle, {})
                    if not pd.get("title") and not pd.get("body_html"):
                        current += 1
                        continue

                    current += 1

                    yield f"data: {json.dumps({'type': 'text_progress', 'current': current, 'total': total, 'handle': handle, 'language': nome_idioma})}\n\n"
                    yield ": keepalive\n\n"

                    result = traduzir_texto_produto(
                        client,
                        title=pd.get("title", ""),
                        body_html=pd.get("body_html", ""),
                        seo_title=pd.get("seo_title", ""),
                        seo_desc=pd.get("seo_description", ""),
                        tags=pd.get("tags", ""),
                        idioma_nome=nome_idioma,
                        marca_de=marca_de,
                        marca_para=marca_para,
                        options=options_by_handle.get(handle),
                    )

                    if result:
                        text_trans[handle] = result
                        yield f"data: {json.dumps({'type': 'text_done', 'current': current, 'total': total, 'handle': handle, 'language': nome_idioma})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Falha ao traduzir texto de {handle} → {nome_idioma}', 'current': current, 'total': total})}\n\n"

            # ── ETAPA 3: Gerar CSV para este idioma ──
            yield f"data: {json.dumps({'type': 'progress', 'current': total, 'total': total, 'image': f'Gerando CSV {lang.upper()}...', 'language': nome_idioma})}\n\n"

            has_images = bool(mapping)
            has_text = bool(text_trans)
            if has_images or has_text:
                csv_content = generate_translated_csv(
                    rows, mapping, lang,
                    text_translations=text_trans
                )
                lang_dir = job_dir / lang
                lang_dir.mkdir(exist_ok=True)
                csv_path = lang_dir / f"products_export_{lang}.csv"
                with open(csv_path, "w", encoding="utf-8") as fout:
                    fout.write(csv_content)

            # Save image mapping for publish route (original_url -> translated_filename)
            if mapping:
                lang_dir = job_dir / lang
                lang_dir.mkdir(exist_ok=True)
                mapping_path = lang_dir / "img_mapping.json"
                with open(mapping_path, "w") as fmap:
                    json.dump(mapping, fmap, ensure_ascii=False)

            # Save translated options for publish route
            if text_trans:
                options_data = {}
                for h, tt in text_trans.items():
                    if "options" in tt:
                        options_data[h] = tt["options"]
                if options_data:
                    lang_dir = job_dir / lang
                    lang_dir.mkdir(exist_ok=True)
                    with open(lang_dir / "options_translations.json", "w") as fopt:
                        json.dump(options_data, fopt, ensure_ascii=False)

            yield f"data: {json.dumps({'type': 'complete', 'lang': lang, 'total_translated': current})}\n\n"

            # Update job record with completed language
            if "user_id" in session:
                conn = get_db()
                job_rec = conn.execute("SELECT languages FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                if job_rec:
                    langs_done = job_rec["languages"]
                    langs_done = f"{langs_done},{lang}" if langs_done else lang
                    conn.execute("UPDATE jobs SET languages = ?, status = 'translated' WHERE job_id = ?", (langs_done, job_id))
                    conn.commit()
                conn.close()

        except Exception as e:
            logger.error(f"[CSV] Erro na tradução SSE: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/jobs/list")
@login_required_api
def jobs_list():
    """List saved jobs for the current user."""
    user_id = session["user_id"]
    conn = get_db()
    rows = conn.execute(
        "SELECT job_id, source_type, source_label, product_count, image_count, languages, status, created_at FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT 20",
        (user_id,)
    ).fetchall()
    conn.close()

    jobs = []
    for r in rows:
        # Check if job directory still exists
        job_dir = OUTPUT_FOLDER / r["job_id"]
        if job_dir.exists():
            jobs.append({
                "job_id": r["job_id"],
                "source_type": r["source_type"],
                "source_label": r["source_label"],
                "product_count": r["product_count"],
                "image_count": r["image_count"],
                "languages": r["languages"],
                "status": r["status"],
                "created_at": r["created_at"],
            })

    return jsonify({"jobs": jobs})


@app.route("/jobs/load/<job_id>")
@login_required_api
def jobs_load(job_id):
    """Load a saved job for resuming."""
    user_id = session["user_id"]
    conn = get_db()
    job_rec = conn.execute(
        "SELECT * FROM jobs WHERE job_id = ? AND user_id = ?", (job_id, user_id)
    ).fetchone()
    conn.close()

    if not job_rec:
        return jsonify({"erro": "Projeto não encontrado"}), 404

    job_dir = OUTPUT_FOLDER / job_id
    state_path = job_dir / "state.json"
    if not state_path.exists():
        return jsonify({"erro": "Dados do projeto não encontrados"}), 404

    with open(state_path) as f:
        state = json.load(f)

    # Get translated languages and their preview products
    languages = [l.strip() for l in job_rec["languages"].split(",") if l.strip()]

    # Get image results from state
    images = state.get("images", [])
    result_images = []
    for i, img in enumerate(images):
        # Try to find thumbnail
        thumb = ""
        original_path = list(job_dir.glob(f"original_{i}_*"))
        if original_path:
            try:
                thumb_img = Image.open(original_path[0])
                thumb_img.thumbnail((120, 120))
                if thumb_img.mode in ("RGBA", "P"):
                    thumb_img = thumb_img.convert("RGB")
                buf = io.BytesIO()
                thumb_img.save(buf, format="JPEG", quality=70)
                thumb = f"data:image/jpeg;base64,{base64.standard_b64encode(buf.getvalue()).decode('utf-8')}"
            except Exception:
                pass

        result_images.append({
            "index": i,
            "url": img.get("url", ""),
            "source": img.get("source", ""),
            "handle": img.get("handle", ""),
            "filename": img.get("filename", ""),
            "has_text": img.get("has_text", False),
            "description": img.get("description", ""),
            "thumbnail": thumb,
        })

    return jsonify({
        "job_id": job_id,
        "source_type": job_rec["source_type"],
        "source_label": job_rec["source_label"],
        "product_count": job_rec["product_count"],
        "image_count": job_rec["image_count"],
        "languages": languages,
        "status": job_rec["status"],
        "images": result_images,
    })


@app.route("/csv/download/<job_id>")
def csv_download(job_id):
    """Gera e baixa ZIP final com todas as pastas de idiomas processadas."""
    job_dir = OUTPUT_FOLDER / job_id
    if not job_dir.exists():
        return jsonify({"erro": "Job não encontrado"}), 404

    zip_path = job_dir / "shopify_translated.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for lang_dir in sorted(job_dir.iterdir()):
            if lang_dir.is_dir() and lang_dir.name in IDIOMAS:
                for f in lang_dir.iterdir():
                    if f.is_file():
                        zf.write(f, f"{lang_dir.name}/{f.name}")

        # Gerar mapping.json
        mapping_data = {}
        for lang_dir in sorted(job_dir.iterdir()):
            if lang_dir.is_dir() and lang_dir.name in IDIOMAS:
                mapping_data[lang_dir.name] = {
                    "files": [f.name for f in lang_dir.iterdir() if f.is_file()]
                }
        zf.writestr("mapping.json", json.dumps(mapping_data, indent=2, ensure_ascii=False))

    return send_file(str(zip_path), as_attachment=True, download_name="shopify_translated.zip")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = not os.environ.get("RAILWAY_ENVIRONMENT")
    print(f"\n  Image Tools rodando em: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
