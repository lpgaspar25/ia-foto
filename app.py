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
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Aviso: google-genai não instalado. Substituição de produto indisponível.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "imagetools-default-secret-key-2024-prod")

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
    """)
    # Add columns if they don't exist (migration for existing DBs)
    try:
        conn.execute("ALTER TABLE shopify_stores ADD COLUMN client_id TEXT NOT NULL DEFAULT ''")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE shopify_stores ADD COLUMN client_secret TEXT NOT NULL DEFAULT ''")
    except Exception:
        pass
    conn.commit()
    conn.close()


init_db()


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
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.5-flash-image",
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
    """Traduz imagem — ChatGPT para 1:1/2:3/3:2, Gemini para outras proporções."""

    # Capturar dimensões originais para preservar após tradução
    original_img = Image.open(io.BytesIO(imagem_bytes))
    original_size = original_img.size  # (width, height)
    logger.info(f"[Tradução] Imagem original: {original_size[0]}×{original_size[1]}")

    # Rotear por proporção
    engine = _get_aspect_ratio_category(original_size[0], original_size[1])

    if engine == "gemini":
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


def traduzir_texto_produto(client: OpenAI, title: str, body_html: str,
                            seo_title: str, seo_desc: str, tags: str,
                            idioma_nome: str, marca_de: str = "", marca_para: str = "") -> dict:
    """Traduz campos textuais de um produto via GPT-4o (chat completion)."""
    marca_instrucao = ""
    if marca_de and marca_para:
        marca_instrucao = f'\nBRAND REPLACEMENT: Replace ALL occurrences of the brand "{marca_de}" with "{marca_para}" in every field.\n'

    prompt = f"""Translate ALL text fields of this Shopify product to {idioma_nome}.
{marca_instrucao}
RULES:
- Keep ALL HTML tags and attributes exactly as they are (only translate the visible text between tags)
- Keep brand names unchanged (unless brand replacement is specified above)
- Keep measurement units (cm, mm, kg, etc.) unchanged
- Keep product codes/SKUs unchanged
- If a field is empty, return it as empty string
- Return ONLY valid JSON (no markdown, no ```), with these exact keys:

{{"title": "translated title", "body_html": "translated HTML", "seo_title": "translated SEO title", "seo_description": "translated SEO description", "tags": "translated tags"}}

Input:
title: {title}
body_html: {body_html}
seo_title: {seo_title}
seo_description: {seo_desc}
tags: {tags}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
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
        conn.commit()
        user_id = cursor.lastrowid

        session["user_id"] = user_id
        conn.close()

        return jsonify({"ok": True, "user": {"id": user_id, "nome": nome, "email": email}})
    except Exception as e:
        conn.close()
        logger.error(f"[Auth] Erro no registro: {e}")
        return jsonify({"erro": "Erro ao criar conta"}), 500


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
    stores = conn.execute(
        "SELECT id, store_name, store_url FROM shopify_stores WHERE user_id = ? ORDER BY updated_at DESC",
        (user["id"],)
    ).fetchall()
    conn.close()

    return jsonify({
        "logged_in": True,
        "user": user,
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

    user_id = session["user_id"]
    conn = get_db()

    # Check if this store already exists for this user
    existing = conn.execute(
        "SELECT id FROM shopify_stores WHERE user_id = ? AND store_url = ?",
        (user_id, store_url)
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
            (user_id, store_name or store_url, store_url, token)
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
    user_id = session["user_id"]
    conn = get_db()

    existing = conn.execute(
        "SELECT id FROM shopify_stores WHERE user_id = ? AND store_url = ?",
        (user_id, store_url)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE shopify_stores SET access_token = ?, store_name = ?, client_id = ?, client_secret = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (access_token, store_name, client_id, client_secret, existing["id"])
        )
    else:
        conn.execute(
            "INSERT INTO shopify_stores (user_id, store_name, store_url, access_token, client_id, client_secret) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, store_name, store_url, access_token, client_id, client_secret)
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
    user_id = session["user_id"]
    conn = get_db()
    store = conn.execute(
        "SELECT id, store_name, store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
        (store_id, user_id)
    ).fetchone()
    conn.close()

    if not store:
        return jsonify({"erro": "Loja não encontrada"}), 404

    return jsonify({"store": dict(store)})


@app.route("/auth/shopify-store/<int:store_id>/reconnect", methods=["POST"])
@login_required_api
def auth_reconnect_shopify_store(store_id):
    """Reconnect a store via OAuth using saved credentials."""
    user_id = session["user_id"]
    conn = get_db()
    store = conn.execute(
        "SELECT id, store_url, client_id, client_secret, store_name FROM shopify_stores WHERE id = ? AND user_id = ?",
        (store_id, user_id)
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
    user_id = session["user_id"]
    conn = get_db()
    store = conn.execute(
        "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
        (store_id, user_id)
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
    user_id = session["user_id"]
    conn = get_db()
    conn.execute(
        "DELETE FROM shopify_stores WHERE id = ? AND user_id = ?",
        (store_id, user_id)
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


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
    return m.group(1) if m else None


@app.route("/shopify/copiar", methods=["POST"])
@login_required_api
def shopify_copiar():
    """Scrape products from a public Shopify store and prepare for translation."""
    data = request.get_json()
    source_url = (data.get("source_url") or "").strip()
    dest_store_id = data.get("dest_store_id")

    if not source_url:
        return jsonify({"erro": "URL da loja fonte é obrigatória"}), 400

    # Detect if the URL points to a specific product
    product_handle = _extract_product_handle(source_url)
    store_domain = _clean_store_url(source_url)

    # Load destination store credentials
    dest_store_url = ""
    dest_token = ""
    if dest_store_id:
        user_id = session["user_id"]
        conn = get_db()
        store = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
            (dest_store_id, user_id)
        ).fetchone()
        conn.close()
        if store:
            dest_store_url = store["store_url"]
            dest_token = store["access_token"]

    # Fetch products from public endpoint
    try:
        all_products = []

        if product_handle:
            # Single product by handle
            url = f"https://{store_domain}/products/{product_handle}.json"
            resp = http_requests.get(url, timeout=30)
            resp.raise_for_status()
            product = resp.json().get("product")
            if product:
                all_products.append(product)
        else:
            # All products from store
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
                if page > 10:  # safety limit
                    break

        if not all_products:
            return jsonify({"erro": "Nenhum produto encontrado. Verifique a URL."}), 400

    except http_requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return jsonify({"erro": "Produto ou loja não encontrada. Verifique a URL."}), 404
        return jsonify({"erro": f"Erro ao acessar loja: {e}"}), 500
    except Exception as e:
        logger.error(f"[Shopify Copiar] Erro: {e}")
        return jsonify({"erro": f"Erro: {e}"}), 500

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
        conn = get_db()
        store = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
            (store_id, session["user_id"])
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
        conn = get_db()
        store = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
            (store_id, session["user_id"])
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
    if dest_store_id:
        user_id = session["user_id"]
        conn = get_db()
        dest = conn.execute(
            "SELECT store_url, access_token FROM shopify_stores WHERE id = ? AND user_id = ?",
            (dest_store_id, user_id)
        ).fetchone()
        conn.close()
        if dest:
            store_url = dest["store_url"]
            token = dest["access_token"]

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

        # Collect translated images and track description image filenames
        translated_images = []
        desc_filenames = set()  # filenames of description images (for CDN matching)
        desc_orig_urls = desc_urls_by_handle.get(handle, set())

        if lang_dir.exists():
            for img_file in lang_dir.iterdir():
                if img_file.is_file() and img_file.suffix in (".webp", ".jpg", ".jpeg", ".png") and handle in img_file.stem:
                    with open(img_file, "rb") as f:
                        img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
                    translated_images.append({
                        "attachment": img_b64,
                        "filename": img_file.name,
                    })
                    # Check if this is a description image
                    for orig_url, rel_path in img_mapping.items():
                        if orig_url in desc_orig_urls and Path(rel_path).name == img_file.name:
                            desc_filenames.add(img_file.name)

        # For copy: also include original images that weren't translated
        if is_copy and not translated_images:
            for img in product.get("images", []):
                src = img.get("src", "")
                if src:
                    translated_images.append({"src": src})

        if translated_images:
            product_data["images"] = translated_images

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
                if new_id:
                    product_ids_for_collection.append(new_id)

                # ── Replace description image URLs in body_html with CDN URLs ──
                if new_id and desc_filenames:
                    new_images = new_product.get("images", [])
                    # Build filename → CDN URL map
                    cdn_map = {}
                    for cdn_img in new_images:
                        cdn_src = cdn_img.get("src", "")
                        cdn_fname = cdn_src.split("/")[-1].split("?")[0]
                        cdn_map[cdn_fname] = cdn_src

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

                    # If no URL replacements happened (e.g. user edited text, images lost),
                    # append translated description images at the end of body_html
                    if not body_updated and desc_filenames:
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
                        logger.info(f"[Shopify] body_html de {handle} atualizado com imagens de descrição")

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

    return jsonify({
        "ok": True,
        "updated": updated,
        "collection_added": collection_added,
        "errors": errors,
        "language": nome_idioma,
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
