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
import requests as http_requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
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

# Cloud: usar /tmp para storage efêmero; local: pastas relativas
if os.environ.get("RAILWAY_ENVIRONMENT"):
    UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "img-uploads"
    OUTPUT_FOLDER = Path(tempfile.gettempdir()) / "img-output"
else:
    UPLOAD_FOLDER = Path(__file__).parent / "uploads"
    OUTPUT_FOLDER = Path(__file__).parent / "output"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)


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


def converter_para_jpg(imagem_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(imagem_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85, optimize=True, progressive=True, subsampling=0)
    return buffer.getvalue()


def traduzir_imagem(client: OpenAI, imagem_bytes: bytes, mime_type: str,
                     idioma_nome: str, marca_de: str = "", marca_para: str = "") -> Optional[bytes]:
    """Traduz imagem via OpenAI API."""

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
                            return base64.standard_b64decode(img_b64)
            if hasattr(item, "image") and item.image:
                img_b64 = item.image.get("b64_json") if isinstance(item.image, dict) else getattr(item.image, "b64_json", None)
                if img_b64:
                    return base64.standard_b64decode(img_b64)

        resp_dict = response.model_dump() if hasattr(response, "model_dump") else {}
        for output_item in resp_dict.get("output", []):
            if output_item.get("type") == "image_generation_call":
                result = output_item.get("result", {})
                if isinstance(result, dict) and "b64_json" in result:
                    return base64.standard_b64decode(result["b64_json"])
            for key in ("result", "content", "image"):
                val = output_item.get(key)
                if isinstance(val, dict) and "b64_json" in val:
                    return base64.standard_b64decode(val["b64_json"])
                if isinstance(val, list):
                    for v in val:
                        if isinstance(v, dict) and "b64_json" in v:
                            return base64.standard_b64decode(v["b64_json"])

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
                return base64.standard_b64decode(img_data.b64_json)
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
    return render_template("index.html", idiomas=IDIOMAS)


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
                resultado_jpg = converter_para_jpg(resultado)
                nome_saida = f"{nome_base}_{codigo}.jpg"
                caminho_saida = job_dir / nome_saida

                with open(caminho_saida, "wb") as f:
                    f.write(resultado_jpg)

                preview_b64 = base64.standard_b64encode(resultado_jpg).decode("utf-8")

                resultados.append({
                    "idioma": codigo,
                    "idioma_nome": nome_idioma,
                    "imagem_original": arquivo.filename,
                    "arquivo": nome_saida,
                    "tamanho_kb": round(len(resultado_jpg) / 1024),
                    "preview": f"data:image/jpeg;base64,{preview_b64}",
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
        resultado_jpg = converter_para_jpg(resultado)
        nome_saida = f"substituido_{job_id}.jpg"
        caminho_saida = job_dir / nome_saida

        with open(caminho_saida, "wb") as f:
            f.write(resultado_jpg)

        preview_b64 = base64.standard_b64encode(resultado_jpg).decode("utf-8")

        return jsonify({
            "job_id": job_id,
            "preview": f"data:image/jpeg;base64,{preview_b64}",
            "download_url": f"/download/{job_id}/{nome_saida}",
            "tamanho_kb": round(len(resultado_jpg) / 1024),
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

    return jsonify({
        "job_id": job_id,
        "product_count": product_count,
        "total_images": len(images),
        "images": result_images,
    })


@app.route("/csv/traduzir/stream")
def csv_traduzir_stream():
    """SSE endpoint para traduzir imagens e/ou texto do CSV."""
    job_id = request.args.get("job_id", "")
    selected = request.args.get("images", "")  # "0,2,5" ou "none"
    idiomas_str = request.args.get("idiomas", "")  # "en,es"
    marca_de = request.args.get("marca_de", "")
    marca_para = request.args.get("marca_para", "")
    traduzir_texto = request.args.get("traduzir_texto", "false") == "true"

    if not job_id or not idiomas_str:
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
    idiomas_list = [x.strip() for x in idiomas_str.split(",") if x.strip()]

    def generate():
        try:
            with open(state_path) as f:
                state = json.load(f)

            rows = state["rows"]
            images_info = state["images"]
            client = get_client()

            # Calcular total de tarefas
            img_tasks = len(selected_indices) * len(idiomas_list)
            # Produtos únicos para tradução de texto
            unique_handles = []
            if traduzir_texto:
                seen = set()
                for r in rows:
                    h = r.get("Handle", "")
                    if h and h not in seen and (r.get("Title") or r.get("Body (HTML)")):
                        seen.add(h)
                        unique_handles.append(h)
            text_tasks = len(unique_handles) * len(idiomas_list) if traduzir_texto else 0
            total = img_tasks + text_tasks
            current = 0
            mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

            # mapping: {lang: {original_url: new_relative_path}}
            mappings = {lang: {} for lang in idiomas_list}
            # text_translations: {lang: {handle: {title, body_html, ...}}}
            text_trans = {lang: {} for lang in idiomas_list}

            # ── ETAPA 1: Traduzir imagens ──
            for img_idx in selected_indices:
                if img_idx >= len(images_info):
                    continue

                img_info = images_info[img_idx]
                img_filename = img_info.get("filename", f"img_{img_idx}")

                original_path = job_dir / f"original_{img_idx}_{img_filename}"
                if not original_path.exists():
                    for lang in idiomas_list:
                        current += 1
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Imagem não encontrada: {img_filename}', 'current': current, 'total': total})}\n\n"
                    continue

                with open(original_path, "rb") as f:
                    img_bytes = f.read()

                ext = Path(img_filename).suffix.lower()
                mime = mime_types.get(ext, "image/png")
                handle = img_info.get("handle", "product")

                for lang in idiomas_list:
                    current += 1
                    nome_idioma = IDIOMAS.get(lang, lang)

                    yield f"data: {json.dumps({'type': 'progress', 'current': current, 'total': total, 'image': img_filename, 'language': nome_idioma})}\n\n"

                    resultado = traduzir_imagem(client, img_bytes, mime, nome_idioma, marca_de, marca_para)

                    if resultado:
                        resultado_jpg = converter_para_jpg(resultado)
                        nome_base = Path(img_filename).stem
                        nome_saida = f"{handle}_{nome_base}_{lang}.jpg"

                        lang_dir = job_dir / lang
                        lang_dir.mkdir(exist_ok=True)
                        caminho_saida = lang_dir / nome_saida

                        with open(caminho_saida, "wb") as fout:
                            fout.write(resultado_jpg)

                        mappings[lang][img_info["url"]] = f"{lang}/{nome_saida}"

                        try:
                            thumb = Image.open(io.BytesIO(resultado_jpg))
                            thumb.thumbnail((150, 150))
                            buf = io.BytesIO()
                            thumb.save(buf, format="JPEG", quality=75)
                            preview_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
                        except Exception:
                            preview_b64 = ""

                        yield f"data: {json.dumps({'type': 'image_done', 'current': current, 'total': total, 'image': img_filename, 'language': nome_idioma, 'lang_code': lang, 'output_file': nome_saida, 'preview': f'data:image/jpeg;base64,{preview_b64}' if preview_b64 else ''})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Falha ao traduzir {img_filename} → {nome_idioma}', 'current': current, 'total': total})}\n\n"

                    time.sleep(2)

            # ── ETAPA 2: Traduzir texto dos produtos ──
            if traduzir_texto and unique_handles:
                # Construir dados dos produtos (pegar da primeira row de cada handle)
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
                        # Pular se não tem conteúdo para traduzir
                        for lang in idiomas_list:
                            current += 1
                        continue

                    for lang in idiomas_list:
                        current += 1
                        nome_idioma = IDIOMAS.get(lang, lang)

                        yield f"data: {json.dumps({'type': 'text_progress', 'current': current, 'total': total, 'handle': handle, 'language': nome_idioma})}\n\n"

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
                            text_trans[lang][handle] = result
                            yield f"data: {json.dumps({'type': 'text_done', 'current': current, 'total': total, 'handle': handle, 'language': nome_idioma})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'error', 'message': f'Falha ao traduzir texto de {handle} → {nome_idioma}', 'current': current, 'total': total})}\n\n"

                        time.sleep(1)

            # ── ETAPA 3: Gerar CSVs e ZIP ──
            yield f"data: {json.dumps({'type': 'progress', 'current': total, 'total': total, 'image': 'Gerando ZIP...', 'language': ''})}\n\n"

            zip_path = job_dir / "shopify_translated.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for lang in idiomas_list:
                    lang_dir = job_dir / lang
                    if lang_dir.exists():
                        for img_file in lang_dir.iterdir():
                            if img_file.is_file():
                                zf.write(img_file, f"{lang}/{img_file.name}")

                for lang in idiomas_list:
                    has_images = bool(mappings[lang])
                    has_text = bool(text_trans.get(lang))
                    if has_images or has_text:
                        csv_content = generate_translated_csv(
                            rows, mappings[lang], lang,
                            text_translations=text_trans.get(lang)
                        )
                        zf.writestr(f"{lang}/products_export_{lang}.csv", csv_content)

                zf.writestr("mapping.json", json.dumps(mappings, indent=2, ensure_ascii=False))

            download_url = f"/csv/download/{job_id}"
            yield f"data: {json.dumps({'type': 'complete', 'download_url': download_url, 'total_translated': current})}\n\n"

        except Exception as e:
            logger.error(f"[CSV] Erro na tradução SSE: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/csv/download/<job_id>")
def csv_download(job_id):
    """Download ZIP com imagens traduzidas e CSVs."""
    zip_path = OUTPUT_FOLDER / job_id / "shopify_translated.zip"
    if not zip_path.exists():
        return jsonify({"erro": "Arquivo não encontrado"}), 404
    return send_file(str(zip_path), as_attachment=True, download_name="shopify_translated.zip")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = not os.environ.get("RAILWAY_ENVIRONMENT")
    print(f"\n  Image Tools rodando em: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
