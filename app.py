#!/usr/bin/env python3
"""
Tradutor de Imagens - Web App
Flask server com interface visual para traduzir imagens de produtos.
"""
from __future__ import annotations

import os
import sys
import base64
import io
import time
import uuid
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image

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
RULES:
- Keep the EXACT same layout, positions, fonts, colors, sizes, and design
- Only change the text content — nothing else in the image should change
- Keep measurement units (cm, mm, kg, etc.) unchanged
- Translate measurement labels (ALTURA→HEIGHT, LENTE→LENS, FRENTE→FRONT, PONTE→BRIDGE, HASTE→TEMPLE, MEDIDAS→MEASUREMENTS, INCLUSO EM SEU PEDIDO→INCLUDED IN YOUR ORDER, etc.)
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
    img.save(buffer, format="JPEG", quality=95)
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

    except Exception:
        pass

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
    except Exception:
        pass

    return None


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = not os.environ.get("RAILWAY_ENVIRONMENT")
    print(f"\n  Image Tools rodando em: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
