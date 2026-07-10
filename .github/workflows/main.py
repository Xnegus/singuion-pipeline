"""
Sin Guion (Rosa Maria Palacios) capture pipeline.
Polls the La Republica RSS feed, grabs unprocessed episode days (the show
publishes TWO audio parts per day), transcribes both parts with Gemini,
generates one topic report per day, and uploads both files to Google Drive.
State (which feed items are done) lives in state/processed.json.

Adapted from the Comite de Lectura pipeline reference.
"""

import os
import re
import json
import time
import html as html_lib
import pathlib
import datetime

import requests
import feedparser
from google import genai
from google.genai import types

# --- Configuration ----------------------------------------------------------
FEED_URL = "https://podcast.larepublica.pe/rss/podcast-sin-guion.xml"
GEMINI_MODEL = "gemini-2.5-flash"
STATE_FILE = pathlib.Path("state/processed.json")
# Only process a day once its last part is at least this old, unless part 2/2
# is already present. Protects against running between part 1 and part 2.
SETTLE_MINUTES = 45


# --- State: dedup by item id ------------------------------------------------
def load_state():
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    return set()


def save_state(done):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(sorted(done), ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --- Feed reading + new-day detection ---------------------------------------
def part_number(title):
    m = re.search(r"Parte\s*(\d)", title, re.I)
    return int(m.group(1)) if m else 1


def pick_new_day(done):
    """Return the newest fully-published day with unprocessed items:
    {"date": "YYYY-MM-DD", "items": [entry, ...]} sorted part 1 first."""
    feed = feedparser.parse(FEED_URL)
    days = {}
    for entry in feed.entries:
        guid = entry.get("id") or entry.get("link")
        pp = entry.get("published_parsed")
        if not pp:
            continue
        audio_url = None
        for enc in entry.get("enclosures", []):
            if enc.get("type", "").startswith("audio"):
                audio_url = enc.get("href")
                break
        if not audio_url:
            continue
        day = time.strftime("%Y-%m-%d", pp)
        days.setdefault(day, []).append({
            "guid": guid,
            "title": entry.get("title", ""),
            "audio_url": audio_url,
            "published": entry.get("published", ""),
            "published_ts": time.mktime(pp),
        })

    for day in sorted(days, reverse=True):
        items = days[day]
        if all(i["guid"] in done for i in items):
            continue
        has_final_part = any(re.search(r"Parte\s*\d\s*/\s*\d", i["title"], re.I)
                             and part_number(i["title"]) >= 2 for i in items)
        age_min = (time.time() - max(i["published_ts"] for i in items)) / 60
        if not has_final_part and age_min < SETTLE_MINUTES:
            print(f"Day {day} still settling ({age_min:.0f} min old, no final part); skipping this run.")
            continue
        items.sort(key=lambda i: (part_number(i["title"]), i["published_ts"]))
        return {"date": day, "items": items}
    return None


def download_audio(url, dest):
    with requests.get(url, stream=True, timeout=180, allow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return dest


# --- Processing with Gemini --------------------------------------------------
def gemini_client():
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def transcribe(client, audio_path):
    uploaded = client.files.upload(file=audio_path)
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name == "FAILED":
        raise RuntimeError("Gemini could not process the audio.")

    prompt = (
        "Transcribe este audio en espanol de forma literal y completa. "
        "Devuelve unicamente el texto transcrito, sin comentarios, encabezados "
        "ni marcas de tiempo."
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[uploaded, prompt],
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=32000),
    )
    return resp.text.strip()


def build_report(client, transcript, day):
    titles = "\n".join(f"- {i['title']}" for i in day["items"])
    prompt = f"""Eres un analista que prepara un reporte por tema del programa diario peruano "Sin Guion" de Rosa Maria Palacios (RMP).

Fecha del episodio: {day['date']}
El episodio se publica en partes; titulos oficiales de las partes:
{titles}

Transcripcion completa del episodio (todas las partes, en orden):
{transcript}

Redacta un reporte en espanol, en formato Markdown, con esta estructura:
1. Un encabezado H1 con un titulo para el episodio y una linea con la fecha de publicacion.
2. Una seccion por cada noticia o tema tratado, con un subtitulo claro y un resumen
   de 3 a 6 lineas que capture hechos, cifras, fechas y nombres propios mencionados
   en la transcripcion. Preserva los analisis legales y constitucionales de RMP,
   que son su sello distintivo.
3. Si hay una entrevista o segmento de preguntas del publico (usualmente la Parte 2),
   dale su propia seccion con el nombre del entrevistado y los puntos principales.

No inventes datos que no esten en la transcripcion. Se preciso y conciso.
La ortografia en espanol debe ser perfecta, con todas las tildes y enes correctas."""
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=8000),
    )
    return resp.text.strip()


# --- Google Drive sink (upload as the user via OAuth refresh token) ----------
def drive_service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials(
        token=None,
        refresh_token=os.environ["GDRIVE_REFRESH_TOKEN"],
        client_id=os.environ["GDRIVE_CLIENT_ID"],
        client_secret=os.environ["GDRIVE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds)


def upload_to_drive(svc, local_path, name, folder_id):
    from googleapiclient.http import MediaFileUpload

    meta = {"name": name, "parents": [folder_id]}
    media = MediaFileUpload(local_path, mimetype="text/markdown", resumable=False)
    svc.files().create(body=meta, media_body=media, fields="id").execute()


# --- Orchestration ------------------------------------------------------------
def slugify(text):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60].strip("-")


def main():
    done = load_state()
    day = pick_new_day(done)
    if not day:
        print("No new episode day.")
        return

    print(f"Processing {day['date']} ({len(day['items'])} part(s))")
    client = gemini_client()

    sections = []
    for n, item in enumerate(day["items"], 1):
        audio = f"parte{n}.mp3"
        download_audio(item["audio_url"], audio)
        text = transcribe(client, audio)
        sections.append(f"## Parte {n}: {item['title']}\n\n{text}")
    transcript = "\n\n".join(sections)

    report = build_report(client, transcript, day)

    # Slug from part 1's title, dropping the "| RMP #SinGuion | (Parte x/y)" tail
    t1 = re.split(r"\|", day["items"][0]["title"])[0]
    t1 = re.sub(r"^\d{2}\.\d{2}\s*", "", t1)  # drop DD.MM prefix
    base = f"{day['date']}_{slugify(t1)}"
    f_trans = f"{base}_transcripcion.md"
    f_rep = f"{base}_reporte.md"
    pathlib.Path(f_trans).write_text(transcript, encoding="utf-8")
    pathlib.Path(f_rep).write_text(report, encoding="utf-8")

    folder = os.environ.get("GDRIVE_FOLDER_ID")
    if folder:
        svc = drive_service()
        upload_to_drive(svc, f_rep, f_rep, folder)
        upload_to_drive(svc, f_trans, f_trans, folder)

    for item in day["items"]:
        done.add(item["guid"])
    save_state(done)
    print("Done:", base)


if __name__ == "__main__":
    main()
