"""
Shared dashboard tracking step for the Comite de Lectura and Sin Guion (RMP)
pipelines. Runs in GitHub Actions right after the capture step.

Design (v2, delta merge):
  1. Reads hilos.md and the Archivos folder straight from Google Drive.
  2. Uses the CURRENTLY PUBLISHED dashboard JSON (news.json / rmp.json) as its
     memory of which episodes are already processed. Never writes to Drive.
  3. For each NEW episode, Claude Opus reads the full current JSON + the episode
     sources but returns ONLY A SMALL DELTA (new/updated entries per thread,
     state, synthesis). Python applies the delta deterministically and keeps
     the meta bookkeeping. Small outputs = cheap + no giant-JSON parse failures.
  4. Publishes the updated JSON to the dashboard via upload.php after each
     episode, guarded by quality gates.

Config (env):
  SOURCE_NAME           e.g. "Comite de Lectura - Las noticias con Augusto Townsend"
  MOOD_SUBJECT          whose mood the semaforo thread tracks
  DAILY_NEWS_FOLDER_ID  Drive folder that holds hilos.md
  ARCHIVOS_FOLDER_ID    Drive folder with YYYY-MM-DD_*_reporte.md / _transcripcion.md
  OUTPUT_PATH           "news.json" or "rmp.json"
  DASH_JSON_URL         public URL of the currently published JSON
  MIN_EPISODE_DATE      optional floor (YYYY-MM-DD); older episodes are skipped
  ANTHROPIC_MODEL       optional, defaults to claude-opus-4-8

Secrets (env):
  ANTHROPIC_API_KEY
  GDRIVE_CLIENT_ID / GDRIVE_CLIENT_SECRET / GDRIVE_REFRESH_TOKEN
  DASH_UPLOAD_URL / DASH_UPLOAD_TOKEN
"""

import os
import re
import sys
import json
import datetime

import requests

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
MAX_ATTEMPTS = 2   # Opus calls per episode before giving up


def cfg(name, default=None, required=False):
    v = os.environ.get(name, default)
    if required and not v:
        sys.exit(f"Missing required env var: {name}")
    return v


def lima_today():
    tz = datetime.timezone(datetime.timedelta(hours=-5))  # America/Lima, no DST
    return datetime.datetime.now(tz).strftime("%Y-%m-%d")


def iso_week(date_str):
    y, m, d = (int(x) for x in date_str.split("-"))
    return datetime.date(y, m, d).isocalendar()[:2]


# --- Google Drive (read only, as the user via OAuth refresh token) -----------
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
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def drive_list_children(svc, folder_id):
    files, page = [], None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageSize=1000, pageToken=page,
        ).execute()
        files.extend(resp.get("files", []))
        page = resp.get("nextPageToken")
        if not page:
            break
    return files


def drive_download_text(svc, file_id):
    data = svc.files().get_media(fileId=file_id).execute()
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return str(data)


def read_hilos(svc, folder_id):
    for f in drive_list_children(svc, folder_id):
        if f["name"].strip().lower() == "hilos.md":
            return drive_download_text(svc, f["id"])
    sys.exit(f"hilos.md not found in Drive folder {folder_id}")


def collect_episodes(svc, folder_id):
    """Return {date: {"reporte": id, "transcripcion": id}} for the Archivos folder."""
    eps = {}
    for f in drive_list_children(svc, folder_id):
        name = f["name"]
        m = DATE_RE.match(name)
        if not m:
            continue
        date = m.group(1)
        kind = ("reporte" if "_reporte" in name
                else "transcripcion" if "_transcripcion" in name else None)
        if not kind:
            continue
        eps.setdefault(date, {})[kind] = f["id"]
    return eps


# --- Dashboard I/O -----------------------------------------------------------
def fetch_published_json(url):
    try:
        r = requests.get(url + "?_=" + lima_today(),
                         headers={"User-Agent": BROWSER_UA}, timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Could not fetch current {url}: {e}. Starting empty.")
        return None


def publish(json_obj, path):
    url = cfg("DASH_UPLOAD_URL", required=True)
    token = cfg("DASH_UPLOAD_TOKEN", required=True)
    body = json.dumps(json_obj, ensure_ascii=False, indent=2).encode("utf-8")
    r = requests.post(
        url,
        data={"token": token, "path": path},
        files={"file": (path, body, "application/json")},
        headers={"User-Agent": BROWSER_UA},
        timeout=120,
    )
    print("upload.php ->", r.status_code, r.text[:300])
    r.raise_for_status()


# --- Claude Opus: delta extraction -------------------------------------------
def anthropic_client():
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def processed_dates(doc):
    if not doc:
        return set()
    return set((doc.get("meta") or {}).get("processed_episodes") or [])


EMPTY_DOC = {"meta": {"processed_episodes": []}, "threads": []}


def build_prompt(current_doc, hilos, source_name, mood_subject, ep_date,
                 reporte, transcripcion):
    current_json = json.dumps(current_doc, ensure_ascii=False, indent=2)
    last = max(processed_dates(current_doc)) if processed_dates(current_doc) else None
    week_note = ""
    if last and iso_week(ep_date) != iso_week(last):
        week_note = (f"- ATENCION: {ep_date} cae en una semana ISO distinta al ultimo episodio ({last}). "
                     "En el hilo de temas de la semana, incluye updated_entries que muevan las entradas "
                     "con group \"semana-en-curso\" a group \"historial\" antes de agregar lo nuevo.\n")
    return f"""Eres el motor de seguimiento de hilos del podcast "{source_name}" para un dashboard web personal. Lee el JSON actual y el episodio nuevo, y devuelve UNICAMENTE UN DELTA en JSON con los cambios que este episodio provoca. NO devuelvas el documento completo.

## Hilos vigentes (hilos.md). El seguimiento debe corresponder a estos; un hilo de hilos.md que no exista en el JSON actual se crea via el delta (incluye title y mode).
{hilos}

## JSON actual de seguimiento (contexto; NO lo re-emitas)
{current_json}

## Episodio nuevo a integrar: {ep_date}

### Reporte por temas
{reporte}

### Transcripcion completa
{transcripcion}

## Formato del delta (tu UNICA salida)
{{
  "threads": [
    {{
      "slug": "slug-existente-o-nuevo",
      "title": "solo obligatorio si el hilo es NUEVO",
      "mode": "cumulative | replace (solo si el hilo es NUEVO)",
      "synthesis": "opcional: nueva sintesis del hilo si cambia",
      "watchlist": "opcional",
      "state": "opcional (solo hilo del animo): green | amber | red",
      "state_rationale": "opcional: 1-2 frases",
      "episode": {{"date": "{ep_date}", "title": "..."}},
      "replace_entries": [entrada, ...],
      "new_entries": [entrada, ...],
      "updated_entries": [{{"match_title": "titulo EXACTO de la entrada existente", "entry": entrada_completa_reescrita}}]
    }}
  ]
}}
entrada = {{"title": str, "dates": ["YYYY-MM-DD", ...], "text": str, "summary": str, "quote"?: str, "group"?: str, "tags"?: [str], "links"?: [...]}}

## Reglas
- Usa EXACTAMENTE los slugs existentes del JSON actual. No inventes slugs nuevos para hilos que ya existen.
- Hilos "replace" (p. ej. temas de la ultima edicion): usa replace_entries con la lista completa nueva (episodio {ep_date}), e incluye "episode".
- Hilos "cumulative": si un tema (saga) YA tiene entrada, usa updated_entries (match_title = titulo exacto actual) y en la entrada reescrita AGREGA "{ep_date}" a dates y actualiza text/summary; NO la dupliques en new_entries. Temas realmente nuevos van en new_entries con dates=["{ep_date}"].
- CADA entrada (nueva o reescrita) lleva "summary": resumen ejecutivo denso en hechos (actores, fechas, cifras), menos de 100 palabras, en espanol.
- Hilo del animo sobre la viabilidad del Peru ({mood_subject}): emite state y state_rationale reflejando el animo VIGENTE (no el promedio historico), basado en los comentarios editoriales de la transcripcion; incluye una cita breve en quote de la entrada si aplica. Cambia el estado solo si el tono lo justifica.
- Etiqueta "hapag-lloyd" en tags cuando el tema toque puertos, comercio exterior, commodities, tipo de cambio, fletes o politica economica peruana.
{week_note}- Omite del delta todo hilo sin cambios. NO toques meta (el sistema la actualiza).
- Ortografia espanola completa (tildes y enes). NO uses em-dash ni el simbolo de tilde solo. Montos en dolares formato USD 250,500.50. No inventes nada que no este en las fuentes.

## Salida
UNICAMENTE el JSON del delta. Sin texto antes o despues, sin ```."""


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")
    return json.loads(text[start:end + 1])


def validate_delta(delta, ep_date):
    if not isinstance(delta, dict) or not isinstance(delta.get("threads"), list):
        raise ValueError("Delta has no threads list.")
    if not delta["threads"]:
        raise ValueError("Delta is empty (no thread changes for the episode).")
    for th in delta["threads"]:
        if not th.get("slug"):
            raise ValueError("Delta thread without slug.")
        for ent in (th.get("replace_entries") or []) + (th.get("new_entries") or []) + \
                   [u.get("entry") for u in th.get("updated_entries") or []]:
            if not isinstance(ent, dict) or not ent.get("title"):
                raise ValueError(f"Invalid entry in thread {th['slug']}.")
            if not ent.get("summary"):
                raise ValueError(f"Entry '{ent.get('title')}' in {th['slug']} lacks summary.")
            if not ent.get("dates"):
                raise ValueError(f"Entry '{ent.get('title')}' in {th['slug']} lacks dates.")


def opus_delta(client, model, current_doc, hilos, source_name, mood_subject,
               ep_date, reporte, transcripcion):
    prompt = build_prompt(current_doc, hilos, source_name, mood_subject,
                          ep_date, reporte, transcripcion)
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with client.messages.stream(
                model=model,
                max_tokens=32000,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                final = stream.get_final_message()
            text = "".join(b.text for b in final.content if getattr(b, "type", "") == "text")
            delta = extract_json(text)
            validate_delta(delta, ep_date)
            return delta
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            print(f"  Attempt {attempt}/{MAX_ATTEMPTS} for {ep_date} returned a bad delta: {e}")
    raise RuntimeError(f"Could not get a valid delta for {ep_date}: {last_err}")


# --- Deterministic merge ------------------------------------------------------
def entry_max_date(e):
    return max(e.get("dates") or [""])


def merge_delta(current, delta, ep_date, today, source_name):
    doc = current
    meta = doc.setdefault("meta", {})
    meta.setdefault("title", "Seguimiento de hilos")
    meta.setdefault("source", source_name)
    threads = doc.setdefault("threads", [])

    for th in delta["threads"]:
        slug = th["slug"]
        t = next((x for x in threads if x.get("slug") == slug), None)
        if t is None:
            if not th.get("title"):
                print(f"  WARNING: delta references unknown thread '{slug}' without title; skipping it.")
                continue
            t = {"id": max([x.get("id") or 0 for x in threads], default=0) + 1,
                 "slug": slug, "title": th["title"],
                 "mode": th.get("mode") or "cumulative", "entries": []}
            threads.append(t)
        for field in ("synthesis", "watchlist", "state", "state_rationale", "episode"):
            if th.get(field) is not None:
                t[field] = th[field]
        entries = t.setdefault("entries", [])

        if th.get("replace_entries") is not None:
            entries[:] = th["replace_entries"]
        else:
            for u in th.get("updated_entries") or []:
                target = next((e for e in entries if (e.get("title") or "").strip() ==
                               (u.get("match_title") or "").strip()), None)
                if target is None:
                    print(f"  WARNING: no entry titled '{u.get('match_title')}' in {slug}; adding as new.")
                    entries.append(u["entry"])
                else:
                    target.clear()
                    target.update(u["entry"])
            for ne in th.get("new_entries") or []:
                dup = next((e for e in entries if (e.get("title") or "").strip() ==
                            (ne.get("title") or "").strip()), None)
                if dup is not None:
                    dup.clear()
                    dup.update(ne)
                else:
                    entries.append(ne)
        entries.sort(key=entry_max_date, reverse=True)

    pe = sorted(set((meta.get("processed_episodes") or [])) | {ep_date})
    meta["processed_episodes"] = pe
    meta["last_update"] = today
    meta["last_episode"] = pe[-1]
    meta["episodes_covered"] = len(pe)
    meta["date_range"] = f"{pe[0]} a {pe[-1]}"
    return doc


# --- Orchestration -----------------------------------------------------------
def main():
    source_name = cfg("SOURCE_NAME", required=True)
    mood_subject = cfg("MOOD_SUBJECT", required=True)
    daily_folder = cfg("DAILY_NEWS_FOLDER_ID", required=True)
    archivos_folder = cfg("ARCHIVOS_FOLDER_ID", required=True)
    output_path = cfg("OUTPUT_PATH", required=True)
    dash_json_url = cfg("DASH_JSON_URL", required=True)
    model = cfg("ANTHROPIC_MODEL", "claude-opus-4-8")
    today = lima_today()

    svc = drive_service()
    episodes = collect_episodes(svc, archivos_folder)
    if not episodes:
        print("No episode files in Archivos; nothing to do.")
        return

    current = fetch_published_json(dash_json_url) or dict(EMPTY_DOC)
    already = processed_dates(current)
    new_dates = sorted(d for d in episodes if d not in already)

    min_date = cfg("MIN_EPISODE_DATE")
    if min_date:
        skipped = [d for d in new_dates if d < min_date]
        if skipped:
            print(f"Skipping {len(skipped)} episode(s) older than MIN_EPISODE_DATE={min_date}: {skipped}")
        new_dates = [d for d in new_dates if d >= min_date]

    if not new_dates:
        print(f"No new episodes. {len(already)} already processed. Nothing to publish.")
        return

    print(f"New episodes to process (chronological): {new_dates}")
    hilos = read_hilos(svc, daily_folder)
    client = anthropic_client()

    published_any = False
    for d in new_dates:
        parts = episodes[d]
        if "reporte" not in parts or "transcripcion" not in parts:
            print(f"  {d}: missing reporte or transcripcion, skipping this episode.")
            continue
        reporte = drive_download_text(svc, parts["reporte"])
        transcripcion = drive_download_text(svc, parts["transcripcion"])
        prev_count = len(processed_dates(current))
        print(f"  Processing {d} with {model} (delta mode) ...")
        delta = opus_delta(client, model, current, hilos, source_name,
                           mood_subject, d, reporte, transcripcion)
        updated = merge_delta(current, delta, d, today, source_name)

        # Quality gates (merge is deterministic, but keep the belt and braces).
        if d not in processed_dates(updated):
            sys.exit(f"{d} not registered after merge; aborting.")
        if len(processed_dates(updated)) != prev_count + 1:
            sys.exit(f"Episode count did not grow by 1 for {d}; aborting.")
        if not updated.get("threads"):
            sys.exit(f"Result for {d} has no threads; aborting.")

        publish(updated, output_path)          # publish per episode so progress persists
        current = updated
        published_any = True
        changed = [t["slug"] for t in delta["threads"]]
        print(f"  Published {output_path} through {d}: "
              f"{len(processed_dates(current))} episodes, {len(current['threads'])} threads. "
              f"Changed threads: {changed}")

    print("Done." if published_any else "Nothing published.")


if __name__ == "__main__":
    main()
