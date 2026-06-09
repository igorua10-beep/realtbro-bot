"""
Real Estate AI Matching System
FastAPI + Aiogram 3.x + Qdrant + OpenAI Embeddings + GPT Query Expansion
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import os
import re
import pandas as pd

from fastapi import FastAPI
from contextlib import asynccontextmanager

from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# ── Credentials ───────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL",     "https://YOUR_CLUSTER.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "YOUR_QDRANT_API_KEY")

ADMIN_IDS_RAW  = os.getenv("ADMIN_IDS", "")
ADMIN_IDS: set[int] = {int(x) for x in ADMIN_IDS_RAW.split(",") if x.strip()}

COLLECTION_NAME = "realtbro_clients"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536
CSV_PATH        = "database.csv"
TOP_K           = 10

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COL = {
    "id":        "Код заявки",
    "name":      "Назва",
    "category":  "Категорія об'єкта",
    "deal":      "Тип угоди",
    "type":      "Тип нерухомості",
    "area":      "Площа загальна",
    "district":  "Райони",
    "location":  "Локації",
    "price":     "Ціна",
    "price_per": "Ціна за",
    "currency":  "Валюта ціни",
    "rooms":     "Кількість кімнат",
    "phone":     "Номер телефону",
    "comment":   "Додаткова інформація",
    "status":    "статус",
}

ARCHIVE_WORDS = ["архів", "архівні", "відмінені", "відмінена", "закриті"]

GPT_SYSTEM_PROMPT = """Ти — асистент ріелторського агентства в Хмельницькому.
Твоя задача — розширити короткий запит менеджера до детального опису що шукає клієнт.

Правила:
- Відповідай ТІЛЬКИ розширеним текстом запиту, без пояснень
- Додавай типові деталі для такого типу нерухомості (поверх, тип приміщення, оренда чи купівля)
- Використовуй українську мову
- Максимум 3-4 речення

Приклади:
Запит: "аптека"
Відповідь: "Приміщення під аптеку в оренду. Торгова площа від 50 до 120 м². Перший поверх, прохідне місце з гарним трафіком. Різні райони міста."

Запит: "склад з рампою"
Відповідь: "Складське приміщення в оренду з рампою для заїзду вантажівок. Площа від 300 м², висока стеля. Промислова зона або околиці міста."

Запит: "офіс центр 30 метрів"
Відповідь: "Офісне приміщення в оренду в центрі міста. Площа близько 30 м². Перший або другий поверх, окремий вхід або в бізнес-центрі."
"""


async def expand_query(query: str) -> str:
    """Use GPT to expand a short query into a detailed description."""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            temperature=0.3,
        )
        expanded = response.choices[0].message.content.strip()
        logger.info("Query expanded: '%s' -> '%s'", query, expanded)
        return expanded
    except Exception as e:
        logger.warning("GPT expansion failed, using original query: %s", e)
        return query


def normalize_range(val: str, unit: str = "") -> str:
    val = val.strip()
    if not val or val == "-":
        return ""
    if re.match(r'^-\d', val):
        return f"до {val[1:]}{unit}"
    if re.match(r'^\d+-$', val):
        return f"від {val[:-1]}{unit}"
    if re.match(r'^\d+[\.,]?\d*-\d+[\.,]?\d*$', val):
        parts = val.split("-")
        return f"від {parts[0]} до {parts[1]}{unit}"
    return f"{val}{unit}"


def build_text(row: pd.Series) -> str:
    parts = []
    name = str(row.get(COL["name"], "") or "").strip()
    if name:
        parts.append(f"Запит: {name}")
    cat = str(row.get(COL["category"], "") or "").strip()
    if cat:
        parts.append(f"Категорія: {cat}")
    deal = str(row.get(COL["deal"], "") or "").strip()
    if deal:
        parts.append(f"Тип угоди: {deal}")
    prop_type = str(row.get(COL["type"], "") or "").strip()
    if prop_type:
        parts.append(f"Тип нерухомості: {prop_type}")
    area = normalize_range(str(row.get(COL["area"], "") or ""), "м²")
    if area:
        parts.append(f"Площа: {area}")
    rooms = str(row.get(COL["rooms"], "") or "").strip()
    if rooms and rooms != "-":
        parts.append(f"Кімнат: {rooms}")
    price = normalize_range(str(row.get(COL["price"], "") or ""))
    price_per = str(row.get(COL["price_per"], "") or "").strip()
    currency = str(row.get(COL["currency"], "") or "").strip()
    if price:
        parts.append(f"Ціна: {price} {currency} {price_per}".strip())
    location = str(row.get(COL["location"], "") or "").strip()
    if location and location not in ("-", "Хмельницкий"):
        parts.append(f"Район/локація: {location}")
    district = str(row.get(COL["district"], "") or "").strip()
    if district and district != "-":
        parts.append(f"Район: {district}")
    comment = str(row.get(COL["comment"], "") or "").strip()
    if comment:
        parts.append(f"Коментар: {comment}")
    return ". ".join(p for p in parts if p)


async def embed(text: str) -> list[float]:
    response = await openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


async def ensure_collection() -> None:
    existing = [c.name for c in (await qdrant_client.get_collections()).collections]
    if COLLECTION_NAME not in existing:
        await qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("Collection '%s' created.", COLLECTION_NAME)
    else:
        logger.info("Collection '%s' already exists.", COLLECTION_NAME)
    await qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="status",
        field_schema="keyword",
    )


async def index_csv(path: str = CSV_PATH) -> None:
    df = pd.read_csv(path, dtype=str).fillna("")
    await ensure_collection()
    batch_size = 50
    points: list[PointStruct] = []
    for i, (_, row) in enumerate(df.iterrows()):
        text = build_text(row)
        if not text.strip():
            continue
        vector = await embed(text)
        status = str(row.get(COL["status"], "")).strip().lower()
        points.append(PointStruct(
            id=i,
            vector=vector,
            payload={
                "client_id": row.get(COL["id"],       ""),
                "name":      row.get(COL["name"],     ""),
                "phone":     row.get(COL["phone"],    ""),
                "comment":   row.get(COL["comment"],  ""),
                "status":    status,
                "price":     row.get(COL["price"],    ""),
                "area":      row.get(COL["area"],     ""),
                "location":  row.get(COL["location"], ""),
                "text":      text,
            },
        ))
        if len(points) >= batch_size:
            await qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info("Upserted %d points...", i + 1)
            points = []
    if points:
        await qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Indexing complete. Total records: %d", i + 1)


async def search_clients(query: str, top_k: int = TOP_K, status_filter: str = "активна") -> tuple[str, list[dict]]:
    expanded = await expand_query(query)
    vector = await embed(expanded)
    qdrant_filter = Filter(
        must=[FieldCondition(key="status", match=MatchValue(value=status_filter))]
    )
    results = await qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
        query_filter=qdrant_filter,
    )
    return expanded, [
        {"score": round(hit.score, 4), **hit.payload}
        for hit in results.points
    ]


bot = Bot(token=TELEGRAM_TOKEN)
dp  = Dispatcher()


def is_admin(user_id: int) -> bool:
    return not ADMIN_IDS or user_id in ADMIN_IDS


def format_results(results: list[dict], query: str, expanded: str) -> str:
    if not results:
        return "Нічого не знайдено."
    lines = [
        f"🔎 Запит розширено:\n_{expanded}_\n",
        f"🏠 Топ-{len(results)} клієнтів:\n"
    ]
    for rank, r in enumerate(results, start=1):
        area = normalize_range(r.get("area", ""), "м²") or "—"
        price = normalize_range(r.get("price", "")) or "—"
        lines.append(
            f"{rank}. {r.get('name', '—')}\n"
            f"   📞 {r.get('phone', '—')}\n"
            f"   📐 {area} | 💰 {price}\n"
            f"   📍 {r.get('location', '—')}\n"
            f"   Score: {r.get('score')}"
        )
    return "\n".join(lines)


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    if not is_admin(message.from_user.id):
        return
    await message.answer(
        "👋 Привіт! Надішли опис об'єкту — знайду клієнтів.\n\n"
        "Команди:\n"
        "/top10 <текст> — топ 10\n"
        "/top20 <текст> — топ 20\n"
        "/reindex — оновити базу\n\n"
        "Для архіву додай слово 'архів' в запит.\n"
        "Або просто напиши текст 🎤"
    )


@dp.message(Command("reindex"))
async def cmd_reindex(message: Message) -> None:
    if not is_admin(message.from_user.id):
        return
    await message.answer("⏳ Починаю індексацію…")
    try:
        await index_csv()
        await message.answer("✅ Індексацію завершено.")
    except Exception as exc:
        logger.exception("Reindex failed")
        await message.answer(f"❌ Помилка: {exc}")


@dp.message(Command("top10"))
async def cmd_top10(message: Message) -> None:
    if not is_admin(message.from_user.id):
        return
    query = (message.text or "").removeprefix("/top10").strip()
    if not query:
        await message.answer("Введи опис після /top10")
        return
    await _do_search(message, query, top_k=10)


@dp.message(Command("top20"))
async def cmd_top20(message: Message) -> None:
    if not is_admin(message.from_user.id):
        return
    query = (message.text or "").removeprefix("/top20").strip()
    if not query:
        await message.answer("Введи опис після /top20")
        return
    await _do_search(message, query, top_k=20)


@dp.message(Command("search"))
async def cmd_search(message: Message) -> None:
    if not is_admin(message.from_user.id):
        return
    query = (message.text or "").removeprefix("/search").strip()
    if not query:
        await message.answer("Введи опис після /search")
        return
    await _do_search(message, query)


@dp.message(lambda m: m.voice is not None)
async def handle_voice(message: Message) -> None:
    if not is_admin(message.from_user.id):
        return
    await message.answer("🎤 Транскрибую голосове…")
    try:
        file = await bot.get_file(message.voice.file_id)
        file_bytes = await bot.download_file(file.file_path)
        transcript = await openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=("voice.ogg", file_bytes, "audio/ogg"),
            language="uk",
        )
        query = transcript.text.strip()
        await message.answer(f"🗣 Розпізнано: «{query}»")
        await _do_search(message, query)
    except Exception as exc:
        logger.exception("Voice failed")
        await message.answer(f"❌ Помилка голосу: {exc}")


@dp.message()
async def handle_text(message: Message) -> None:
    if not is_admin(message.from_user.id):
        return
    query = (message.text or "").strip()
    if query:
        await _do_search(message, query)


async def _do_search(message: Message, query: str, top_k: int = TOP_K) -> None:
    if any(w in query.lower() for w in ARCHIVE_WORDS):
        status = "відмінена"
        await message.answer("🗂 Шукаю серед відмінених клієнтів…")
    else:
        status = "активна"
        await message.answer("🔍 Аналізую запит і шукаю клієнтів…")
    try:
        expanded, results = await search_clients(query, top_k=top_k, status_filter=status)
        await message.answer(format_results(results, query, expanded))
    except Exception as exc:
        logger.exception("Search failed")
        await message.answer(f"❌ Помилка пошуку: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(dp.start_polling(bot, handle_signals=False))
    yield
    await bot.session.close()


app = FastAPI(title="RealT Bro AI Matching", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/index")
async def trigger_index():
    asyncio.create_task(index_csv())
    return {"status": "indexing started"}


if __name__ == "__main__":
    async def main():
        await ensure_collection()
        await index_csv()
        await dp.start_polling(bot)

    asyncio.run(main())