"""
Варіант 14. Агент-рекомендатор фільмів і книжок за настроєм користувача.
Агент з'ясовує настрій і тип (фільм/книжка), викликає інструмент recommend,
який повертає твори з каталогу (з жанром і коротким сюжетом), і формує відповідь.

Встановлення:  pip install openai huggingface_hub gradio
Backend:       BACKEND=groq (GROQ_API_KEY, безкоштовно) | hf (HF_TOKEN) | openai (OPENAI_API_KEY)
               для groq потрібно: pip install openai gradio
Запуск:        python recommender_agent.py
"""
import json
import os
import re
import requests
import gradio as gr

BACKEND = os.getenv("BACKEND", "openai")  # "openai" або "hf"

# ---------- Каталог ----------
MOODS = ["сумний", "радісний", "втомлений", "стресовий",
         "нудьгую", "романтичний", "енергійний"]

CATALOG = [
    # Фільми
    {"kind": "фільм", "title": "Форрест Гамп (1994)", "genre": "драма, комедія",
     "moods": ["сумний", "втомлений", "нудьгую"],
     "plot": "Добрий чоловік із невисоким IQ випадково стає учасником ключових подій "
             "історії США й усе життя кохає подругу дитинства Дженні."},
    {"kind": "фільм", "title": "Амелі (2001)", "genre": "романтична комедія",
     "moods": ["радісний", "романтичний", "нудьгую"],
     "plot": "Мрійлива парижанка Амелі таємно допомагає людям навколо бути щасливішими, "
             "а тим часом сама шукає кохання."},
    {"kind": "фільм", "title": "Паразити (2019)", "genre": "трилер, чорна комедія",
     "moods": ["нудьгую", "енергійний"],
     "plot": "Бідна родина поступово влаштовується на роботу в дім багатих господарів, "
             "але стосунки двох світів виходять з-під контролю."},
    {"kind": "фільм", "title": "Мій сусід Тоторо (1988)", "genre": "анімація, сімейне фентезі",
     "moods": ["втомлений", "стресовий", "сумний"],
     "plot": "Дві сестри переїжджають у сільський будинок і знайомляться з добрими "
             "лісовими духами."},
    {"kind": "фільм", "title": "Інтерстеллар (2014)", "genre": "наукова фантастика",
     "moods": ["енергійний", "нудьгую"],
     "plot": "Астронавти вирушають крізь червоточину шукати нову домівку для людства, "
             "поки Земля занепадає."},
    {"kind": "фільм", "title": "Перед сходом сонця (1995)", "genre": "романтична драма",
     "moods": ["романтичний", "стресовий"],
     "plot": "Американець і француженка знайомляться в поїзді та проводять ніч, "
             "гуляючи Віднем і розмовляючи про життя."},
    # Книжки
    {"kind": "книжка", "title": "«Маленький принц» (А. де Сент-Екзюпері)",
     "genre": "філософська казка",
     "moods": ["сумний", "втомлений", "стресовий"],
     "plot": "Пілот зустрічає в пустелі хлопчика з далекої планети, який розповідає "
             "про свої подорожі та дружбу з трояндою."},
    {"kind": "книжка", "title": "«Автостопом по Галактиці» (Д. Адамс)",
     "genre": "комедійна фантастика",
     "moods": ["радісний", "нудьгую", "енергійний"],
     "plot": "Артур Дент дивом уникає знищення Землі та мандрує космосом разом "
             "із чудернацьким супутником Фордом Префектом."},
    {"kind": "книжка", "title": "«Гобіт» (Дж. Р. Р. Толкін)", "genre": "фентезі",
     "moods": ["енергійний", "втомлений", "нудьгую"],
     "plot": "Спокійний гобіт Більбо вирушає з гномами у мандрівку, щоб повернути "
             "скарб, який охороняє дракон."},
    {"kind": "книжка", "title": "«Тіні забутих предків» (М. Коцюбинський)",
     "genre": "повість, романтична драма",
     "moods": ["романтичний", "сумний"],
     "plot": "Кохання Івана і Марічки, дітей ворожих родів, розгортається серед "
             "Карпат і має трагічний кінець."},
    {"kind": "книжка", "title": "«Пес Баскервілів» (А. Конан Дойл)", "genre": "детектив",
     "moods": ["нудьгую", "енергійний"],
     "plot": "Шерлок Холмс і доктор Ватсон розслідують смерть сера Чарльза Баскервіля, "
             "пов'язану з легендою про примарного собаку."},
    {"kind": "книжка", "title": "«Гаррі Поттер і філософський камінь» (Дж. Роулінг)",
     "genre": "фентезі",
     "moods": ["радісний", "стресовий", "нудьгую"],
     "plot": "Сирота Гаррі дізнається, що він чарівник, і вирушає навчатися "
             "до школи Гоґвортс."},
]


# ---------- Інструмент ----------
def recommend(mood: str, kind: str = "будь-що", exclude=None) -> str:
    """Повертає до 2 творів, що пасують до настрою, крім уже запропонованих."""
    mood = mood.lower().strip()
    if isinstance(exclude, str):
        exclude = [exclude]
    shown = [e.lower() for e in (exclude or []) if e]

    def is_new(c):
        t = c["title"].lower()
        return not any(s in t or t in s for s in shown)

    pool = [c for c in CATALOG if kind in (c["kind"], "будь-що") and is_new(c)]
    items = [c for c in pool if mood in c["moods"]]
    if len(items) < 2:  # доповнюємо іншими творами потрібного типу
        items += [c for c in pool if c not in items]
    if not items:
        return json.dumps({"info": "У каталозі більше немає нових творів"},
                          ensure_ascii=False)
    return json.dumps(items[:2], ensure_ascii=False)


SYSTEM_PROMPT = f"""Ти агент-рекомендатор фільмів і книжок. Відповідай українською.
Алгоритм:
1. Якщо користувач ще не сказав свій настрій — запитай, який він (один короткий запит).
   Якщо не зрозуміло, що саме він хоче (фільм чи книжку) — уточни це теж.
2. Коли настрій відомий, обери найближче значення зі списку: {", ".join(MOODS)}.
3. Виклич інструмент recommend, args: {{"mood": "<значення зі списку>", "kind": "фільм" | "книжка" | "будь-що", "exclude": ["назви творів, які ти вже радив у цій розмові"]}}.
   Якщо користувач просить «ще ідеї/інші варіанти» — обов'язково передай у exclude
   назви ВСІХ творів, які ти вже пропонував. Якщо інструмент повертає info про те,
   що нових творів немає — чесно скажи про це користувачу.
4. Фінальна відповідь: для кожного твору вкажи назву, жанр і короткий опис сюжету
   (1-2 речення), використовуючи ЛИШЕ дані з інструменту. Не вигадуй інших творів.

Кожна твоя відповідь — ЛИШЕ один JSON-об'єкт без іншого тексту:
{{"tool": "recommend", "args": {{...}}}}   - виклик інструменту
{{"answer": "текст для користувача"}}      - відповідь або уточнювальне питання"""


# ---------- Звернення до моделі ----------
_client = None
_groq_model = None


def pick_groq_model(client) -> str:
    """Бере модель із змінної GROQ_MODEL або обирає з доступних на акаунті."""
    global _groq_model
    if _groq_model:
        return _groq_model
    _groq_model = os.getenv("GROQ_MODEL")
    if not _groq_model:
        ids = [m.id for m in client.models.list().data]
        print("[GROQ] доступні моделі:", ids)
        bad = ("whisper", "guard", "tts", "orpheus", "compound", "embed", "safeguard")
        chat = [i for i in ids if not any(b in i.lower() for b in bad)]
        for key in ("llama-3.3-70b", "llama-4", "gpt-oss-120b", "llama-3.1-8b",
                    "gpt-oss", "qwen", "llama"):
            found = [i for i in chat if key in i.lower()]
            if found:
                _groq_model = found[0]
                break
        else:
            _groq_model = chat[0]
    print("[GROQ] використовується модель:", _groq_model)
    return _groq_model


def llm(messages) -> str:
    global _client
    if BACKEND == "groq":  # безкоштовний ключ: console.groq.com
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                             base_url="https://api.groq.com/openai/v1")
        r = _client.chat.completions.create(model=pick_groq_model(_client),
                                            messages=messages)
        return r.choices[0].message.content
    if BACKEND == "openai":
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        r = _client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return r.choices[0].message.content
    if _client is None:
        from huggingface_hub import InferenceClient
        _client = InferenceClient(model="Qwen/Qwen2.5-72B-Instruct",
                                  token=os.getenv("HF_TOKEN"))
    return _client.chat_completion(messages, max_tokens=600).choices[0].message.content


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


# ---------- Цикл агента ----------
def run_agent(user_message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        content = h["content"]
        if isinstance(content, list):  # новий Gradio: [{"type": "text", "text": "..."}]
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        messages.append({"role": h["role"], "content": content})
    if isinstance(user_message, dict):  # на випадок мультимодального формату
        user_message = user_message.get("text", "")
    messages.append({"role": "user", "content": user_message})

    for _ in range(4):
        reply = llm(messages)
        data = parse_json(reply)
        if data is None:
            return reply
        if "answer" in data:
            return data["answer"]

        args = data.get("args", {})
        try:
            result = recommend(**args)
        except TypeError as e:
            result = f"Невірні аргументи: {e}"
        print(f"[TOOL] recommend({args}) -> {result}")  # лог для звіту

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Результат інструменту: {result}"})
    return "Не вдалося підібрати рекомендацію. Спробуйте переформулювати запит."


if __name__ == "__main__":
    gr.ChatInterface(
        run_agent,
        title="Рекомендатор фільмів і книжок",
        description="Розкажіть про свій настрій — і я підберу, що подивитися або почитати.",
    ).launch()
