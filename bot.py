from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, util
import streamlit as st


st.set_page_config(
    page_title="UPL Assistant", page_icon="⚽", layout="centered"
)

st.title("RAG-бот")
st.write(
    "Система пошуку та аналізу новин української Прем'єр-ліги в"
    " реальному часі."
)



@st.cache_resource
def load_and_scrape_data():
  url = "https://upl.ua/ua"
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
          " like Gecko) Chrome/120.0.0.0 Safari/537.36"
      )
  }

  chunks = []
  try:
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
      soup = BeautifulSoup(response.text, "html.parser")
      elements = soup.find_all([
          "p",
          "h2",
          "h3",
          "span",
          "a",
          "td",
          "th",
          "tr",
          "div",
      ])
      for el in elements:
        text = el.get_text(separator=" ", strip=True)
        if 4 < len(text) < 300:
          chunks.append(text)

    if not chunks:
      chunks = [
          (
              "Офіційний сайт Української Прем'єр-ліги містить розклад матчів,"
              " рахунки зустрічей та турнірну таблицю."
          )
      ]
  except Exception as e:
    chunks = [
        (
            f"Не вдалося підключитися до сайту через мережу ({e}). Використовуються"
            " базові дані."
        )
    ]

  return list(set(chunks))


@st.cache_resource
def load_model():
  return SentenceTransformer("all-MiniLM-L6-v2")


with st.spinner("Завантаження даних та інндексація бази знань..."):
  corpus = load_and_scrape_data()
  model = load_model()
  corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

st.success(
    f"Базу успішно сформовано! Ороблено інформаційних блоків: {len(corpus)}."
)


user_query = st.text_input(
    "Введіть запитання :"
)

if user_query:
  with st.spinner("Аналізую актуальні дані..."):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    hits = util.semantic_search(
        query_embedding, corpus_embeddings, top_k=4
    )[0]

    matched_texts = []
    for hit in hits:
      text = corpus[hit["corpus_id"]]
      if text not in matched_texts:
        matched_texts.append(text)

  
    summary_text = (
        "За результатами останніх даних з офіційного джерела: "
        + " ".join(matched_texts)
    )

  st.markdown("### 📝 Зведена відповідь системи:")
  st.success(summary_text)

  with st.expander("Переглянути знайдені первинні фрагменти (джерела):"):
    for i, text in enumerate(matched_texts):
      st.markdown(f"**{i+1}.** {text}")