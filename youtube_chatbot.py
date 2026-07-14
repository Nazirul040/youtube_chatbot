from dotenv import load_dotenv
load_dotenv()

from urllib.parse import urlparse, parse_qs

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- Page setup ---
st.title("SIMPLE YOUTUBE CHATBOT")
st.caption(
    "NOTE - This chatbot answers only from the transcript of the video you provide, "
    "not from the internet. It may give incomplete or incorrect answers if the "
    "transcript doesn't cover your question."
)

PERSIST_DIR = "chroma_db"


def extract_video_id(url: str):
    """Supports youtube.com/watch?v=, youtu.be/, /embed/, and /shorts/ links."""
    if not url:
        return None
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/") or None

    if host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        for prefix in ("/embed/", "/shorts/"):
            if parsed.path.startswith(prefix):
                return parsed.path[len(prefix):].split("/")[0] or None
    return None


@st.cache_resource(show_spinner=False)
def get_embedder():
    return NVIDIAEmbeddings(model="nvidia/nv-embed-v1")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatNVIDIA(
        model="nvidia/nemotron-3-nano-30b-a3b",
        temperature=0.4,
        max_completion_tokens=1500,
    )


@st.cache_resource(show_spinner="Fetching transcript...")
def build_chain(video_id: str):
    yt_api = YouTubeTranscriptApi()
    transcript = yt_api.fetch(video_id=video_id, languages=["en"])
    text = " ".join(snippet.text for snippet in transcript)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    embedder = get_embedder()
    collection_name = f"yt_{video_id}"

    vector_store = Chroma(
        embedding_function=embedder,
        persist_directory=PERSIST_DIR,
        collection_name=collection_name,
    )
    if vector_store._collection.count() == 0:
        vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5},
    )

    def context_text(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Answer ONLY from the provided transcript "
            "context. If the context is insufficient, just say you don't know.\n"
            "Context - {context}\nQuestion - {question}"
        ),
        input_variables=["context", "question"],
    )

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(context_text),
            "question": RunnablePassthrough(),
        }
    )
    return parallel_chain | prompt | get_llm() | StrOutputParser()


# --- Video input ---
url = st.text_input("Enter the video URL").strip()
video_id = extract_video_id(url)

if url and not video_id:
    st.error("Couldn't find a video ID in that URL. Check the link and try again.")
    st.stop()

if not video_id:
    st.info("Paste a YouTube video URL above to get started.")
    st.stop()

st.caption(f"Video ID: {video_id}")

try:
    chain = build_chain(video_id)
except Exception as e:
    st.error(f"Couldn't process this video: {e}")
    st.stop()

# --- Chat ---
question = st.text_input("Enter your question", key="question_input")

if st.button("SUBMIT") and question:
    with st.spinner("Thinking..."):
        try:
            answer = chain.invoke(question)
            st.success("We got your answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error while answering: {e}")