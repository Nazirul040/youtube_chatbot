import os
from dotenv import load_dotenv
load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- Fetching Transcript ---
video_id = "aircAruvnKk"
try:
    yt_api = YouTubeTranscriptApi()
    transcript = yt_api.fetch(video_id=video_id, languages=['en'])
    text = " ".join(snippet.text for snippet in transcript)
except Exception as e:
    print(f"Transcript not available / YouTube API limit reached")
    raise SystemExit(1)

# --- Splitter ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.create_documents([text])

# --- Embedding and Vector Database ---
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

persist_directory = 'chroma_db'
collection_name = 'my_collection_yt'

if os.path.isdir(persist_directory) and os.listdir(persist_directory):
    vector_store = Chroma(
        embedding_function=embedder,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
else:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

# --- Retriever ---
retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k': 3,
        'fetch_k': 5,
        'lambda_mult': 0.5
    }
)

def context_text(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# --- Prompt Template ---
prompt = PromptTemplate(
    template=(
        'You are a helpful assistant. Answer ONLY from the provided transcript '
        'context. If the context is insufficient, just say dont know. '
        'Context - {context}\n Question - {question}'
    ),
    input_variables=['context', 'question']
)

# --- LLM ---
llm = ChatNVIDIA(
    model="nvidia/nemotron-3-nano-30b-a3b",
    temperature=0.4,
    max_completion_tokens=1500
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(context_text),
    'question': RunnablePassthrough()
})
chain = parallel_chain | prompt | llm | parser

# --- Chat loop ---
while True:
    question = input('You : ')

    if question.lower() == 'quit':
        break

    result = chain.invoke(question)
    print(result)