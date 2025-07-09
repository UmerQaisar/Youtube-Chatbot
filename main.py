import os
from uuid import uuid4
import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize Pinecone and Embedding
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key)

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction"
)

hugging_model = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task="text-generation")
llm = ChatHuggingFace(llm=hugging_model)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say: "The transcript doesn't contain that information."

--- Transcript Context ---
{context}

--- Question ---
{question}
""",
    input_variables=["context", "question"]
)

parser = StrOutputParser()

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# Streamlit UI
st.set_page_config(page_title="YouTube Video QA", layout="centered")
st.title("🎥 YouTube Video Q&A Assistant")

# Session state for storing values between steps
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Step 1: Enter Video Link
if st.session_state.video_id is None:
    video_link = st.text_input("Paste a YouTube video link:")
    if st.button("Next") and video_link:
        parsed_url = urlparse(video_link)
        video_id = parse_qs(parsed_url.query).get('v', [None])[0]

        if not video_id:
            st.error("Invalid YouTube URL.")
            st.stop()

        safe_video_id = video_id.lower()
        index_name = f"video-{safe_video_id}"

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
        except TranscriptsDisabled:
            st.error("This video has no captions/transcript.")
            st.stop()

        # Chunk the transcript
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.create_documents([transcript])

        # Create or load index
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            index = pc.Index(index_name)
            vector_store = PineconeVectorStore(index=index, embedding=embedding)

            uuids = [str(uuid4()) for _ in range(len(chunks))]
            vector_store.add_documents(documents=chunks, ids=uuids)
        else:
            index = pc.Index(index_name)
            vector_store = PineconeVectorStore(index=index, embedding=embedding)

        st.session_state.video_id = video_id
        st.session_state.vector_store = vector_store
        st.rerun()

# Step 2: Ask Questions
else:
    st.subheader("Ask questions about the video")
    question = st.text_input("Enter your question")
    retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser

    if st.button("Ask"):
        with st.spinner("Thinking..."):
            answer = main_chain.invoke(question)
            st.success(answer)

    if st.button("Start Over"):
        st.session_state.video_id = None
        st.session_state.vector_store = None
        st.rerun()
