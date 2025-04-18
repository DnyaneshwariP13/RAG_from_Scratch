import os
import streamlit as st
from dotenv import load_dotenv
from utils.transcription import get_video_id, get_transcript, save_transcript
from utils.vector_store import get_embeddings, process_documents
from models.model import create_qa_chain
from evaluation.evaluator import QAEvaluator
# Load environment variables
load_dotenv()

def main():
    st.title("YouTube Video QA Assistant")
    
    # Initializing session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    '''    
    if st.sidebar.button("Run Evaluation"):
        with st.spinner("Running QA evaluation..."):
            try:
                qa_chain = create_qa_chain(
                    st.session_state.vector_store.as_retriever()
                )
                evaluator = QAEvaluator(qa_chain)
                
                df = evaluator.evaluate("evaluation/test_cases.json")
                report = evaluator.summary_report(df)
                
                st.subheader("Evaluation Results")
                st.write("### Aggregate Metrics")
                st.metric("Average Exact Match", f"{report['avg_exact_match']:.2%}")
                st.metric("Average F1 Score", f"{report['avg_f1_score']:.2%}")
                st.metric("Semantic Similarity", f"{report['avg_semantic_sim']:.2%}")
                st.metric("Context Relevance", f"{report['avg_context_rel']:.2%}")
                
                st.write("### Detailed Results")
                st.dataframe(df)
                
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")
    '''
    
    # insert url
    url = st.text_input("Enter YouTube URL:", placeholder="https://youtube.com/watch?v=...")
    
    if st.button("Process Video"):
        with st.spinner("Processing video..."):
            try:
                # Get transcript
                video_id = get_video_id(url)
                transcript = get_transcript(video_id)
                
                if transcript:
                    # Save transcript
                    save_transcript(transcript)
                    
                    # Initialize embeddings
                    embeddings = get_embeddings()
                    
                    # Process documents and create vector store
                    st.session_state.vector_store = process_documents(embeddings)
                    
                    # Display preview
                    st.subheader("Transcript Preview")
                    st.text(transcript[:1000] + "...")
                    
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
    
    # Question answering section
    if st.session_state.vector_store:
        question = st.text_input("Ask questions about the video content:", 
                               placeholder="What was discussed about...?")
        
        if question:
            with st.spinner("Searching for answer..."):
                try:
                    qa_chain = create_qa_chain(
                        st.session_state.vector_store.as_retriever()
                    )
                    answer = qa_chain.invoke(question)
                    st.subheader("Answer")
                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    
    # Download section
    if os.path.exists("transcript.txt"):
        with open("transcript.txt", "r") as f:
            st.download_button(
                label="Download Transcript",
                data=f,
                file_name="transcript.txt",
                mime="text/plain"
            )
    

if __name__ == "__main__":
    main()