from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

def create_qa_chain(retriever):
    """Create QA chain with prompt template"""
    template = """Answer the question based on context below and provide explanation if possible. If you can't answer, reply "I don't know."
    
    Context: {context}
    Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.0-flash-001"
        )
        | StrOutputParser()
    )