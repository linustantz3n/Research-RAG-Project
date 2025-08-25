import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


CHROMA_PATH = "chroma"


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def find_similar_chunks(query_text):
    embedding_func = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)

    results = db.similarity_search_with_relevance_scores(query_text, k=4)
    if len(results) == 0 or results[0][1] < 0.65:
        print(f"unable to find results")
        return

    return results


def main():
    # Test query instead of interactive input
    query_text = input("Question: ")
    results = find_similar_chunks(query_text)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI()
    response_text = model.invoke(prompt).content

    print(f"Response: {response_text}\n---\nRelevant Quotes:\n{context_text}")


if __name__ == '__main__':
    
    main()