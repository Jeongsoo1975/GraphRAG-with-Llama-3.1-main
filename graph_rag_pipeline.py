# graph_rag_pipeline.py

# ==============================================================================
# 1. 패키지 설치 (사전 준비)
# ==============================================================================
# 아래 주석 처리된 패키지들을 스크립트 실행 전에 미리 설치해야 합니다.
# 터미널에서 다음 명령어를 실행하세요:
# pip install langchain langchain-community langchain-ollama langchain-experimental neo4j tiktoken python-dotenv json-repair langchain-openai langchain_core

# ==============================================================================
# 2. 라이브러리 임포트
# ==============================================================================
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions

# ==============================================================================
# 3. 환경 설정 및 초기화
# ==============================================================================
# .env 파일에서 환경 변수 로드
load_dotenv()

# Ollama 및 Neo4j 클라이언트 초기화
# 참고: 스크립트 실행 전 Ollama 서버가 실행 중이어야 하며,
# 'llama3.1'과 'mxbai-embed-large' 모델이 설치되어 있어야 합니다.
# (예: ollama pull llama3.1, ollama pull mxbai-embed-large)
llm = OllamaFunctions(model="llama3.1", temperature=0, format="json")
graph = Neo4jGraph()
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# ==============================================================================
# 4. 데이터 로드 및 그래프 생성/저장
# ==============================================================================
def load_and_process_data():
    """텍스트 파일을 로드하고 그래프를 생성하여 Neo4j에 저장합니다."""
    print("데이터 로딩 및 청크 분할을 시작합니다...")
    loader = TextLoader(file_path="dummytext.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
    documents = text_splitter.split_documents(documents=docs)
    print(f"총 {len(documents)}개의 문서 청크가 생성되었습니다.")

    print("LLM을 사용하여 텍스트에서 그래프 문서를 추출합니다...")
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    print(f"총 {len(graph_documents)}개의 그래프 문서가 생성되었습니다.")
    # print("\n--- 샘플 그래프 문서 ---")
    # print(graph_documents[0])
    # print("----------------------\n")


    print("생성된 그래프를 Neo4j 데이터베이스에 추가합니다...")
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    print("그래프 저장이 완료되었습니다.")

# ==============================================================================
# 5. 인덱스 생성 (Vector & Full-text)
# ==============================================================================
def create_indexes():
    """Neo4j에 벡터 인덱스와 Full-text 인덱스를 생성합니다."""
    print("Neo4j 벡터 인덱스를 생성/로드합니다...")
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    
    print("Neo4j Full-text 인덱스를 생성합니다...")
    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )

    def create_fulltext_index(tx):
        query = """
        CREATE FULLTEXT INDEX `fulltext_entity_id` IF NOT EXISTS
        FOR (n:__Entity__)
        ON EACH [n.id];
        """
        tx.run(query)

    try:
        with driver.session() as session:
            session.execute_write(create_fulltext_index)
            print("Full-text 인덱스가 성공적으로 생성되었거나 이미 존재합니다.")
    finally:
        driver.close()
        
    return vector_index

# ==============================================================================
# 6. RAG 체인 및 리트리버 함수 정의
# ==============================================================================

# 질문에서 개체를 추출하기 위한 Pydantic 모델 및 프롬프트
class Entities(BaseModel):
    """Identifying information about entities."""
    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

entity_chain = llm.with_structured_output(Entities)

# Graph 리트리버 함수
def graph_retriever(question: str) -> str:
    """질문에 언급된 개체의 주변 정보를 그래프에서 수집합니다."""
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result

# 하이브리드 리트리버 함수
def full_retriever(question: str, vector_retriever):
    """그래프와 벡터 검색 결과를 결합하여 최종 컨텍스트를 생성합니다."""
    print("\n--- 그래프 리트리버 실행 중... ---")
    graph_data = graph_retriever(question)
    print("그래프 데이터:\n", graph_data)
    
    print("\n--- 벡터 리트리버 실행 중... ---")
    vector_data_docs = vector_retriever.invoke(question)
    vector_data = [el.page_content for el in vector_data_docs]
    print("벡터 데이터 (첫 번째 청크):", vector_data[0] if vector_data else "결과 없음")

    final_data = f"""Graph data:
{graph_data}
vector data:
{"#Document ".join(vector_data)}
    """
    return final_data

# 최종 답변 생성 체인
def create_rag_chain(vector_index):
    """최종 RAG 체인을 구성합니다."""
    vector_retriever = vector_index.as_retriever()
    
    template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": lambda x: full_retriever(x["question"], vector_retriever),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ==============================================================================
# 7. 메인 실행 블록
# ==============================================================================
if __name__ == "__main__":
    # 1단계: 데이터 로드 및 그래프 생성 (필요 시 한 번만 실행)
    # 이미 데이터가 Neo4j에 있다면 이 부분을 주석 처리할 수 있습니다.
    # graph.query("MATCH (n) DETACH DELETE n") # 기존 데이터 초기화가 필요할 경우 주석 해제
    # load_and_process_data()

    # 2단계: 인덱스 생성 및 RAG 체인 구성
    vector_index = create_indexes()
    chain = create_rag_chain(vector_index)

    # 3단계: 체인 실행 및 질문에 대한 답변 받기
    question = "Who is Nonna Lucia? Did she teach anyone about restaurants or cooking?"
    print(f"\n\n질문: {question}")
    
    # RunnablePassthrough를 사용하지 않고 직접 딕셔너리를 전달합니다.
    answer = chain.invoke({"question": question})
    
    print("\n--- 최종 답변 ---")
    print(answer)
    print("------------------")\