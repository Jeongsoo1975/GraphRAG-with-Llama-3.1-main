import os
import json
import webbrowser
from neo4j import GraphDatabase
from pyvis.network import Network
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Neo4j 연결 정보
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
# docker-compose.yaml 파일에서 확인한 올바른 비밀번호 설정
NEO4J_PASSWORD = "your_password"  # docker-compose.yaml에 설정된 비밀번호

print(f"연결 정보: {NEO4J_URI}, 사용자: {NEO4J_USERNAME}")

# 전체 그래프 시각화
def visualize_entire_graph():
    # pyvis 네트워크 객체 생성
    net = Network(height="900px", width="100%", directed=True, notebook=False, bgcolor="#ffffff", font_color="black")
    
    # 노드 그룹별 색상 설정 (노드 추가 시 직접 사용)
    node_colors = {
        "Entity": "#4285F4",
        "__Entity__": "#4285F4",
        "Document": "#DB4437",
        "Chunk": "#0F9D58",
        "Unknown": "#F4B400"
    }
    
    try:
        # Neo4j 연결
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # 전체 노드와 관계 가져오기
            result = session.run("""
                MATCH (n)-[r]->(m)
                RETURN n, r, m
            """)
            
            nodes = set()  # 중복 노드 방지
            edge_count = 0
            
            # 결과 처리
            for record in result:
                source_node = record["n"]
                target_node = record["m"]
                relationship = record["r"]
                
                # Neo4j v5 이상에서는 id 대신 element_id를 사용
                source_id = source_node.element_id
                target_id = target_node.element_id
                
                # 소스 노드 추가
                if source_id not in nodes:
                    nodes.add(source_id)
                    
                    # 노드 라벨 및 그룹 설정
                    labels = list(source_node.labels)
                    node_type = labels[0] if labels else "Unknown"
                    
                    # 노드 이름 결정 로직 개선 - 다양한 속성을 우선순위에 따라 검색
                    if node_type == "__Entity__" or node_type == "Entity":
                        # Entity 노드는 name 속성 우선
                        node_name = get_meaningful_name(source_node, ["name", "title", "label", "value"])
                    elif node_type == "Document":
                        # Document 노드는 title 속성 우선
                        node_name = get_meaningful_name(source_node, ["title", "name", "content", "text"])
                    else:
                        # 기타 노드 유형
                        node_name = get_meaningful_name(source_node, ["name", "title", "id", "value"])
                    
                    # 노드 속성 정보 생성 (마우스 오버 시 표시)
                    title_html = f"<b>{node_type}</b>: {node_name}<br>"
                    for key, value in source_node.items():
                        title_html += f"{key}: {value}<br>"
                    
                    # 노드 색상 선택
                    color = node_colors.get(node_type, "#cccccc")
                    
                    # 노드 크기 (Entity 노드는 더 크게)
                    size = 25 if node_type in ["Entity", "__Entity__"] else 20
                    
                    # 노드 추가
                    net.add_node(source_id, 
                                label=node_name, 
                                title=title_html,
                                color=color,
                                size=size)
                
                # 타겟 노드 추가
                if target_id not in nodes:
                    nodes.add(target_id)
                    
                    # 노드 라벨 및 그룹 설정
                    labels = list(target_node.labels)
                    node_type = labels[0] if labels else "Unknown"
                    
                    # 노드 이름 결정 로직 개선 - 다양한 속성을 우선순위에 따라 검색
                    if node_type == "__Entity__" or node_type == "Entity":
                        # Entity 노드는 name 속성 우선
                        node_name = get_meaningful_name(target_node, ["name", "title", "label", "value"])
                    elif node_type == "Document":
                        # Document 노드는 title 속성 우선
                        node_name = get_meaningful_name(target_node, ["title", "name", "content", "text"])
                    else:
                        # 기타 노드 유형
                        node_name = get_meaningful_name(target_node, ["name", "title", "id", "value"])
                    
                    # 노드 속성 정보 생성 (마우스 오버 시 표시)
                    title_html = f"<b>{node_type}</b>: {node_name}<br>"
                    for key, value in target_node.items():
                        title_html += f"{key}: {value}<br>"
                    
                    # 노드 색상 선택
                    color = node_colors.get(node_type, "#cccccc")
                    
                    # 노드 크기 (Entity 노드는 더 크게)
                    size = 25 if node_type in ["Entity", "__Entity__"] else 20
                    
                    # 노드 추가
                    net.add_node(target_id, 
                                label=node_name, 
                                title=title_html,
                                color=color,
                                size=size)
                
                # 관계(엣지) 추가
                rel_type = relationship.type
                rel_id = relationship.element_id
                
                # 관계 속성 정보 생성 (마우스 오버 시 표시)
                title_html = f"<b>Relationship</b>: {rel_type}<br>"
                for key, value in relationship.items():
                    title_html += f"{key}: {value}<br>"
                
                # 엣지 추가
                net.add_edge(source_id, 
                            target_id, 
                            title=title_html,
                            label=rel_type,
                            arrows="to")
                edge_count += 1
            
            print(f"총 {len(nodes)}개의 노드와 {edge_count}개의 관계를 가져왔습니다.")
        
        # 물리 레이아웃 설정
        net.barnes_hut(
            gravity=-5000,
            central_gravity=0.1,
            spring_length=150,
            spring_strength=0.04,
            damping=0.09,
            overlap=0.2
        )
        
        # 안정화 설정
        net.toggle_stabilization(True)
        
        # 버튼 추가 (물리 레이아웃 조정용)
        net.show_buttons(filter_=['physics'])
        
        # HTML 파일로 저장
        output_file = "neo4j_graph_visualization.html"
        net.save_graph(output_file)
        
        print(f"그래프가 '{output_file}' 파일로 저장되었습니다.")
        
        # 웹 브라우저에서 HTML 파일 열기
        file_path = os.path.abspath(output_file)
        print(f"파일 경로: {file_path}")
        webbrowser.open('file://' + file_path, new=2)
        
        # 드라이버 종료
        driver.close()
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

# 노드에서 의미 있는 이름을 추출하는 함수
def get_meaningful_name(node, property_priorities):
    """
    노드에서 의미 있는 이름을 추출합니다.
    
    Args:
        node: Neo4j 노드
        property_priorities: 속성 우선순위 목록
    
    Returns:
        추출된 이름 문자열
    """
    # 우선순위에 따라 속성 확인
    for prop in property_priorities:
        if prop in node and node[prop]:
            # 텍스트가 너무 길면 잘라냄
            value = str(node[prop])
            if len(value) > 30:
                return value[:27] + "..."
            return value
    
    # 모든 속성 순회
    for key, value in node.items():
        if value and isinstance(value, (str, int, float)):
            value_str = str(value)
            if len(value_str) > 30:
                return value_str[:27] + "..."
            return value_str
    
    # 아무것도 찾지 못한 경우 ID 반환
    return str(node.element_id)

# 노드 유형 및 관계 통계 출력
def print_graph_statistics():
    try:
        # Neo4j 연결
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # 노드 유형별 개수
            node_types = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS node_type, count(*) AS count
                ORDER BY count DESC
            """)
            
            print("노드 유형별 통계:")
            for record in node_types:
                print(f"- {record['node_type']}: {record['count']}개")
            
            # 관계 유형별 개수
            rel_types = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(*) AS count
                ORDER BY count DESC
            """)
            
            print("\n관계 유형별 통계:")
            for record in rel_types:
                print(f"- {record['rel_type']}: {record['count']}개")
            
            # Entity 노드의 속성 확인
            entity_properties = session.run("""
                MATCH (e:__Entity__)
                RETURN keys(e) AS properties
                LIMIT 1
            """)
            
            record = entity_properties.single()
            if record:
                print("\nEntity 노드의 속성 목록:")
                print(", ".join(record["properties"]))
        
        # 드라이버 종료
        driver.close()
        
    except Exception as e:
        print(f"통계 정보 조회 중 오류가 발생했습니다: {e}")

# 메인 실행
if __name__ == "__main__":
    print("Neo4j 그래프 시각화를 시작합니다...")
    
    # .env 파일 확인
    try:
        with open('.env', 'r') as f:
            print(".env 파일이 존재합니다.")
            env_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            for line in env_lines:
                if '=' in line:
                    key = line.split('=')[0].strip()
                    print(f"환경 변수: {key}")
    except FileNotFoundError:
        print(".env 파일이 없습니다. 기본값을 사용합니다.")
    
    print_graph_statistics()
    visualize_entire_graph()
    print("시각화가 완료되었습니다.") 