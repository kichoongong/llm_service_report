#!/usr/bin/env python3
"""
향상된 PDF 문서 구조화 처리기
자동차 매뉴얼에 최적화된 문서 구조화 및 임베딩 벡터 저장
"""
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 환경 변수 로드
load_dotenv()


@dataclass
class DocumentSection:
    """문서 섹션 정보를 저장하는 데이터 클래스"""

    title: str
    content: str
    page_number: int
    section_type: str  # 'header', 'content', 'table', 'list', 'warning', 'note'
    level: int  # 헤더 레벨 (1, 2, 3, ...)
    parent_section: Optional[str] = None
    metadata: Dict[str, Any] = None


class EnhancedPDFProcessor:
    """향상된 PDF 문서 구조화 처리기"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pages = []
        self.sections = []
        self.structure_info = {}

        # 자동차 매뉴얼 특화 키워드 패턴
        self.car_keywords = {
            "engine": ["엔진", "모터", "시동", "가속", "연료", "오일", "냉각수"],
            "safety": ["안전", "사고", "에어백", "벨트", "브레이크", "ABS", "ESP"],
            "comfort": ["편의", "시트", "에어컨", "히터", "온도", "조절"],
            "navigation": ["내비게이션", "GPS", "길찾기", "지도", "경로", "목적지"],
            "audio": ["오디오", "음악", "라디오", "스피커", "볼륨", "음질"],
            "lighting": ["조명", "라이트", "불빛", "전조등", "후미등", "실내등"],
            "door": ["문", "창문", "잠금", "열기", "닫기", "자동문"],
            "maintenance": ["점검", "정비", "교체", "수리", "서비스", "체크"],
            "warning": ["주의", "경고", "위험", "주의사항", "주의하세요"],
            "information": ["정보", "참고", "설명", "안내", "도움말"],
        }

        # 섹션 헤더 패턴 (한국어 자동차 매뉴얼에 특화)
        self.header_patterns = [
            r"^[0-9]+\.[0-9]*\s+[가-힣\s]+$",  # 1.1 제목
            r"^[0-9]+\s+[가-힣\s]+$",  # 1 제목
            r"^[A-Z][A-Z\s]+$",  # 대문자 제목
            r"^[가-힣]+[가-힣\s]*:$",  # 한글 제목:
            r"^[가-힣]+[가-힣\s]*\s*\([가-힣\s]+\)$",  # 한글 제목 (부제목)
        ]

        # 표 감지 패턴
        self.table_patterns = [
            r"^\s*\|.*\|.*$",  # 파이프 구분자
            r"^\s*[가-힣\s]+\s+[가-힣\s]+\s+[가-힣\s]+$",  # 3열 이상 정렬
            r"^\s*[가-힣]+.*\s+[0-9]+.*\s+[가-힣]+.*$",  # 데이터 행
        ]

        # 경고/주의사항 패턴
        self.warning_patterns = [
            r"주의[사항]*[:：]",
            r"경고[:：]",
            r"위험[:：]",
            r"주의하세요",
            r"주의[!！]",
            r"경고[!！]",
        ]

    def load_pdf(self) -> List[Document]:
        """PDF 파일을 로드합니다."""
        print(f"📄 PDF 파일 로딩 중: {self.pdf_path}")
        loader = PyMuPDFLoader(self.pdf_path)
        self.pages = loader.load()
        print(f"✅ {len(self.pages)}개 페이지 로드 완료")
        return self.pages

    def detect_section_type(self, content: str) -> str:
        """텍스트 내용을 분석하여 섹션 타입을 감지합니다."""
        content_lower = content.lower()

        # 경고/주의사항 감지
        for pattern in self.warning_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return "warning"

        # 표 감지
        for pattern in self.table_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return "table"

        # 리스트 감지
        if re.search(r"^\s*[•·▪▫-]\s", content, re.MULTILINE):
            return "list"

        # 헤더 감지
        for pattern in self.header_patterns:
            if re.match(pattern, content.strip()):
                return "header"

        return "content"

    def extract_sections_from_page(
        self, page: Document, page_num: int
    ) -> List[DocumentSection]:
        """페이지에서 섹션을 추출합니다."""
        content = page.page_content
        sections = []

        # 페이지를 문단으로 분할
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        current_section = None
        section_level = 1

        for para in paragraphs:
            if not para:
                continue

            section_type = self.detect_section_type(para)

            # 헤더인 경우
            if section_type == "header":
                # 이전 섹션 저장
                if current_section:
                    sections.append(current_section)

                # 새 섹션 시작
                current_section = DocumentSection(
                    title=para,
                    content="",
                    page_number=page_num,
                    section_type=section_type,
                    level=section_level,
                    metadata=page.metadata.copy(),
                )
                section_level += 1

            else:
                # 내용을 현재 섹션에 추가
                if current_section:
                    if current_section.content:
                        current_section.content += "\n\n" + para
                    else:
                        current_section.content = para
                    current_section.section_type = section_type
                else:
                    # 헤더가 없는 내용은 독립 섹션으로 처리
                    current_section = DocumentSection(
                        title=f"페이지 {page_num} 내용",
                        content=para,
                        page_number=page_num,
                        section_type=section_type,
                        level=1,
                        metadata=page.metadata.copy(),
                    )

        # 마지막 섹션 저장
        if current_section:
            sections.append(current_section)

        return sections

    def analyze_document_structure(self) -> Dict[str, Any]:
        """문서의 전체 구조를 분석합니다."""
        print("🔍 문서 구조 분석 중...")

        all_sections = []
        structure_stats = {
            "total_pages": len(self.pages),
            "total_sections": 0,
            "section_types": {},
            "page_sections": {},
            "hierarchy": {},
            "tables": [],
            "warnings": [],
            "lists": [],
        }

        for i, page in enumerate(self.pages):
            page_sections = self.extract_sections_from_page(page, i + 1)
            all_sections.extend(page_sections)
            structure_stats["page_sections"][i + 1] = len(page_sections)

            for section in page_sections:
                # 섹션 타입별 통계
                section_type = section.section_type
                if section_type not in structure_stats["section_types"]:
                    structure_stats["section_types"][section_type] = 0
                structure_stats["section_types"][section_type] += 1

                # 특수 섹션 수집
                if section_type == "table":
                    structure_stats["tables"].append(
                        {
                            "page": i + 1,
                            "title": section.title,
                            "content": (
                                section.content[:100] + "..."
                                if len(section.content) > 100
                                else section.content
                            ),
                        }
                    )
                elif section_type == "warning":
                    structure_stats["warnings"].append(
                        {
                            "page": i + 1,
                            "title": section.title,
                            "content": (
                                section.content[:100] + "..."
                                if len(section.content) > 100
                                else section.content
                            ),
                        }
                    )
                elif section_type == "list":
                    structure_stats["lists"].append(
                        {
                            "page": i + 1,
                            "title": section.title,
                            "content": (
                                section.content[:100] + "..."
                                if len(section.content) > 100
                                else section.content
                            ),
                        }
                    )

        structure_stats["total_sections"] = len(all_sections)
        self.sections = all_sections
        self.structure_info = structure_stats

        print(f"📊 총 섹션 수: {structure_stats['total_sections']}")
        print(f"📊 섹션 타입별 분포: {structure_stats['section_types']}")
        print(f"📊 표 개수: {len(structure_stats['tables'])}")
        print(f"📊 경고사항 개수: {len(structure_stats['warnings'])}")

        return structure_stats

    def categorize_content(self, content: str) -> List[str]:
        """내용을 분석하여 카테고리를 분류합니다."""
        categories = []
        content_lower = content.lower()

        for category, keywords in self.car_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                categories.append(category)

        return categories if categories else ["general"]

    def create_enhanced_chunks(self) -> List[Document]:
        """구조화된 섹션을 기반으로 향상된 청크를 생성합니다."""
        print("✂️ 구조화된 청크 생성 중...")

        chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # 섹션 기반으로 크기 증가
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        for i, section in enumerate(self.sections):
            # 섹션별 메타데이터 생성
            categories = self.categorize_content(section.content)

            # 섹션이 너무 크면 분할
            if len(section.content) > 1000:
                section_chunks = text_splitter.split_text(section.content)
                for j, chunk_text in enumerate(section_chunks):
                    chunk_metadata = {
                        **section.metadata,
                        "section_title": section.title,
                        "section_type": section.section_type,
                        "section_level": section.level,
                        "page_number": section.page_number,
                        "categories": ", ".join(categories),
                        "chunk_index": j,
                        "total_chunks_in_section": len(section_chunks),
                        "is_structured": True,
                        "document_type": "car_manual",
                        "language": "ko",
                        "chunk_length": len(chunk_text),
                    }

                    chunk = Document(page_content=chunk_text, metadata=chunk_metadata)
                    chunks.append(chunk)
            else:
                # 섹션 전체를 하나의 청크로 처리
                chunk_metadata = {
                    **section.metadata,
                    "section_title": section.title,
                    "section_type": section.section_type,
                    "section_level": section.level,
                    "page_number": section.page_number,
                    "categories": ", ".join(categories),
                    "chunk_index": 0,
                    "total_chunks_in_section": 1,
                    "is_structured": True,
                    "document_type": "car_manual",
                    "language": "ko",
                    "chunk_length": len(section.content),
                }

                chunk = Document(page_content=section.content, metadata=chunk_metadata)
                chunks.append(chunk)

        print(f"✅ {len(chunks)}개 구조화된 청크 생성 완료")
        return chunks

    def create_vector_database(
        self, chunks: List[Document], collection_name: str = "car_manual"
    ) -> Chroma:
        """구조화된 청크를 벡터 데이터베이스에 저장합니다."""
        print("🧠 임베딩 벡터 생성 및 저장 중...")

        # 임베딩 모델 초기화
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # 벡터 저장소 디렉토리 생성
        persist_directory = f"./vector_db/{collection_name}"
        if os.path.exists(persist_directory):
            import shutil

            shutil.rmtree(persist_directory)

        os.makedirs(persist_directory, exist_ok=True)

        # 배치 처리로 벡터 DB 생성
        batch_size = 50
        vector_db = None

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            print(
                f"배치 {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} 처리 중..."
            )

            if vector_db is None:
                # 첫 번째 배치로 초기 DB 생성
                vector_db = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                )
            else:
                # 나머지 배치 추가
                vector_db.add_documents(batch)

        print(f"✅ 벡터 데이터베이스 생성 완료: {persist_directory}")
        return vector_db

    def save_structure_analysis(self, output_path: str = "structure_analysis.json"):
        """구조 분석 결과를 JSON 파일로 저장합니다."""
        analysis_data = {
            "document_info": {
                "file_path": self.pdf_path,
                "total_pages": self.structure_info.get("total_pages", 0),
                "total_sections": self.structure_info.get("total_sections", 0),
            },
            "section_analysis": self.structure_info,
            "sections": [
                {
                    "title": section.title,
                    "type": section.section_type,
                    "level": section.level,
                    "page": section.page_number,
                    "content_length": len(section.content),
                    "content_preview": (
                        section.content[:200] + "..."
                        if len(section.content) > 200
                        else section.content
                    ),
                }
                for section in self.sections
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)

        print(f"📄 구조 분석 결과 저장: {output_path}")

    def process_document(self) -> Tuple[Chroma, Dict[str, Any]]:
        """전체 문서 처리 파이프라인을 실행합니다."""
        print("🚗 자동차 매뉴얼 PDF 구조화 처리 시작...")

        # 1. PDF 로드
        self.load_pdf()

        # 2. 구조 분석
        structure_info = self.analyze_document_structure()

        # 3. 구조화된 청크 생성
        chunks = self.create_enhanced_chunks()

        # 4. 벡터 데이터베이스 생성
        vector_db = self.create_vector_database(chunks)

        # 5. 구조 분석 결과 저장
        self.save_structure_analysis()

        print("🎉 문서 구조화 처리 완료!")
        return vector_db, structure_info


def main():
    """메인 실행 함수"""
    pdf_file = "data/RS4_2025_ko_KR.pdf"

    if not Path(pdf_file).exists():
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_file}")
        return

    # 향상된 PDF 처리기 실행
    processor = EnhancedPDFProcessor(pdf_file)
    vector_db, structure_info = processor.process_document()

    # 검색 테스트
    print("\n🔍 구조화된 검색 테스트...")
    test_queries = [
        "엔진 시동 방법",
        "안전벨트 사용법",
        "에어컨 조작",
        "주의사항",
        "표 정보",
    ]

    for query in test_queries:
        print(f"\n검색어: '{query}'")
        results = vector_db.similarity_search(query, k=2)
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.metadata.get('section_title', 'N/A')}")
            print(f"     타입: {result.metadata.get('section_type', 'N/A')}")
            print(f"     페이지: {result.metadata.get('page_number', 'N/A')}")
            print(f"     내용: {result.page_content[:100]}...")


if __name__ == "__main__":
    main()
