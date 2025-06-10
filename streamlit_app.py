import streamlit as st
import base64
import pickle
import os
#from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory


# ✅ 페이지 설정
st.set_page_config(page_title="민법 상담 챗봇", layout="centered")

st.markdown("""
<style>
body, p, div, span, h1, h2, h3, h4, h5, h6, label, textarea {
    font-family: 'Nanum Myeongjo', 'Gowun Batang', serif !important;
    color: #FFFFFF !important;
}


div[data-baseweb="select"] > div {
    background-color: #626F47 !important;
    color: white !important;
    border-radius: 20px !important;
    border: none !important;           
    box-shadow: none !important;       
}

li[role="option"] {
    background-color: #27391C !important;
    color: black !important;
}

/* 사이드바 */
section[data-testid="stSidebar"] > div:first-child {
    background-color: #18230F !important;
}


/* 멀티라인 입력 */
textarea {
    background-color: #555555 !important;
    color: white !important;
}

/* 사용자 말풍선 */
.chat-bubble-human {
    background-color: #789DBC;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 1rem;
    color: #F0F0F0;
    max-width: 65%;
    margin-left: auto;
}

/* AI 말풍선 */
.chat-bubble-ai {
    background-color: #254D70;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 1rem;
    color: #111827;
    max-width: 65%;
}
</style>
""", unsafe_allow_html=True)


# ✅ 제목
st.title("민법 상담 챗봇")

# ✅ 환경변수 로드
#load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

import requests

def download_file_from_drive(file_id: str, save_path: str):
    """Google Drive에서 파일을 다운로드합니다."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(response.content)
    st.success(f"✅ {os.path.basename(save_path)} 다운로드 완료!")

# ✅ 다운로드 대상 목록 (예시)
files_to_download = {
    "pdf_documents.pkl": "15o-tCm2g-CyFN-VRNpUFc57hfOsvAHYg",
    "pdf_embeddings.pkl": "1JNHnhQdydc74Nb0N1iyWSkzbWiF2U8mb",
    "고소장_documents.pkl": "1FAGxnRHdnp1byu83HIIrOshIa_yhm24m",
    "고소장_embeddings.pkl": "1A8RURLc1MNa4wQbkQGy_fysfR8tOX7ja",
    "계약_documents.pkl" : "1jSOlWeigzOFhBdTf3bKHbmAA9Vf1XzMi",
    "계약_embeddings.pkl" : "1wXRpI9nsa9zzb60wZ8xqrMeQ-MKYq5ao",
    "교통(자동차)_documents.pkl" : "13LiyWLI_lYYc0Fd6R0bVzww7i7ETfJ3d",
    "교통(자동차)_embeddings.pkl" : "1OBqMxzVIYSgj1hXEwPPLcEamj3ybykUn",
    "기타_documents.pkl" : "17GDNfaCYei9218LM2hS7OzaEaaL7O-_D",
    "기타_embeddings.pkl" : "1GTKpR8ZvEa-dePD_rtnzVxJxvIAMnOfl",
    "노동_documents.pkl" : "1AhhiJQSWuFQB6EUYPsYjvklt4FuNTvyx",
    "노동_embeddings.pkl" : "1wQHEUqR0gzTk6cel-4qu2B8BPCZgQ76o",
    "대금_documents.pkl" : "1s3sntC64PPr6enojAWgNSCdi6WEjJ7MM",
    "대금_embeddings.pkl" : "1xg-muxmIaq668YHzy6zmojhmKFiK5JOX",
    "부동산_documents.pkl" : "16Ms2bgGhmUnCTHymCqUL0aLh8I1g5igX",
    "부동산_embeddings.pkl" : "1OxWUcwj3qDXmJbp3p626P82i1_4uehD3",
    "사기 및 형사_documents.pkl" : "1YAbXEGuaXjtn1fvHwOM7BKhU3QN8gO5M",
    "사기 및 형사_embeddings.pkl" : "1LCc4OoTBBSxTcA7tUFG34V6LL47ltgRT",
    "상속_documents.pkl" : "1c1kEpXG5u-5uA217ELvgFUfOOpmgp6CE",
    "상속_embeddings.pkl" : "1cYYxp8UiaMBcV0DFbr3dnNKGeJuJRjZx",
    "손해배상_documents.pkl" : "1k_Fd6Hoag1RDRvp5Vc_yOXPWq3gzDXip",
    "손해배상_embeddings.pkl" : "1v4B65q6gdyk-PVNlDoIRdjIf0ipFySH4",
    "합의서_documents.pkl" : "12MpaJYWMx1rRm5f5l3H0uZapY2HM8ihz",
    "합의서_embeddings.pkl" : "1yOV9v-uy31t4r10Qbv0dXLZC9stdpIcU"
}

# ✅ 파일 존재 여부 확인 후 다운로드
for filename, file_id in files_to_download.items():
    local_path = os.path.join("precomputed", filename)
    if not os.path.exists(local_path):
        st.info(f"📥 {filename} 다운로드 중...")
        download_file_from_drive(file_id, local_path)



BASE_DIR="precomputed"





# ✅ 배경 이미지 삽입 함수
def add_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        font-family: 'Nanum Myeongjo', 'Gowun Batang', serif;
        color: #000000 !important;
    }}
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
 </style>
    """, unsafe_allow_html=True)

def render_example_block(text: str):
    styled_html = f"""
    <div style="
        background-color: #254D70;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 16px;
        font-family: 'Nanum Myeongjo', 'Gowun Batang', serif;
        white-space: pre-wrap;
        ">
        {text}
    </div>
    """
    st.markdown(styled_html, unsafe_allow_html=True)

        
def contains_legal_example(text: str):
    example_keywords = ["고소장", "합의서", "계약서", "예시:", "고   소   장", "합   의   서"]
    return any(keyword in text for keyword in example_keywords)


# ✅ 배경 이미지 경로 설정
add_bg_from_local("background.jpg")

# ✅ 사이드바 메뉴
st.sidebar.title("🔧 메뉴")
menu_option = st.sidebar.radio("페이지 선택", ["🤖 챗봇", "📘 프로젝트 소개"])

# ✅ 설정


CATEGORIES = [
    "상담 분류를 지정해주세요", "고소장", "합의서", 
    "교통(자동차)", "사기 및 형사", "부동산", "노동", "대금", "손해배상", "상속", "계약", "기타"
]



def find_pkl_file(category_name, file_type):
    filename = f"{category_name}_{file_type}.pkl"
    full_path = os.path.join(BASE_DIR, filename)
    return full_path if os.path.exists(full_path) else None



def find_pdf_pkl(file_type):
    filename = f"pdf_{file_type}.pkl"
    full_path = os.path.join(BASE_DIR, filename)
    return full_path if os.path.exists(full_path) else None

@st.cache_resource
def load_embeddings(category):
    pdf_docs, pdf_vecs = [], []
    pdf_doc_path = find_pdf_pkl("documents")
    pdf_vec_path = find_pdf_pkl("embeddings")
    if pdf_doc_path and pdf_vec_path:
        with open(pdf_doc_path, "rb") as f:
            pdf_docs = pickle.load(f)
        with open(pdf_vec_path, "rb") as f:
            pdf_vecs = pickle.load(f)
            
    cat_docs, cat_vecs = [], []
    if category and category != "상담 분류를 지정해주세요":
        cat_doc_path = find_pkl_file(category, "documents")
        cat_vec_path = find_pkl_file(category, "embeddings")
        if cat_doc_path and cat_vec_path:
            with open(cat_doc_path, "rb") as f:
                cat_docs = pickle.load(f)
            with open(cat_vec_path, "rb") as f:
                cat_vecs = pickle.load(f)
                
    # 디버깅용 출력
    st.write(f"PDF 문서/임베딩: {len(pdf_docs)}/{len(pdf_vecs)}")
    st.write(f"카테고리 문서/임베딩: {len(cat_docs)}/{len(cat_vecs)}")

    # 데이터 없으면 안내
    if (len(pdf_docs) == 0 or len(pdf_vecs) == 0) and (len(cat_docs) == 0 or len(cat_vecs) == 0):
        st.warning("pkl 파일이 없거나, 데이터가 비어 있습니다.")
    return pdf_docs, pdf_vecs, cat_docs, cat_vecs

from langchain_core.documents import Document

def build_vectorstore(_pdf_docs, _pdf_vecs, _cat_docs, _cat_vecs, category):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    cat_docs_wrapped = [
        Document(page_content=doc) if isinstance(doc, str) else doc
        for doc in _cat_docs
    ]
    cat_pairs = [(doc.page_content, vec) for doc, vec in zip(cat_docs_wrapped, _cat_vecs)]

    pdf_docs_wrapped = [
        Document(page_content=doc) if isinstance(doc, str) else doc
        for doc in _pdf_docs
    ]
    pdf_pairs = [(doc.page_content, vec) for doc, vec in zip(pdf_docs_wrapped, _pdf_vecs)]

    st.write(f"📄 벡터스토어 구성 중 - PDF {len(pdf_pairs)}, CAT {len(cat_pairs)}")

    vectorstore = FAISS.from_embeddings(pdf_pairs + cat_pairs, embeddings)
    return vectorstore




@st.cache_resource
def initialize_rag_chain(model_name, _vectorstore):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 10})
    qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """당신은 대한민국 민법과 형사법, 기타 주요 법률에 정통한 AI 법률 상담 챗봇입니다.  
     다음의 지침에 따라 공손하고 신뢰감 있는 한국어로 답변하세요.
     
     📘 [법률 개념 설명]
     - 사용자가 질문한 법률 개념이나 조항에 대해 **정확하고 자세하게** 설명합니다.
     - 반드시 관련 법률 조문 번호를 함께 제공합니다.  
       예: "민법 제750조 (불법행위)", "형법 제347조 (사기죄)", "민사소송법 제217조"

    📄 [고소장 및 합의서 초안 작성]
    - 사용자가 고소장 또는 합의서의 양식을 요청할 때, 구체적인 예시와 함께 자세하게 어떤 정보가 필요한지 설명합니다.
    - 사용자의 입력을 바탕으로 **문서 초안을 우선 생성**하세요.
    - 필요한 정보가 부족한 경우, 먼저 가능한 범위만 작성한 뒤 **누락된 항목을 정중히 요청**하세요.  
      예: "피고인의 이름이 필요합니다", "사건 발생 일자를 알려주세요"
    - 고소장과 합의서에 들어갈 기본 항목들을 함께 안내합니다.  
      예: 사건 개요, 당사자 정보, 청구 내용 등

    📌 [판결 예측 원칙]
    - 사용자의 질문이 **구체적인 상황**(ex. 신호위반, 피해자 경상 등)을 포함할 경우:
      → **가정 없이**, 해당 내용을 기반으로 관련 법, 일반적 판결, 예상 처벌 수위(벌금, 면허정지 등)를 반드시 제시합니다.
      → 예: "벌금 300~500만 원", "면허정지 30일", "위자료 100~200만 원"

    - 질문이 **포괄적이거나 부족할 경우**:
      → 현실적인 시나리오를 직접 설정하고 가정을 명확히 제시한 후 예측합니다.
      → 예: "가정: 가해자는 신호위반 초범이며, 피해자는 치료 2주 경상의 부상을 입었습니다..."
      → 예: "가정: 임대인이 계약기간 종료 전에 일방적으로 계약을 해지한 경우..."

    - 어떤 경우에도 **'정확히 알 수 없습니다' 또는 '더 구체적 정보가 필요합니다'** 라는 답은 하지 마세요.
      → 대신, 일반적인 사례 기준에서 설명하고 예측을 제시하세요.
    - 답변에는 **정량적 정보**(벌금액, 위자료, 배상비, 형량 등)를 반드시 포함하세요.


    📚 [문서 기반 답변 (RAG)]
    - context로 제공된 문서(판결문, 사례, 조문 등)를 최우선 근거로 사용하세요.
    - 문서가 없거나 부족한 경우에도 일반적인 판례 경향을 근거로 신중히 설명합니다.
    - 확인되지 않은 정보를 단정적으로 말하지 마세요. 단, 일반론적 판단은 회피하지 말고 책임감 있게 표현하세요.

    ⚠️ [기타 주의사항]
    - 모든 설명은 **공손하고 명확한 문체**로 전달하세요.
    - 혼란을 줄 수 있는 불확실한 추측은 피하고, 필요한 경우 "해당 정보는 문서에 없습니다"라고 안내하세요.
{context}
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

    llm = ChatOpenAI(model=model_name)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

# ✅ 챗봇 페이지
if menu_option == "🤖 챗봇":
    st.title("👩‍⚖️ 민법 상담 챗봇")
    st.markdown("##### 정중하고 신뢰감 있는 법률 상담을 제공합니다.")

    model_choice = st.selectbox("GPT 모델을 선택해주세요", ("gpt-4o", "gpt-3.5-turbo-0125"))
    category = st.selectbox("상담 분류를 선택하세요", CATEGORIES, index=0)

    if category == "상담 분류를 지정해주세요":
        st.info("상담 분류를 먼저 선택해 주세요.")
        st.stop()

    pdf_docs, pdf_vecs, cat_docs, cat_vecs = load_embeddings(category)
    if (not pdf_docs or not pdf_vecs) and (not cat_docs or not cat_vecs):
        st.warning("챗봇에 사용할 데이터가 없습니다. pkl 파일을 확인하세요.")
        st.stop()

    vectorstore = build_vectorstore(pdf_docs, pdf_vecs, cat_docs, cat_vecs, category)

    rag_chain = initialize_rag_chain(model_name=model_choice, _vectorstore=vectorstore)

    chat_history = StreamlitChatMessageHistory(key="chat_messages")
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    if not chat_history.messages:
        chat_history.add_ai_message("안녕하세요! 민법에 대해 궁금한 점을 물어보세요.")

    for msg in chat_history.messages:
        if msg.type == "ai":
            st.markdown(f"<div class='chat-bubble-ai'>{msg.content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-human'>{msg.content}</div>", unsafe_allow_html=True)

    if prompt := st.chat_input("질문을 입력해주세요 :)"):
        st.markdown(f"<div class='chat-bubble-human'>{prompt}</div>", unsafe_allow_html=True)
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke({"input": prompt}, config)
            answer = response.get("answer", "답변을 생성하지 못했습니다.")
            if contains_legal_example(answer):
                render_example_block(answer)  # 예시니까 검정 배경 출력
            else:
                st.markdown(f"<div class='chat-bubble-ai'>{answer}</div>", unsafe_allow_html=True)


# ✅ 소개 페이지
elif menu_option == "📘 프로젝트 소개":
    st.title("📘 프로젝트 소개")
    st.markdown("""
    ### 👩‍⚖️ 민법 상담 챗봇 프로젝트
    이 프로젝트는 사용자가 민법 관련 문서를 보다 쉽게 작성하고,
    자주 묻는 법률 질문에 대해 도움을 받을 수 있도록 설계되었습니다.

    ---
    #### 📌 주요 기능
    - 자연어 기반 민법 상담 제공
    - 고소장, 합의서, 계약 해제 조건 등에 대한 자동 응답
    - Streamlit UI와 사용자 친화적 인터페이스

    #### 🛠 기술 스택
    - Python, Streamlit
    - LangChain + FAISS
    - OpenAI API


    민법에 대한 접근성과 정보 전달을 개선하기 위해 지속적으로 개선해 나가겠습니다.
    """)
