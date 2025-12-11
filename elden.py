import streamlit as st
import os
import tempfile
import uuid
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# ==========================================
# 1. 配置与初始化 (Configuration)
# ==========================================

st.set_page_config(
    page_title="Elden Ring: The Shattered Conversation", 
    page_icon="💍", 
    layout="wide"
)

# 使用 Cornell API 配置
# 尝试从环境变量获取 Key，如果本地测试没有设置，请手动填入
api_key = os.environ.get("API_KEY") 
base_url = "https://api.ai.it.cornell.edu"

if not api_key:
    st.error("⚠️ API Key not found. Please set your API_KEY in the environment variables.")
    st.stop()

# 初始化 LLM (用于生成对话) - 使用 GPT-4o 获取更好的角色扮演效果
llm = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.7,  # 稍微调高温度，让 NPC 对话更生动
    openai_api_key=api_key,
    openai_api_base=base_url
)

# 初始化 Embedding (用于检索)
embeddings = OpenAIEmbeddings(
    model="openai.text-embedding-3-small",
    openai_api_key=api_key,
    openai_api_base=base_url
)

# ==========================================
# 2. 角色设定 (Persona System) - Innovation
# ==========================================
# 这里定义了不同 NPC 的“灵魂”，通过 System Prompt 注入
NPC_PERSONAS = {
    "Ranni the Witch": {
        "description": "The enigmatic lunar princess.",
        "prompt": """
        You are Ranni the Witch from Elden Ring.
        Tone: Cold, mysterious, archaic, and regal. You often use old English (thee, thou, thy).
        Personality: You seek to overthrow the Golden Order and usher in the Age of Stars. You are a demigod but discarded your Empyrean flesh.
        Constraint: Do NOT act like a robotic assistant. Act entirely as Ranni. If the retrieved context is missing, be vague and mysterious about the stars.
        """
    },
    "Melina": {
        "description": "Your guide and the Kindling Maiden.",
        "prompt": """
        You are Melina, a guide to the Tarnished.
        Tone: Soft-spoken, dutiful, slightly melancholic but supportive.
        Personality: You wish to guide the player to the Erdtree to fulfill your purpose. You often quote Queen Marika's echoes.
        Constraint: Refer to the user as "Tarnished". Keep answers concise but poetic.
        """
    },
    "Iron Fist Alexander": {
        "description": "The jovial Warrior Jar.",
        "prompt": """
        You are Alexander, the Iron Fist! A sentient Warrior Jar.
        Tone: Boisterous, hearty, optimistic, and loud!
        Personality: You seek to become a great champion by stuffing your insides with the remains of warriors.
        Constraint: Use exclamations! Refer to the user as "my friend" or "brave warrior". Talk about getting stuck in holes if relevant.
        """
    }
}

# ==========================================
# 3. 状态管理 (Session State)
# ==========================================

if "messages" not in st.session_state:
    # 默认第一条消息
    st.session_state.messages = [{"role": "assistant", "content": "Greetings, Tarnished. Which lore fragments shall we explore today?"}]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_file_hashes" not in st.session_state:
    st.session_state.processed_file_hashes = set()

if "current_persona" not in st.session_state:
    st.session_state.current_persona = "Ranni the Witch"

# ==========================================
# 4. 后端逻辑函数 (Backend Logic)
# ==========================================

def create_file_hash(uploaded_file):
    """创建文件哈希以避免重复处理 (来自 chat_with_pdf.py)"""
    content_preview = uploaded_file.getvalue()[:100] if hasattr(uploaded_file, 'getvalue') else b''
    hash_input = f"{uploaded_file.name}_{uploaded_file.size}_{content_preview}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def process_documents(uploaded_files):
    """读取上传的 Lore 文件 (PDF/TXT) 并存入 ChromaDB"""
    if not uploaded_files:
        return False
    
    all_documents = []
    new_files_count = 0
    
    for uploaded_file in uploaded_files:
        file_hash = create_file_hash(uploaded_file)
        
        if file_hash in st.session_state.processed_file_hashes:
            continue
            
        with st.spinner(f"Communing with the Greater Will (Processing {uploaded_file.name})..."):
            text_content = ""
            
            # 处理 PDF
            if uploaded_file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                try:
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    text_content = "\n".join([doc.page_content for doc in docs])
                finally:
                    os.unlink(tmp_file_path)
            # 处理 TXT
            elif uploaded_file.type == "text/plain":
                text_content = uploaded_file.read().decode("utf-8")
                
            if text_content:
                # 添加元数据
                all_documents.append(Document(
                    page_content=text_content, 
                    metadata={"source": uploaded_file.name}
                ))
                st.session_state.processed_file_hashes.add(file_hash)
                new_files_count += 1
    
    if all_documents:
        # 分块 (Chunking) - 设置较小的 chunk 以获取精确的 Lore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = text_splitter.split_documents(all_documents)
        
        # 存入 ChromaDB
        if st.session_state.vector_store is None:
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=f"elden_ring_lore_{uuid.uuid4().hex[:8]}"
            )
        else:
            st.session_state.vector_store.add_documents(chunks)
            
        return True
    return False

def get_chat_history_str(max_history=4):
    """格式化最近的对话历史，让 AI 拥有记忆"""
    history = st.session_state.messages[-max_history:]
    history_str = ""
    for msg in history:
        role = "Player" if msg["role"] == "user" else "NPC"
        history_str += f"{role}: {msg['content']}\n"
    return history_str

def generate_npc_response(question, persona_name):
    """核心 RAG 逻辑：检索 + 角色扮演生成"""
    
    # 1. 检查是否有知识库
    if st.session_state.vector_store is None:
        return "Tarnished, you possess no memory fragments (Please upload Lore documents first).", []
    
    # 2. 检索 (Retrieval)
    # 检索 Top 4 相关片段
    docs = st.session_state.vector_store.similarity_search(question, k=4)
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # 3. 构建 Prompt
    persona_prompt = NPC_PERSONAS[persona_name]["prompt"]
    chat_history = get_chat_history_str()
    
    full_template = f"""
    {persona_prompt}
    
    You are engaging in a conversation with a player.
    Use the following "Lore Knowledge" to answer the player's question.
    If the answer isn't in the Lore, stay in character and improvise vaguely.
    
    ---
    LORE KNOWLEDGE:
    {context_text}
    ---
    
    CONVERSATION HISTORY:
    {chat_history}
    
    Player: {question}
    Response:
    """
    
    # 4. 生成回答
    response = llm.invoke(full_template)
    return response.content, docs

# ==========================================
# 5. 前端界面 (Streamlit UI)
# ==========================================

# --- 侧边栏: 设置与资源 ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/b/b9/Elden_Ring_Box_Art.jpg/220px-Elden_Ring_Box_Art.jpg", caption="Elden Ring Lore Companion", width=150)
    st.header("⚙️ Game Setup")
    
    # 1. 角色选择
    st.subheader("Select NPC")
    selected_persona = st.selectbox(
        "Who do you want to talk to?",
        options=list(NPC_PERSONAS.keys()),
        index=0
    )
    # 如果切换了角色，记录状态
    if selected_persona != st.session_state.current_persona:
        st.session_state.current_persona = selected_persona
        # 可选：切换角色时是否清空历史？为了演示Memory功能，建议保留，或者加个Toast提示
        st.toast(f"Summoned {selected_persona}!")
    
    st.info(f"**Current Persona:** {NPC_PERSONAS[selected_persona]['description']}")
    
    st.divider()
    
    # 2. 文件上传
    st.subheader("📜 Lore Fragments (Knowledge Base)")
    uploaded_files = st.file_uploader(
        "Upload PDF/TXT Lore", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if process_documents(uploaded_files):
            st.success("✅ Fragments processed into Memory!")
            
    # 显示状态
    if st.session_state.vector_store:
        doc_count = st.session_state.vector_store._collection.count()
        st.markdown(f"*Current Memory Fragments: {doc_count} chunks*")
    
    st.divider()
    if st.button("🗑️ Reset World (Clear All)"):
        st.session_state.clear()
        st.rerun()

# --- 主界面: 聊天窗口 ---
st.title("Elden Ring: The Shattered Conversation")
st.caption(f"Talking to: **{st.session_state.current_persona}** | Powered by RAG & OpenAI")

# 显示历史消息
for msg in st.session_state.messages:
    # 自定义头像
    avatar = "👤" if msg["role"] == "user" else "🧙‍♀️"
    if msg["role"] == "assistant":
        if "Alexander" in st.session_state.current_persona: avatar = "🏺"
        elif "Melina" in st.session_state.current_persona: avatar = "🔥"
    
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# 处理用户输入
if prompt := st.chat_input("Speak thy mind, Tarnished..."):
    # 1. 记录并显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)
    
    # 2. 生成回复
    with st.chat_message("assistant", avatar="🧙‍♀️"):
        with st.spinner(f"{st.session_state.current_persona} is pondering the stars..."):
            response_text, source_docs = generate_npc_response(prompt, st.session_state.current_persona)
            
            st.markdown(response_text)
            
            # 3. 复杂性展示 (Complexity): 显示思维链/来源
            if source_docs:
                with st.expander("🔮 See NPC's Thoughts & Lore Source"):
                    st.markdown("**Reasoning:** The NPC retrieved these fragments to answer you:")
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Fragment {i+1} (from {doc.metadata.get('source')}):**")
                        st.caption(doc.page_content[:300] + "...") # 只显示前300字
    
    # 4. 记录 AI 回复到历史
    st.session_state.messages.append({"role": "assistant", "content": response_text})
