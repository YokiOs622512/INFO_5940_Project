import streamlit as st
import os
import tempfile
import uuid
import hashlib
import json
import time
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

# API 配置
api_key = os.environ.get("API_KEY") 
base_url = "https://api.ai.it.cornell.edu"

if not api_key:
    st.error("⚠️ API Key not found. Please set your API_KEY in the environment variables.")
    st.stop()

# 初始化 LLM
llm = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.7,
    openai_api_key=api_key,
    openai_api_base=base_url
)

# 初始化 Embedding
embeddings = OpenAIEmbeddings(
    model="openai.text-embedding-3-small",
    openai_api_key=api_key,
    openai_api_base=base_url
)

# ==========================================
# 2. 游戏数据与状态管理 (Game State)
# ==========================================

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Greetings, Tarnished. Upload the World Codex (JSON) to begin thy journey."}]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_file_hashes" not in st.session_state:
    st.session_state.processed_file_hashes = set()

if "current_persona" not in st.session_state:
    st.session_state.current_persona = "Ranni the Witch"

# --- 新增：游戏引擎状态 ---
if "game_data" not in st.session_state:
    st.session_state.game_data = None # 存储完整的 JSON 数据字典
if "current_location_id" not in st.session_state:
    st.session_state.current_location_id = None # 玩家当前位置 ID

# ==========================================
# 3. 角色设定 (Persona System)
# ==========================================
NPC_PERSONAS = {
    "Ranni the Witch": {
        "description": "The enigmatic lunar princess.",
        "prompt": """
        You are Ranni the Witch. 
        Tone: Cold, mysterious, archaic (thee, thou).
        Role: You are the Dungeon Master and Guide. You must guide the player through the world descriptions provided in the context.
        Constraint: Describe the current location vividly based on the Game State provided. Do NOT break character.
        """
    },
    "Melina": {
        "description": "Your guide and the Kindling Maiden.",
        "prompt": """
        You are Melina.
        Tone: Soft-spoken, dutiful, supportive.
        Role: You guide the Tarnished to the Erdtree.
        Constraint: Offer advice on where to go next based on the available exits.
        """
    },
    "Iron Fist Alexander": {
        "description": "The jovial Warrior Jar.",
        "prompt": """
        You are Alexander, the Iron Fist!
        Tone: Boisterous, hearty, loud!
        Role: You want to find strong opponents and glory!
        Constraint: Describe battles and locations with excitement!
        """
    }
}

# ==========================================
# 4. 后端逻辑 (Backend Logic)
# ==========================================

def create_file_hash(uploaded_file):
    content_preview = uploaded_file.getvalue()[:100] if hasattr(uploaded_file, 'getvalue') else b''
    hash_input = f"{uploaded_file.name}_{uploaded_file.size}_{content_preview}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def process_documents(uploaded_files):
    """读取文件：既存入 RAG 用于检索，也解析 JSON 用于游戏逻辑"""
    if not uploaded_files:
        return False
    
    all_documents = []
    
    for uploaded_file in uploaded_files:
        file_hash = create_file_hash(uploaded_file)
        if file_hash in st.session_state.processed_file_hashes:
            continue
            
        with st.spinner(f"Processing {uploaded_file.name}..."):
            text_content = ""
            
            # --- JSON 处理 (关键更新) ---
            if uploaded_file.type == "application/json":
                try:
                    # 1. 解析为 Python 字典，存入 Game State 用于逻辑控制
                    data = json.load(uploaded_file)
                    st.session_state.game_data = data
                    
                    # 尝试设置初始位置 (默认取 locations 的第一个 key)
                    if not st.session_state.current_location_id and "locations" in data:
                        first_loc = list(data["locations"].keys())[0]
                        st.session_state.current_location_id = first_loc
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"The world has been reconstructed. We begin at the **{data['locations'][first_loc]['name']}**."
                        })

                    # 2. 转换为文本，存入 RAG 用于 Lore 检索
                    text_content = json.dumps(data, indent=2, ensure_ascii=False)
                except Exception as e:
                    st.error(f"Error parsing JSON: {e}")
                    continue

            # --- PDF/TXT 处理 ---
            elif uploaded_file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                try:
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    text_content = "\n".join([doc.page_content for doc in docs])
                finally:
                    os.unlink(tmp_file_path)
            elif uploaded_file.type == "text/plain":
                text_content = uploaded_file.read().decode("utf-8")
                
            if text_content:
                all_documents.append(Document(page_content=text_content, metadata={"source": uploaded_file.name}))
                st.session_state.processed_file_hashes.add(file_hash)
    
    if all_documents:
        # 存入 ChromaDB
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(all_documents)
        
        if st.session_state.vector_store is None:
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=f"elden_ring_{uuid.uuid4().hex[:8]}"
            )
        else:
            st.session_state.vector_store.add_documents(chunks)
        return True
    return False

def get_current_game_context():
    """获取当前游戏状态的文本描述，用于注入 Prompt"""
    if not st.session_state.game_data or not st.session_state.current_location_id:
        return "Game State: No world data loaded. Just chat normally."
    
    loc_id = st.session_state.current_location_id
    loc_data = st.session_state.game_data["locations"].get(loc_id, {})
    
    context = f"""
    --- CURRENT GAME STATE ---
    Current Location ID: {loc_id}
    Location Name: {loc_data.get('name', 'Unknown')}
    Description: {loc_data.get('description', '')}
    Available Exits/Choices: {list(loc_data.get('exits', []))}
    Boss Here: {loc_data.get('boss', 'None')}
    --------------------------
    """
    return context

def generate_npc_response(user_input, persona_name):
    """生成回复：结合 RAG + 游戏状态"""
    
    # 1. 准备上下文
    game_context = get_current_game_context()
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
    
    # 2. RAG 检索 (Lore)
    rag_context = ""
    source_docs = []
    if st.session_state.vector_store:
        # 结合用户问题和当前地点进行检索
        query = f"{user_input} {st.session_state.current_location_id}"
        source_docs = st.session_state.vector_store.similarity_search(query, k=3)
        rag_context = "\n".join([d.page_content for d in source_docs])

    # 3. 构建 Prompt
    persona_prompt = NPC_PERSONAS[persona_name]["prompt"]
    
    full_prompt = f"""
    {persona_prompt}
    
    MISSION: 
    You are guiding the player through a text adventure game.
    Use the [CURRENT GAME STATE] to describe where the player is and what they see.
    Use the [LORE KNOWLEDGE] to add depth and history to the location.
    
    [CURRENT GAME STATE]:
    {game_context}
    
    [LORE KNOWLEDGE]:
    {rag_context}
    
    [CHAT HISTORY]:
    {chat_history}
    
    Player Input: {user_input}
    
    Response (Stay in character, describe the scene, list options if asked):
    """
    
    response = llm.invoke(full_prompt)
    return response.content, source_docs

def handle_movement(target_location_id):
    """处理玩家点击按钮后的移动逻辑"""
    st.session_state.current_location_id = target_location_id
    
    # 获取新地点信息
    loc_name = st.session_state.game_data["locations"][target_location_id]["name"]
    
    # 在聊天记录中模拟一条“系统消息”
    st.session_state.messages.append({"role": "user", "content": f"(Travels to {loc_name})"})
    
    # 强制让 NPC 立即根据新地点生成一段描述
    with st.spinner("Travelling..."):
        response, _ = generate_npc_response(f"I have arrived at {loc_name}. What do I see?", st.session_state.current_persona)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

# ==========================================
# 5. 前端界面 (Streamlit UI)
# ==========================================

# --- 侧边栏 ---
with st.sidebar:
    st.title("⚙️ Game Controls")
    
    selected_persona = st.selectbox("Choose Guide", list(NPC_PERSONAS.keys()))
    if selected_persona != st.session_state.current_persona:
        st.session_state.current_persona = selected_persona
    
    st.subheader("📁 Upload World (JSON Required)")
    uploaded_files = st.file_uploader(
        "Upload game_world.json & Lore", 
        type=["json", "pdf", "txt"],
        accept_multiple_files=True
    )
    if uploaded_files:
        if process_documents(uploaded_files):
            st.success("World Loaded!")
    
    # 显示当前状态 (调试用)
    if st.session_state.current_location_id:
        st.info(f"📍 Location: {st.session_state.current_location_id}")

    if st.button("Restart Game"):
        st.session_state.messages = []
        st.session_state.current_location_id = list(st.session_state.game_data["locations"].keys())[0] if st.session_state.game_data else None
        st.rerun()

# --- 主界面 ---
st.title("Elden Ring: The Shattered Conversation")
st.caption(f"Guide: **{st.session_state.current_persona}** | Mode: **Interactive RPG**")

# 1. 显示聊天历史
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🧙‍♀️"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# 2. 游戏交互区域 (Action Bar) - 只有在加载了 JSON 后才显示
if st.session_state.game_data and st.session_state.current_location_id:
    
    current_loc = st.session_state.game_data["locations"].get(st.session_state.current_location_id)
    
    if current_loc:
        st.write("---")
        st.subheader("⚔️ Actions & Travel")
        
        # 获取当前地点的出口 (Exits)
        exits = current_loc.get("exits", [])
        events = current_loc.get("events", [])
        
        # 动态生成按钮
        # 使用 Streamlit 的列布局来放置按钮
        cols = st.columns(len(exits) + 1 if exits else 1)
        
        # 遍历生成“移动”按钮
        for idx, exit_id in enumerate(exits):
            # 获取目标地点的名字（如果存在）
            dest_name = st.session_state.game_data["locations"].get(exit_id, {}).get("name", exit_id)
            if cols[idx].button(f"👣 Go to {dest_name}", key=f"btn_{exit_id}"):
                handle_movement(exit_id)
        
        # 检查是否到了结局
        if "ending" in current_loc.get("events", [{}])[0]: # 简化的结局检测
            st.warning("✨ An Ending is upon you. Speak to make your choice.")

# 3. 聊天输入框
if prompt := st.chat_input("Speak to your guide..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)
    
    with st.chat_message("assistant", avatar="🧙‍♀️"):
        with st.spinner("Thinking..."):
            response_text, source_docs = generate_npc_response(prompt, st.session_state.current_persona)
            st.markdown(response_text)
            
            if source_docs:
                with st.expander("🔮 See Reasoning & Game Data"):
                    st.json(get_current_game_context()) # 展示当前游戏状态作为证据
                    st.write("Retrieval Context:", [d.page_content[:200] for d in source_docs])
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
