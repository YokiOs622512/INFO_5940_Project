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
# 1. 配置与初始化
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

llm = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.7,
    openai_api_key=api_key,
    openai_api_base=base_url
)

embeddings = OpenAIEmbeddings(
    model="openai.text-embedding-3-small",
    openai_api_key=api_key,
    openai_api_base=base_url
)

# ==========================================
# 2. 状态管理
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Greetings, Tarnished. Upload the World Codex (JSON) to begin thy journey."}]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_file_hashes" not in st.session_state:
    st.session_state.processed_file_hashes = set()

if "current_persona" not in st.session_state:
    st.session_state.current_persona = "Ranni the Witch"

# 游戏状态
if "game_data" not in st.session_state:
    st.session_state.game_data = None 
if "current_location_id" not in st.session_state:
    st.session_state.current_location_id = None 

# ==========================================
# 3. 角色设定
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
# 4. 后端逻辑
# ==========================================

def create_file_hash(uploaded_file):
    content_preview = uploaded_file.getvalue()[:100] if hasattr(uploaded_file, 'getvalue') else b''
    hash_input = f"{uploaded_file.name}_{uploaded_file.size}_{content_preview}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def process_documents(uploaded_files):
    if not uploaded_files:
        return False
    
    all_documents = []
    
    for uploaded_file in uploaded_files:
        file_hash = create_file_hash(uploaded_file)
        if file_hash in st.session_state.processed_file_hashes:
            continue
            
        with st.spinner(f"Processing {uploaded_file.name}..."):
            text_content = ""
            
            # JSON 处理
            if uploaded_file.type == "application/json":
                try:
                    data = json.load(uploaded_file)
                    st.session_state.game_data = data
                    
                    # 初始化位置
                    if not st.session_state.current_location_id and "locations" in data:
                        first_loc = list(data["locations"].keys())[0]
                        st.session_state.current_location_id = first_loc
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"The world has been reconstructed. We begin at the **{data['locations'][first_loc]['name']}**."
                        })

                    text_content = json.dumps(data, indent=2, ensure_ascii=False)
                except Exception as e:
                    st.error(f"Error parsing JSON: {e}")
                    continue

            # PDF/TXT 处理
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
    """获取当前游戏状态的文本描述"""
    if not st.session_state.game_data or not st.session_state.current_location_id:
        return "Game State: No world data loaded. Just chat normally."
    
    loc_id = st.session_state.current_location_id
    loc_data = st.session_state.game_data["locations"].get(loc_id, {})
    
    # 返回纯文本字符串
    context = f"""
    --- CURRENT GAME STATE ---
    Current Location ID: {loc_id}
    Location Name: {loc_data.get('name', 'Unknown')}
    Description: {loc_data.get('description', '')}
    Available Exits: {list(loc_data.get('exits', []))}
    Boss Here: {loc_data.get('boss', 'None')}
    --------------------------
    """
    return context

def generate_npc_response(user_input, persona_name):
    # 准备上下文
    game_context = get_current_game_context()
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
    
    # RAG 检索
    rag_context = ""
    source_docs = []
    if st.session_state.vector_store:
        query = f"{user_input} {st.session_state.current_location_id}"
        source_docs = st.session_state.vector_store.similarity_search(query, k=3)
        rag_context = "\n".join([d.page_content for d in source_docs])

    # 构建 Prompt
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
    st.session_state.current_location_id = target_location_id
    loc_name = st.session_state.game_data["locations"][target_location_id]["name"]
    
    with st.spinner("Travelling..."):
        # 模拟移动后 NPC 自动描述
        response, _ = generate_npc_response(f"I have arrived at {loc_name}. Describe my surroundings.", st.session_state.current_persona)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

# ==========================================
# 5. 前端界面
# ==========================================

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
    
    if st.session_state.current_location_id:
        st.info(f"📍 Location: {st.session_state.current_location_id}")

    if st.button("Restart Game"):
        st.session_state.messages = []
        if st.session_state.game_data:
            st.session_state.current_location_id = list(st.session_state.game_data["locations"].keys())[0]
        st.rerun()

# 主界面
st.title("Elden Ring: The Shattered Conversation")
st.caption(f"Guide: **{st.session_state.current_persona}** | Mode: **Interactive RPG**")

# 显示历史
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🧙‍♀️"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# --- 游戏交互区域 (Action Bar) ---
if st.session_state.game_data and st.session_state.current_location_id:
    
    current_loc = st.session_state.game_data["locations"].get(st.session_state.current_location_id)
    
    if current_loc:
        st.write("---")
        st.subheader("⚔️ Actions & Travel")
        
        exits = current_loc.get("exits", [])
        cols = st.columns(len(exits) + 1 if exits else 1)
        
        for idx, exit_id in enumerate(exits):
            dest_name = st.session_state.game_data["locations"].get(exit_id, {}).get("name", exit_id)
            if cols[idx].button(f"👣 Go to {dest_name}", key=f"btn_{exit_id}"):
                handle_movement(exit_id)
        
        if "ending" in current_loc.get("events", [{}])[0]: 
            st.warning("✨ An Ending is upon you. Speak to make your choice.")

# 输入框
if prompt := st.chat_input("Speak to your guide..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)
    
    with st.chat_message("assistant", avatar="🧙‍♀️"):
        with st.spinner("Thinking..."):
            response_text, source_docs = generate_npc_response(prompt, st.session_state.current_persona)
            st.markdown(response_text)
            
            # --- ✅ 修复点：安全显示上下文 ---
            if source_docs:
                with st.expander("🔮 See Reasoning & Game Data"):
                    # 1. 修复 JSON 解析错误：改用 st.text
                    st.markdown("**Current Game Logic State:**")
                    st.text(get_current_game_context()) 
                    
                    # 2. 修复 Numpy 报错：改用 st.markdown + 循环，不使用 st.write([])
                    st.markdown("---")
                    st.markdown("**Retrieval Context (RAG):**")
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Fragment {i+1}:**")
                        st.caption(doc.page_content[:300] + "...") # 安全截取字符串
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
