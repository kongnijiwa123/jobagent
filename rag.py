"""
RAG 模块：把 jobs.db 里的 JD 原文建成向量库，然后用自然语言提问

  python rag.py build                 # 建索引（JD 有新增后重跑，增量）
  python rag.py ask "哪些岗位要求 MCP"  # 提问
  python rag.py                       # 交互模式，连续问

技术栈：LangChain + Chroma（本地向量库）+ bge-small-zh（本地中文 embedding）+ DeepSeek（生成）

安装：
  pip install langchain langchain-community langchain-openai langchain-chroma chromadb sentence-transformers

第一次 build 会下载 bge-small-zh-v1.5 模型（约 100MB）。国内下载慢的话先设镜像：
  $env:HF_ENDPOINT="https://hf-mirror.com"

原理（面试会问）：
  1. 切块：每条 JD 按 500 字切，带上公司/岗位/薪资等元数据
  2. 向量化：bge 把每块文本变成 512 维向量，存进 Chroma
  3. 检索：问题也向量化，找最相似的 k 块
  4. 生成：把这 k 块 + 问题一起喂 DeepSeek，让它只根据这些材料回答并标注来源
"""

import os, sys, json, sqlite3, hashlib

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

DB_PATH = "jobs.db"
INDEX_DIR = "chroma_index"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 6

NOISE = ("收藏", "举报", "去App", "前往App", "与BOSS", "扫码", "立即沟通")


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_store():
    return Chroma(collection_name="jobs",
                  embedding_function=get_embeddings(),
                  persist_directory=INDEX_DIR)


def clean(raw: str) -> str:
    return "\n".join(l for l in raw.splitlines()
                     if l.strip() and not any(x in l for x in NOISE))


# ---------------------------------------------------------------- build

def build():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    rows = db.execute("select * from jobs").fetchall()
    store = get_store()

    # 增量：已入库的 job id 跳过
    existing = set()
    got = store.get(include=["metadatas"])
    for m in got["metadatas"]:
        existing.add(m["job_id"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", "；", "。", "，", " "])

    docs, n_new = [], 0
    for r in rows:
        if r["id"] in existing:
            continue
        n_new += 1
        header = (f"公司：{r['company']}｜岗位：{r['title']}｜城市：{r['city']}｜"
                  f"薪资：{r['salary_min']}-{r['salary_max']}k｜经验：{r['exp_years']}年｜"
                  f"学历：{r['degree']}｜类型：{r['role_type']}\n")
        meta = {"job_id": r["id"], "company": r["company"] or "", "title": r["title"] or "",
                "city": r["city"] or "", "role_type": r["role_type"] or "",
                "salary_min": r["salary_min"] or 0, "salary_max": r["salary_max"] or 0,
                "exp_years": r["exp_years"] if r["exp_years"] is not None else -1,
                "degree": r["degree"] or ""}
        for i, chunk in enumerate(splitter.split_text(clean(r["raw"]))):
            # 每块都带上 header，这样检索到任何一块都知道是哪个岗位
            docs.append(Document(page_content=header + chunk,
                                 metadata={**meta, "chunk": i}))

    if docs:
        ids = [hashlib.md5((d.metadata["job_id"] + str(d.metadata["chunk"])).encode()).hexdigest()
               for d in docs]
        print("===== 这里在算向量/调模型")
        store.add_documents(docs, ids=ids)
    print(f"索引完成：新增 {n_new} 条 JD，{len(docs)} 个文本块；"
          f"库内共 {len(existing) + n_new} 条 JD")


# ---------------------------------------------------------------- ask

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是求职助手。只根据下面提供的招聘材料回答问题，材料里没有的不要编。\n"
     "回答时按岗位逐条列出，每条格式：「公司｜岗位｜薪资」+ 一句话说明为什么符合（引用原文关键句）。\n"
     "如果材料里没有符合条件的岗位，直接说没有。\n\n"
     "招聘材料：\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs):
    seen, parts = set(), []
    for d in docs:
        key = (d.metadata["job_id"], d.metadata["chunk"])
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"---\n{d.page_content}")
    return "\n".join(parts)


def ask(question: str, k: int = TOP_K, show_sources=True):
    store = get_store()
    retriever = store.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "deepseek-chat"),
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    )
    chain = PROMPT | llm
    print("===== 这里在算向量/调模型")
    docs = retriever.invoke(question)
    print("===== 这里在算向量/调模型")
    answer = chain.invoke({"context": format_docs(docs), "question": question}).content

    print("\n" + answer)
    if show_sources:
        hit = {}
        for d in docs:
            hit.setdefault(d.metadata["job_id"],
                           f"{d.metadata['company']}｜{d.metadata['title']}")
        print("\n[检索到的岗位] " + "；".join(hit.values()))
    return answer


def repl():
    print("输入问题，回车提问；输入 q 退出。示例：\n"
          "  哪些岗位要求 MCP 或 Function Calling\n"
          "  成都 15k 以上、不要求硕士的开发岗有哪些\n"
          "  哪些岗位需要出差或驻场\n"
          "  要求 RAG 经验的岗位，具体要求是什么\n")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if q:
            ask(q)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        build()
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        ask(" ".join(sys.argv[2:]))
    else:
        if not os.path.exists(INDEX_DIR):
            print("还没建索引，先运行：python rag.py build")
        else:
            repl()
