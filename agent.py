"""
Agent：一个入口，三个工具，模型自己决定走哪条路

  工具 1  query_sql      —— 精确条件、数值筛选、关键词、统计    （SQL，一定对）
  工具 2  search_jd      —— 语义问题、"具体要求是什么"类        （RAG，按意思找）
  工具 3  read_resume    —— 问题涉及"我"时，读取简历

运行：
  python agent.py                       # 交互模式
  python agent.py "成都15k以上不要硕士的开发岗"

原理（面试会问）：
  这是最基础的 tool-calling Agent 循环：
    1. 把工具的「名字 + 说明 + 参数」告诉模型
    2. 模型看问题，决定调哪个工具、传什么参数（这一步叫 function calling）
    3. 程序真正执行工具，把结果塞回对话
    4. 模型看结果，要么继续调工具，要么给出最终回答
  模型不执行任何东西，它只"下指令"；执行的是你的代码。
"""

import os, sys, json, sqlite3
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

import rag  # 复用 rag.py 的向量库

DB_PATH = "jobs.db"
MAX_STEPS = 6

SCHEMA = """表 jobs 的列：
  company(公司) title(岗位名) city(城市) salary_min salary_max(月薪k，整数)
  exp_years(最低经验年限，0=不限) degree(本科/硕士/博士/大专/不限)
  role_type(开发/算法/产品/运营/标注评测/售前/其他) is_coding(0/1)
  company_type(大厂/外包/中小软件公司/国企或银行科技/AI创业公司/其他)
  tech_stack(JSON数组字符串，如 ["Python","RAG"]) domain(业务领域)
  travel remote(0/1/NULL) note(备注) raw(JD原文全文)
关键词查原文用  raw like '%MCP%' ；查技术栈用  tech_stack like '%RAG%'"""


# ---------------------------------------------------------------- tools

@tool(description="""对岗位数据库执行 SELECT 查询。适用于：数值筛选（薪资、经验）、精确条件（城市、学历、岗位类型）、
关键词是否出现（如 MCP、LangChain）、统计计数。只允许 SELECT。
""" + SCHEMA)
def query_sql(sql: str) -> str:
    s = sql.strip().rstrip(";")
    if not s.lower().startswith("select"):
        return "拒绝：只允许 SELECT"
    try:
        db = sqlite3.connect(DB_PATH)
        cur = db.execute(s)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()[:40]
        if not rows:
            return "查询成功，0 行"
        out = [" | ".join(cols)]
        out += [" | ".join("" if v is None else str(v)[:120] for v in r) for r in rows]
        return f"{len(rows)} 行：\n" + "\n".join(out)
    except Exception as e:
        return f"SQL 错误：{e}"


@tool
def search_jd(question: str) -> str:
    """按语义在 JD 原文中检索最相关的片段。适用于：某类岗位的具体要求、职责描述、
    需要阅读原文才能回答的问题。不适合数值筛选和精确关键词统计（那些用 query_sql）。"""
    store = rag.get_store()
    docs = store.as_retriever(search_kwargs={"k": 6}).invoke(question)
    return rag.format_docs(docs)


@tool
def read_resume() -> str:
    """读取求职者本人的简历。当问题涉及"我"（我的差距、我能投哪些、我匹配吗）时调用。"""
    if not os.path.exists("resume.md"):
        return "没有找到 resume.md"
    return open("resume.md", encoding="utf-8").read()


TOOLS = [query_sql, search_jd, read_resume]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM = """你是求职情报助手，手里有一个 AI 应用/Agent 方向的岗位数据库。
回答前先判断该用哪个工具：
- 涉及数字、范围、精确条件、"有多少"、"哪些包含某个词" → query_sql（写 SQL 时严格按列名）
- 涉及"具体要求""职责是什么""怎么描述的" → search_jd
- 涉及"我" → 先 read_resume，再配合其他工具
可以连续调用多个工具。最终回答只基于工具返回的数据，不要编造；列岗位时写「公司｜岗位｜薪资」。"""


# ---------------------------------------------------------------- loop

def run(question: str, verbose=True) -> str:
    llm = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "deepseek-chat"),
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    ).bind_tools(TOOLS)                         # 把工具说明书交给模型

    messages = [SystemMessage(SYSTEM), HumanMessage(question)]

    for step in range(MAX_STEPS):
        resp = llm.invoke(messages)             # 模型决定：调工具 or 直接回答
        messages.append(resp)

        if not resp.tool_calls:                 # 没有工具调用 → 这就是最终答案
            return resp.content

        for call in resp.tool_calls:            # 有工具调用 → 逐个执行
            name, args = call["name"], call["args"]
            if verbose:
                shown = json.dumps(args, ensure_ascii=False)
                print(f"  [step {step+1}] → {name}({shown[:150]})")
            result = TOOL_MAP[name].invoke(args)
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "超过最大步数，停止。"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("\n" + run(" ".join(sys.argv[1:])))
    else:
        print("输入问题，q 退出。试试：\n"
              "  成都15k以上不要硕士的开发岗有哪些\n"
              "  哪些岗位提到MCP\n"
              "  要求RAG的岗位具体要求是什么\n"
              "  我和盯潮那个岗位的差距在哪\n")
        while True:
            q = input("\n> ").strip()
            if q.lower() in ("q", "quit", "exit"):
                break
            if q:
                print("\n" + run(q))
