"""
求职情报 Agent · 第 2 周 v2
功能：读取 jds/ 目录下的 JD 文本 → DeepSeek 抽成结构化 JSON → 存入 SQLite

和 v1 的区别：
  1. 新增字段 role_type（岗位类型：开发/产品/运营/标注评测/售前/算法/其他）
  2. 新增字段 is_coding（是否以写代码为主）、remote（是否可远程）、note（备注）
  3. 用 DeepSeek 的 JSON 模式输出，不用再手动去 ``` 围栏
  4. 老库自动加列；加 --redo 参数可以把已入库的 JD 用新 prompt 重跑一遍
  5. JSON 解析失败自动重试 1 次

准备：
  pip install openai
  set LLM_API_KEY=你的key        （PowerShell 用 $env:LLM_API_KEY="你的key"）

运行：
  python jd_extract.py           # 只处理新 JD
  python jd_extract.py --redo    # 全部重跑（改了 prompt 之后用）
"""

import os, sys, json, glob, sqlite3, hashlib, time
from openai import OpenAI

client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

PROMPT = """你是招聘信息抽取器。从 JD 中抽取字段，只输出一个 JSON 对象。

字段说明：
- company: 公司名，没有写 null
- title: 岗位名（原文）
- city: 城市，没有写 null
- salary_min / salary_max: 月薪（单位 k，整数），没有写 null
- exp_years: 要求的最低工作年限（整数；应届/不限/经验不限填 0），没有写 null
- degree: 学历要求，只能是 "本科" / "硕士" / "博士" / "大专" / "不限"
- role_type: 岗位类型，只能是下面之一：
    "开发"      —— 主要工作是写代码做 AI 应用、Agent、RAG、后端、全栈
    "算法"      —— 模型训练、微调、算法研究
    "产品"      —— 产品经理、需求、方案设计为主
    "运营"      —— 运营、数据运营、内容运营
    "标注评测"  —— 数据标注、评测、质检、数据生产
    "售前"      —— 售前、解决方案、客户对接
    "其他"
  判断依据是「职位描述」里的主要日常工作，不要只看标题。
- is_coding: 日常工作是否以写代码为主（true/false）
- company_type: 只能是 "大厂" / "外包" / "中小软件公司" / "国企或银行科技" / "AI创业公司" / "其他"
- tech_stack: 技术关键词数组，例如 ["Python","LangChain","RAG","SQL"]；JD 没提技术就给 []
- domain: 业务领域（金融/政企/工业/通信/电商/游戏/医药/通用/其他）
- travel: 是否需要出差或驻场（true/false/null）
- remote: 是否可远程（true/false/null）
- note: 一句话备注你觉得求职者该注意的点（比如"实际是外包""要求多方向任选其一"），没有写 null

JD：
<<<
{jd}
>>>"""

FIELDS = ["company", "title", "city", "salary_min", "salary_max", "exp_years", "degree",
          "role_type", "is_coding", "company_type", "tech_stack", "domain",
          "travel", "remote", "note"]


def extract(jd_text: str) -> dict:
    """调模型抽取，失败重试一次"""
    for attempt in range(2):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT.format(jd=jd_text)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        try:
            d = json.loads(raw)
            return {k: d.get(k) for k in FIELDS}
        except json.JSONDecodeError:
            if attempt == 0:
                time.sleep(1)
                continue
            raise


def init_db(path="jobs.db"):
    db = sqlite3.connect(path)
    db.execute("""create table if not exists jobs(
        id text primary key, company text, title text, city text,
        salary_min int, salary_max int, exp_years int, degree text,
        company_type text, tech_stack text, domain text, travel int,
        raw text, created_at text default current_timestamp)""")
    # 老库补列：有就跳过
    existing = {r[1] for r in db.execute("pragma table_info(jobs)")}
    for col, typ in [("role_type", "text"), ("is_coding", "int"),
                     ("remote", "int"), ("note", "text"), ("checked", "int")]:
        if col not in existing:
            db.execute(f"alter table jobs add column {col} {typ}")
    db.commit()
    return db


def upsert(db, jid, d, jd):
    db.execute("""insert or replace into jobs
        (id, company, title, city, salary_min, salary_max, exp_years, degree,
         company_type, tech_stack, domain, travel, raw, created_at,
         role_type, is_coding, remote, note, checked)
        values (?,?,?,?,?,?,?,?,?,?,?,?,?,current_timestamp,?,?,?,?,
                coalesce((select checked from jobs where id=?), 0))""", (
        jid, d["company"], d["title"], d["city"],
        d["salary_min"], d["salary_max"], d["exp_years"], d["degree"],
        d["company_type"], json.dumps(d["tech_stack"] or [], ensure_ascii=False),
        d["domain"], d["travel"], jd,
        d["role_type"], d["is_coding"], d["remote"], d["note"], jid))
    db.commit()


def main():
    redo = "--redo" in sys.argv
    db = init_db()
    files = glob.glob("jds/*.txt")
    print(f"找到 {len(files)} 条 JD" + ("（重跑模式）" if redo else ""))
    ok = fail = skip = 0
    for f in files:
        jd = open(f, encoding="utf-8").read()
        jid = hashlib.md5(jd.encode()).hexdigest()
        if not redo and db.execute("select 1 from jobs where id=?", (jid,)).fetchone():
            skip += 1
            continue
        try:
            d = extract(jd)
        except Exception as e:
            print(f"失败: {f} -> {e}")
            fail += 1
            continue
        upsert(db, jid, d, jd)
        ok += 1
        print(f"入库: {d['company']} | {d['title']} | {d['role_type']} | "
              f"{d['salary_min']}-{d['salary_max']}k | {d['exp_years']}年 | {d['degree']}")
    print(f"\n完成：入库 {ok}，跳过 {skip}，失败 {fail}")


if __name__ == "__main__":
    main()
