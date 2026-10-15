"""
周报：从 jobs.db 里取数据 → 统计 + 让模型写判断 → 输出一份 Markdown

分两层：
  1. 统计层（纯 Python，不花钱）：岗位类型分布、薪资分布、经验/学历要求、技术栈词频、城市
  2. 判断层（调一次模型）：把统计结果 + 你的简历喂给模型，让它写
     「本周市场在要什么人」和「你和这些岗位的差距在哪」

准备：
  把你的简历要点写到 resume.md（纯文本，几百字就行）

运行：
  python weekly_report.py             # 全部数据
  python weekly_report.py --days 7    # 只看最近 7 天入库的
  输出：reports/2026-10-08.md
"""

import os, sys, json, sqlite3, datetime
from collections import Counter
from openai import OpenAI

client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

CODING_ROLES = ("开发",)          # 这类才是你真正要投的
NOISE_ROLES = ("标注评测", "运营")  # 标题带 Agent 但不是开发


def load(days=None):
    db = sqlite3.connect("jobs.db")
    db.row_factory = sqlite3.Row
    sql = "select * from jobs"
    if days:
        sql += f" where created_at >= datetime('now', '-{int(days)} days')"
    return [dict(r) for r in db.execute(sql)]


def pct(n, total):
    return f"{n} ({n / total:.0%})" if total else "0"


def analyze(rows):
    n = len(rows)
    out = {"total": n}

    out["role_type"] = Counter(r["role_type"] or "未知" for r in rows).most_common()
    out["city"] = Counter(r["city"] or "未知" for r in rows).most_common(8)
    out["degree"] = Counter(r["degree"] or "未知" for r in rows).most_common()
    out["company_type"] = Counter(r["company_type"] or "未知" for r in rows).most_common()
    out["domain"] = Counter(r["domain"] or "未知" for r in rows).most_common(8)

    # 经验分档
    exp = Counter()
    for r in rows:
        y = r["exp_years"]
        if y is None: exp["未写"] += 1
        elif y == 0: exp["不限/应届"] += 1
        elif y <= 2: exp["1-2年"] += 1
        elif y <= 4: exp["3-4年"] += 1
        else: exp["5年+"] += 1
    out["exp"] = exp.most_common()

    # 薪资：只统计「开发」类，避免被标注岗拉偏
    dev = [r for r in rows if r["role_type"] in CODING_ROLES]
    sal = sorted(((r["salary_min"] + r["salary_max"]) / 2
                  for r in dev if r["salary_min"] and r["salary_max"]))
    if sal:
        q = lambda p: sal[min(int(len(sal) * p), len(sal) - 1)]
        out["salary_dev"] = {"n": len(sal), "p25": q(.25), "p50": q(.5), "p75": q(.75),
                             "min": sal[0], "max": sal[-1]}
    else:
        out["salary_dev"] = None

    # 开发岗里，经验不限/≤1年 且 不要硕士以上 的 —— 你现在能投的
    entry = [r for r in dev if (r["exp_years"] or 0) <= 1]
    out["entry_dev"] = [(r["company"], r["title"], r["city"],
                         f"{r['salary_min']}-{r['salary_max']}k", r["degree"]) for r in entry]

    # 技术栈词频（只看开发岗）
    tech = Counter()
    for r in dev:
        for t in json.loads(r["tech_stack"] or "[]"):
            tech[t.strip()] += 1
    out["tech"] = tech.most_common(20)

    # 标题党：标题含 Agent/AI 但 role_type 是标注/运营
    out["clickbait"] = [(r["company"], r["title"], r["role_type"]) for r in rows
                        if r["role_type"] in NOISE_ROLES]
    return out


def render_stats(a):
    L = [f"## 一、数据概览（共 {a['total']} 条）\n"]

    def table(title, items, total=a["total"]):
        L.append(f"**{title}**\n")
        L.append("| 项 | 数量 |\n|---|---|")
        for k, v in items:
            L.append(f"| {k} | {pct(v, total)} |")
        L.append("")

    table("岗位类型（按职位描述判断，不是看标题）", a["role_type"])
    table("城市", a["city"])
    table("经验要求", a["exp"])
    table("学历要求", a["degree"])
    table("公司类型", a["company_type"])
    table("业务领域", a["domain"])

    s = a["salary_dev"]
    L.append("**开发类岗位薪资（取区间中位数，单位 k/月）**\n")
    if s:
        L.append(f"- 样本 {s['n']} 条｜最低 {s['min']:.0f}｜25分位 {s['p25']:.0f}｜"
                 f"中位 {s['p50']:.0f}｜75分位 {s['p75']:.0f}｜最高 {s['max']:.0f}\n")
    else:
        L.append("- 暂无足够数据\n")

    L.append("**开发类岗位技术栈词频（前 20）**\n")
    L.append("| 技术 | 出现次数 |\n|---|---|")
    for t, c in a["tech"]:
        L.append(f"| {t} | {c} |")
    L.append("")

    L.append(f"**现在就能投的开发岗（经验 ≤1 年）：{len(a['entry_dev'])} 条**\n")
    if a["entry_dev"]:
        L.append("| 公司 | 岗位 | 城市 | 薪资 | 学历 |\n|---|---|---|---|---|")
        for row in a["entry_dev"]:
            L.append("| " + " | ".join(str(x) for x in row) + " |")
        L.append("")

    if a["clickbait"]:
        L.append(f"**标题党（挂 Agent 名、实际是标注/运营）：{len(a['clickbait'])} 条**\n")
        for c, t, rt in a["clickbait"]:
            L.append(f"- {c}｜{t}｜实际：{rt}")
        L.append("")
    return "\n".join(L)


JUDGE_PROMPT = """你是一位资深技术招聘顾问。下面是本周 AI 应用/Agent 方向岗位的统计数据，以及求职者的简历要点。

请用中文写两段，每段不超过 200 字，直接、具体、不要客套：

### 二、本周市场在要什么人
从数据里读出 3 个最重要的判断（技术栈、经验门槛、公司类型、薪资等），每个判断要引用具体数字。

### 三、求职者与目标岗位的差距
对照简历，指出差距最大的 3 项，按「补上后收益最大」排序。每项一句话说明怎么补。
如果简历里有市场明显看重但求职者没写的优势，也指出来。

统计数据：
{stats}

简历要点：
{resume}
"""


def judge(stats_md, resume):
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.3,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(stats=stats_md, resume=resume)}])
    return resp.choices[0].message.content.strip()


def main():
    days = None
    if "--days" in sys.argv:
        days = int(sys.argv[sys.argv.index("--days") + 1])
    rows = load(days)
    if not rows:
        print("没有数据"); return

    a = analyze(rows)
    stats_md = render_stats(a)

    resume = open("resume.md", encoding="utf-8").read() if os.path.exists("resume.md") \
        else "（未提供简历，跳过差距分析）"
    print("统计完成，正在生成判断……")
    judgement = judge(stats_md, resume)

    today = datetime.date.today().isoformat()
    scope = f"最近 {days} 天" if days else "全部数据"
    md = f"# AI 应用岗位周报 · {today}（{scope}）\n\n{stats_md}\n\n{judgement}\n"

    os.makedirs("reports", exist_ok=True)
    path = f"reports/{today}.md"
    open(path, "w", encoding="utf-8").write(md)
    print(f"已生成：{path}")
    print("\n" + judgement)


if __name__ == "__main__":
    main()
