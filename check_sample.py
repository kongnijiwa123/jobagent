"""
抽样校验 v3：把字段分成两组打分

  事实字段（原文一定有答案，算准确率）：city / salary / exp_years / degree / role_type
  推断字段（原文多半没写，只评"合不合理"）：company_type / domain / travel / remote
  tech_stack 只看有没有编造（抽出来的词原文里找不到）

运行：
  python check_sample.py        # 抽 5 条没检查过的
  python check_sample.py 10     # 抽 10 条
  python check_sample.py stats  # 看统计
"""

import sys, json, sqlite3, random

db = sqlite3.connect("jobs.db")
db.row_factory = sqlite3.Row

# 老库补列
existing = {r[1] for r in db.execute("pragma table_info(jobs)")}
for col in ("fact_ok", "infer_ok", "stack_ok", "check_note"):
    if col not in existing:
        db.execute(f"alter table jobs add column {col} int" if col != "check_note"
                   else "alter table jobs add column check_note text")
db.commit()

NOISE = ("收藏", "举报", "去App", "前往App", "与BOSS", "职位描述", "工作地址",
         "扫码", "立即沟通")

EVIDENCE = {
    "exp_years": ["经验", "年以上", "年及以上", "年左右", "应届"],
    "degree":    ["本科", "硕士", "博士", "大专", "学历"],
    "role_type": ["负责", "职责", "日常", "主要工作", "我们正在"],
    "travel":    ["出差", "驻场", "现场"],
    "remote":    ["远程", "居家"],
    "company":   ["外包", "外派", "驻"],
}


def lines_of(raw):
    return [l.strip() for l in raw.splitlines()
            if l.strip() and not any(x in l for x in NOISE)]


def show(r):
    lines = lines_of(r["raw"])
    print("【原文头部】")
    for l in lines[:5]:
        print("   " + l)

    print("\n【原文证据】")
    seen = set()
    for field, kws in EVIDENCE.items():
        hits = [l for l in lines[5:] if any(k in l for k in kws) and l not in seen][:2]
        for h in hits:
            seen.add(h)
            print(f"   [{field}] {h[:90]}")

    print("\n【事实字段 —— 对照上面打分】")
    print(f"   城市={r['city']}   薪资={r['salary_min']}-{r['salary_max']}k   "
          f"经验={r['exp_years']}年   学历={r['degree']}   岗位类型={r['role_type']}")

    print("\n【推断字段 —— 只看合不合理】")
    print(f"   公司类型={r['company_type']}   领域={r['domain']}   "
          f"出差={r['travel']}   远程={r['remote']}   写代码为主={r['is_coding']}")
    print(f"   备注={r['note']}")

    ts = json.loads(r["tech_stack"] or "[]")
    miss = [t for t in ts if t.lower() not in r["raw"].lower()]
    print(f"\n【技术栈】共 {len(ts)} 个，原文找不到的：{miss or '无'}")
    return miss


def ask(prompt, choices):
    while True:
        a = input(prompt).strip().lower()
        if a in choices:
            return a


def review(n):
    rows = db.execute("select * from jobs where fact_ok is null").fetchall()
    if not rows:
        print("没有待检查的记录了。")
        return
    for i, r in enumerate(random.sample(rows, min(n, len(rows))), 1):
        print("\n" + "=" * 72)
        print(f"[{i}]  {r['company']} | {r['title']}")
        print("-" * 72)
        miss = show(r)
        print("-" * 72)

        a = ask("事实字段全对吗？ y / n / f=看全文 / s=跳过 / q=退出 > ", "ynfsq")
        while a == "f":
            print("\n" + r["raw"] + "\n")
            a = ask("事实字段全对吗？ y / n / s / q > ", "ynsq")
        if a == "q":
            break
        if a == "s":
            continue
        fact_ok = 1 if a == "y" else 0
        note = ""
        if not fact_ok:
            note = input("   哪个错了？（如：role_type应为产品）> ").strip()

        b = ask("推断字段合理吗？ y / n > ", "yn")
        infer_ok = 1 if b == "y" else 0
        if not infer_ok:
            note += " | 推断:" + input("   哪里不合理？> ").strip()

        stack_ok = 0 if miss else 1   # 自动判：有编造就算不合格

        db.execute("""update jobs set fact_ok=?, infer_ok=?, stack_ok=?, check_note=?
                      where id=?""", (fact_ok, infer_ok, stack_ok, note or None, r["id"]))
        db.commit()
    print()
    stats()


def stats():
    tot = db.execute("select count(*) from jobs").fetchone()[0]
    done = db.execute("select count(*) from jobs where fact_ok is not null").fetchone()[0]
    print(f"总计 {tot} 条，已检查 {done} 条")
    if not done:
        return
    f = db.execute("select avg(fact_ok), avg(infer_ok), avg(stack_ok) from jobs "
                   "where fact_ok is not null").fetchone()
    print(f"  事实字段准确率：{f[0]:.0%}   ← 这个是正经指标，目标 ≥ 90%")
    print(f"  推断字段合理率：{f[1]:.0%}")
    print(f"  技术栈无编造率：{f[2]:.0%}")
    errs = db.execute("select company, title, check_note from jobs "
                      "where fact_ok=0 or infer_ok=0").fetchall()
    if errs:
        print("\n有问题的记录（改 prompt 的依据）：")
        for r in errs:
            print(f"  - {r['company']} | {r['title']} | {r['check_note']}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "5"
    stats() if arg == "stats" else review(int(arg))
