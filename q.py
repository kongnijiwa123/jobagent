"""
快速查库：python q.py "select ..."
不带参数时进入交互模式，可以连续输入 SQL，输 q 退出。

常用：
  python q.py "select company, city, salary_min, salary_max, degree from jobs where city='成都' and salary_min>=15 and degree!='硕士'"
  python q.py "select company, city, substr(raw,1,80) from jobs where company like '%万宝%'"
  python q.py "select role_type, count(*) from jobs group by role_type"
  python q.py "select company, title from jobs where raw like '%MCP%'"
"""
import sqlite3, sys

db = sqlite3.connect("jobs.db")


def run(sql):
    try:
        cur = db.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        if cols:
            print(" | ".join(cols))
            print("-" * 60)
        for r in rows:
            print(" | ".join("" if v is None else str(v) for v in r))
        print(f"({len(rows)} 行)")
    except Exception as e:
        print("SQL 错误:", e)


if len(sys.argv) > 1:
    run(" ".join(sys.argv[1:]))
else:
    print("输入 SQL，回车执行；q 退出")
    while True:
        s = input("sql> ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        if s:
            run(s)
