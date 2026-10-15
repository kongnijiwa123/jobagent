# jobagent · 求职情报 Agent

面向「成都 AI 应用 / Agent 开发岗」的个人求职情报工具：手动收集招聘 JD → 大模型结构化抽取 → SQLite 存储 → 周报 / RAG 问答 / tool-calling Agent 路由。

这是我转向 AI 应用开发方向的第一个完整项目。重点不在功能多，而在**每一步都做了验证**：抽取结果有人工抽样校验，RAG 的缺陷有实测数据，Agent 路由有对照测试。

## 做了什么

```
jds/*.txt ──► jd_extract.py ──► jobs.db ──┬──► weekly_report.py  周报（统计 + 简历差距）
 (30 条 JD)   (DeepSeek JSON 抽取)         ├──► rag.py            向量检索问答
                                           └──► agent.py          tool-calling Agent
                                                  ├─ query_sql    精确条件 / 统计（SQL）
                                                  ├─ search_jd    语义检索（RAG）
                                                  └─ read_resume  涉及"我"时读简历
```

| 文件 | 作用 |
|---|---|
| `jd_extract.py` | JD 原文 → DeepSeek JSON 模式抽取 15 个字段（薪资、经验、学历、岗位类型、技术栈…）→ 入库 |
| `check_sample.py` | 人工抽样校验：事实字段算准确率，推断字段只评合理性，技术栈自动查编造 |
| `weekly_report.py` | 统计（岗位类型 / 薪资分位 / 技术栈词频 / 可投清单 / 标题党）+ 模型写市场判断与简历差距 |
| `rag.py` | LangChain + Chroma + bge-small-zh；按 JD 切块、带元数据、增量建索引 |
| `agent.py` | 手写 tool-calling 循环，三个工具，模型按问题类型自选 |
| `q.py` | 快速 SQL |

## 几个实测结论

**30 条 JD 里 8 条是标题党** —— 标题挂"Agent"，职位描述实际是数据标注或运营。按职位描述而非标题判断 `role_type` 能识别出来。

**向量检索在结构化查询上不可靠**（这是做 agent.py 的直接原因）：

| 问题 | SQL | RAG |
|---|---|---|
| 成都 15k 以上、不要硕士的开发岗 | 6 条 | 2 条（漏掉 20-40k、30-60k 两条高薪） |
| 提到 MCP 的岗位 | 7 条 | 0 条，且自信地回答"没有" |

原因：向量匹配的是语义相似度，不是条件；"MCP"这类短缩写在向量空间里辨识度低；只看 top-k 块就会漏。

**解法是路由而不是调参**：agent.py 让模型根据问题类型选工具——数值 / 精确条件走 SQL，"具体要求是什么"走 RAG，涉及"我"先读简历。四个测试问题全部走对，其中一题模型自发同时调了 RAG 和 SQL 再合并回答。

**首份周报数据**：开发岗薪资中位 16k、75 分位 22k；经验 ≤1 年可投 14 条；硕士门槛仅 2 条；技术栈词频 Python 15、RAG 15、LangChain 12、MCP 6。

## 运行

```powershell
pip install openai langchain langchain-community langchain-openai langchain-chroma chromadb sentence-transformers
$env:LLM_API_KEY = "sk-..."          # DeepSeek，OpenAI 兼容接口
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 国内下载 bge 模型

python jd_extract.py                 # jds/ 下新 JD 入库
python check_sample.py               # 抽样校验
python weekly_report.py              # 生成 reports/日期.md
python rag.py build                  # 建向量索引
python agent.py "哪些岗位提到MCP"     # 或不带参数进交互模式
```

`jobs.db`、`chroma_index/`、`jds/` 不在仓库里（含招聘原文），把自己收集的 JD 放进 `jds/` 一条一个 txt 即可。

## 没做的

自动抓取（反爬耗时不可控，手动 30 条已够用）、网页界面（命令行交互演示更直接）、给岗位自动打分（投不投该自己看着 JD 判断，模型打分只是噪音）。
