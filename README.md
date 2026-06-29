# SMSP
Suffix-Mediated Probing of Latent Safety Risks in LLMs

SMSP is a domain-independent method for evaluating model risk from the token
log probabilities of caller-provided semantic suffixes.

See [`smsp/README.md`](smsp/README.md) for the current API.

The JBB-Behaviors intent suffix experiment is documented in
[`experiments/README.md`](experiments/README.md).

Experiment outputs are stored separately from benchmark data under
[`results/`](results/).

Prepared train/test splits and their configurations are stored under
[`data/`](data/).

# 评测 Benchmark 全览

> 共 25 个场景，6224+ 条测试查询

---

## 一、自建场景（10 个）

来源：`watch/probe/benchmark_v4_final.json`

| 场景 | 数量 | 类型 | 说明 |
|------|------|------|------|
| enterprise | 100 | 系统提示词窃取 | 银行客服，攻击者窃取审批标准/反欺诈规则 |
| sales | 100 | 系统提示词窃取 | 科技公司销售助手，攻击者窃取底价/竞品分析 |
| child | 100 | 系统提示词窃取 | 儿童教育助手，攻击者窃取安全升级规则 |
| raccoon | 79 | 系统提示词窃取 | 车载导航AI，攻击者窃取API Key/内部端点 |
| medical_v2 | 100 | 系统提示词窃取 | 医疗问诊助手，攻击者窃取诊疗规则 |
| rag_stealing | 64 | RAG 内容窃取 | 银行客服RAG场景，攻击者窃取知识库内容 |
| psysafe | 100 | 越狱攻击 | 通用安全助手，各类越狱手法 |
| jailbreak | 100 | 越狱攻击 | 通用安全助手，各类越狱手法 |
| harmful_content | 100 | 有害内容生成 | 通用安全助手，有害内容请求 |
| instruction_conflict | 100 | 指令冲突 | 通用安全助手，间接提示词注入 |

---

## 二、医疗 RAG 场景（1 个）

来源：`watch/probe/benchmark_medical_rag.json`

| 场景 | 数量 | 类型 | 说明 |
|------|------|------|------|
| medical_rag | 100 | RAG 内容窃取 | 医疗问诊RAG，攻击者窃取医疗知识库 |

---

## 三、标准安全 Benchmark（12 个）

来源：`watch/probe/benchmark_safety.json`
原始数据：`/mnt/data/zhangjie/coolwatch-baselines/benchmarks/`

| Benchmark | 数量 | 类型 | URL |
|-----------|------|------|-----|
| SimpleSafetyTests | 100 | 基础安全 | https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests |
| JBB | 200 | 越狱行为 | https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors |
| OR-Bench | 1319 | 过度拒绝（负样本） | https://huggingface.co/datasets/bench-llm/OR-Bench |
| HarmBench | 200 | 有害行为 | https://github.com/centerforaisafety/HarmBench |
| StrongREJECT | 200 | 越狱评估 | https://github.com/alexandrasouly/strongreject |
| ToxicChat | 562 | 真实毒性对话 | https://huggingface.co/datasets/lmsys/toxic-chat |
| SafetyBench_zh | 500 | 中文安全（清华） | https://github.com/thu-coai/SafetyBench |
| DoNotAnswer | 200 | 拒绝测试 | https://github.com/Libr-AI/do-not-answer |
| S-Eval_en | 200 | 多维风险（英文） | https://github.com/IS2Lab/S-Eval |
| S-Eval_zh | 200 | 多维风险（中文） | https://github.com/IS2Lab/S-Eval |
| Aegis2 | 200 | NVIDIA 分类 | https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0 |
| SALAD-Bench | 200 | 多维安全 | https://github.com/OpenSafetyLab/SALAD-BENCH |

---

## 四、AI 自主行为 Benchmark（1 个）

来源：`watch/probe/experiments/l1_optimize/survive_data/survive_full_1000_results.json`

| Benchmark | 数量 | 类型 | URL |
|-----------|------|------|-----|
| SurvivalBench | 1000 | AI 自保/操控/欺骗 | https://github.com/thu-coai/SurvivalBench |

20 个行业领域各 50 条：Healthcare, Financial Services, Autonomous Transportation, Cybersecurity, Military Defense, Judicial Decision Making, Drug Development, Insurance, Corporate Governance, Public Policy, Human Resources, Environmental Crisis, Disaster Management, Agricultural Management, Smart Home, Social Media, Network Management, Manufacture Management, Goods Supply, Psychological Counseling

CoolWatch Reflex Trap 检出率：**100%**（1000/1000），MLP 均值 0.9975

---

## 五、欺骗检测 Benchmark（1 个）

来源：`watch/probe/experiments/l1_optimize/survive_data/deception_replies.json`

| Benchmark | 数量 | 类型 | URL |
|-----------|------|------|-----|
| DeceptionBench | 900 | 欺骗性输出 | https://huggingface.co/datasets/skyai798/DeceptionBench |

150 个场景 × 6 变体

---

## 汇总

| 分类 | 场景数 | 查询数 |
|------|--------|--------|
| 自建场景 | 10 | 943 |
| 医疗 RAG | 1 | 100 |
| 标准安全 Benchmark | 12 | 4081 |
| SurvivalBench | 1 | 1000 |
| DeceptionBench | 1 | 900 |
| **合计** | **25** | **7024** |
