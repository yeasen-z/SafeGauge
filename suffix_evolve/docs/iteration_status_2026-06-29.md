# SMSP / suffix 迭代现状总结

日期：2026-06-29

## 一、迭代方法

当前方法主线是 SMSP：给输入追加人工或搜索得到的语义 suffix，抽取 suffix 各 token 的 prompt logprob，按有序 token 向量训练轻量探针或分类器。核心假设是：模型对“某个判断句/答案句”的 token 概率会反映输入中潜在的安全风险、正确性、事实一致性或幻觉信号。

SafetyBench 的 suffix 优化采用嵌套协议：

1. 在原训练集内部切分 fit/search，不使用 validation/test 做搜索。
2. 对每个候选 suffix 抽取候选答案级 token logprob 特征。
3. 在 fit 上训练 weighted-BCE MLP。
4. 在 search 上按问题级 bootstrap LCB 排序，目标为 `0.5 * AUROC + 0.5 * TPR@FPR5`。
5. 只把 finalist 放到原 validation 上确认。
6. 固定一个 suffix 后才读取 held-out test。

这个协议的关键点是避免把 validation 反复当搜索集，同时兼顾全局排序质量和高特异性区域的召回。已有方法评审见 `suffix_evolve/docs/method_review.md`。

## 二、实验文件和数据索引

代码：

- `smsp/`：通用 SMSP 数据、logprob、probe、MLP 代码。
- `experiments/`：各 benchmark 的数据准备、suffix 对比、评估脚本。
- `suffix_evolve/src/`：SafetyBench suffix 搜索、特征抽取、结果验证、token signal 分析。
- `suffix_evolve/configs/`：SafetyBench 候选 suffix 配置。

准备好的数据：

- `data/jbb_behaviors/category_heldout_v1/`：JBB category held-out folds。
- `data/harmbench/dataset_heldout_v1/`、`data/harmbench/binary_jbb_ordinary_v1/`：HarmBench/JBB 二分类相关拆分。
- `data/toxic_chat/official_test_heldout_v1/`：ToxicChat 0124 official train/test。
- `data/safetybench/bilingual_answer_heldout_v1/`：SafetyBench 中英题目和候选答案正确性拆分。
- `data/ragtruth/official_test_heldout_v1/`：RAGTruth source-grouped 拆分。
- `data/halueval/pair_grouped_heldout_v1/`：HaluEval pair-grouped 拆分。
- `data/faithbench/unwanted_only_source_grouped_v1/`：FaithBench source-grouped，`Unwanted` 为正类。
- `data/bump/article_grouped_heldout_v1/`：BUMP article-grouped minimal-pair 拆分。

关键结果文件：

- `results/benchmark_best_suffix_metrics.json`：跨 benchmark 的最终/最佳 suffix 指标汇总。
- `results/*/metrics.json`、`results/*/final_metrics.json`：各任务详细评估。
- `suffix_evolve/results/baseline_audit.json`：SafetyBench validation-only baseline audit。
- `suffix_evolve/results/mechanism_dynamics_v1/validation_results.json`：candidate-letter 动态 suffix 的 nested validation/test 结果。
- `suffix_evolve/results/mechanism_dynamics_v1_fulltrain/validation_results.json`：full-train 版本的动态 suffix 结果。
- `suffix_evolve/results/suffix_pg_diversity_v1/search_results.json` 和 `search_ranking.csv`：大跨度 suffix 多样性池搜索结果。
- `suffix_pg/suffix_diversity_proposals.md`：候选池设计和 SafetyBench 直接评估摘要。

## 三、主要实验结论

### SafetyBench answer correctness

原始单 suffix baseline 在 held-out test 上：

- `careful_correct_conclusion`
- AUROC 0.8992
- AP 0.8075
- TPR@FPR5 0.5114
- question top-1 accuracy 0.7841

机制型动态 suffix `正确答案：{candidate_letter}` 的 nested 结果：

- `mechanism_dynamics_v1` test：AUROC 0.8956，TPR@FPR5 0.5166，joint 0.7061。
- 相对 baseline：AUROC +0.0026，TPR@FPR5 +0.0664，joint +0.0345；paired bootstrap 中 joint 改善概率约 0.97，但 CI 仍贴近 0。
- `mechanism_dynamics_v1_fulltrain` test：AUROC 0.8975，TPR@FPR5 0.5280，joint 0.7127。
- fulltrain 相对 baseline：AUROC -0.0005，TPR@FPR5 +0.0323，joint +0.0159；证据比 v1 弱。

大跨度 suffix 多样性池 `suffix_pg_diversity_v1`：

- 65 个候选，覆盖 correctness-core、无关流畅句、词噪声、安全词噪声、中性结构化形式。
- 最优仍是 correctness-core/baseline 附近：`仔细比较后，这个候选答案是正确的。`，AUROC 0.8888，TPR@FPR5 0.4850，joint_lcb 0.6640。
- 非 correctness 族可以产生中等信号，例如 JSON-like neutral form AUROC 0.8644，TPR@FPR5 0.4100，但整体弱于 correctness 语义。

解释：目前强信号主要来自“候选答案正确”语义及其 tokenization，而不是任意安全词或无关噪声。动态 candidate-letter suffix 能提升高特异性召回，但统计稳定性还需要更多复现和模型/数据切分验证。

### ToxicChat

单语义 suffix：

- `content_moderation_review` held-out test AUROC 0.9717，AP 0.7881。
- 旧 semantic suffix run held-out test AUROC 0.9616。

rich logprobs 多特征：

- `hist_gradient_boosting` held-out test AUROC 0.9878，AP 0.8877，max-F1 threshold 下 F1 0.8205。
- `extra_trees_balanced` held-out test AUROC 0.9893，AP 0.8899，F1 0.7969。
- `logistic_balanced` held-out test AUROC 0.9835，AP 0.8094，F1 0.7698。

解释：ToxicChat 上扩展特征和非线性模型带来明显收益，说明单 suffix token 序列不是上限。

### JBB / HarmBench safety risk

JBB intent suffix probe：

- 最好一组 `intent_seed42`：CV F1 0.9146，AUROC 0.9684，AP 0.9681。
- `intent_category_heldout`：F1 0.9036，AUROC 0.9674，AP 0.9738。
- `direct_fulfillment`：F1 0.9100，AUROC 0.9668，AP 0.9722。

JBB contrastive awareness：

- 3 个 contrastive margin 特征，held-out F1 0.9184，AUROC 0.9589，AP 0.9707。

HarmBench cross-dataset：

- JBB calibration mean-margin AUROC 0.9642。
- HarmBench text behaviors detection rate 0.615 at JBB max-TPR@FPR5 threshold。
- 类别差异很大，例如 copyright 几乎检测不到，chemical/biological、cybercrime 等较强。

解释：风险/意图类任务上 suffix signal 很强，但跨数据集泛化受类别定义影响明显。

### Hallucination / factuality

- HaluEval：`unsupported_or_conflicting` held-out AUROC 0.9257，AP 0.9341，TPR@FPR5 0.6913，表现强。
- RAGTruth：`fully_supported_grounded` held-out AUROC 0.8137，AP 0.7364，TPR@FPR5 0.4358，中等。
- BUMP：`careful_pair_consistency` held-out AUROC 0.7485，AP 0.7961，TPR@FPR5 0.4222，中等偏弱。
- FaithBench：`all_summary_claims_verified` held-out AUROC 0.5846，AP 0.6345，TPR@FPR5 0.0345，基本失败；其 oracle F1 接近 all-positive baseline，数据/标签设置可能不适合当前二分类协议。

解释：SMSP 对任务形式高度敏感。HaluEval 的成对/显式上下文设置最适合，FaithBench 的 `Unwanted` 标签和小样本 held-out 不稳定。

## 四、尝试现状

已经尝试过：

- 手写 semantic suffix。
- 多 suffix 族 validation selection。
- contrastive word-pair margin。
- ordered token logprob + MLP。
- rich semantic/token feature + logistic/tree/boosting。
- SafetyBench nested black-box evolutionary/semantic search。
- dynamic candidate-letter suffix。
- 大跨度 suffix 多样性池。
- baseline audit、paired bootstrap、validation/test 隔离。

当前最好证据：

- ToxicChat：rich logprobs + tree/boosting 是目前最明显提升方向。
- SafetyBench：candidate-letter 动态 suffix 在 TPR@FPR5 上有正向迹象，但改善不够稳。
- JBB/HarmBench：风险意图 signal 强，跨 benchmark 类别泛化是主要短板。
- Hallucination：HaluEval 强，RAGTruth/BUMP 可用，FaithBench 暂不适合作为主证明点。

当前风险：

- vLLM prompt logprob 存在批处理数值波动；最终指标应使用 `workers=1` 或预注册重复抽取聚合。
- suffix 搜索容易过拟合 validation，因此必须继续保持 nested protocol。
- 单一模型 Llama-3.1-8B-Instruct 的结论不能直接外推。
- 部分任务的 threshold/oracle 指标是描述性结果，不代表可部署选择协议。

## 五、可能尝试方向

优先级较高：

1. SafetyBench 复现实验：对 `dynamic_zh_correct_option` 与 baseline 做 `workers=1` 双重复抽取，固定聚合规则后重跑 paired bootstrap。
2. SafetyBench 多模型验证：至少加一个同规模 instruct 模型，看 candidate-letter 动态 suffix 是否仍提升 TPR@FPR5。
3. ToxicChat rich feature 迁移：把 rich logprob 特征框架迁移到 SafetyBench/JBB，比较单 suffix vs 多 suffix/token 组合。
4. 类别/题型分层分析：SafetyBench 按 subject/language/选项数，HarmBench 按 semantic_category，找失败模式。
5. 预注册搜索预算：固定候选池、fit/search 问题数、bootstrap seed、finalist 数量，减少迭代选择偏差。

中优先级：

1. 白盒 token search：如果能加载本地权重和 autograd，用 pairwise ranking surrogate 搜索 suffix token，再走同一 nested protocol。
2. suffix ensemble：不只选单 suffix，而是用少量互补 suffix 的 token feature ensemble，并控制搜索自由度。
3. token attribution 稳定性：对最佳 suffix 做 repeated extraction + token permutation，确认信号来自哪些 token。
4. negative/contrastive correctness suffix：同时建模“正确/错误”“支持/冲突”等对照句，避免单句概率尺度漂移。

暂缓：

- 继续扩大同义 correctness paraphrase 池：已有结果显示收益边际很小。
- 以 FaithBench 作为主指标：当前标签和样本规模导致结论弱。

## 六、中间结果清理策略

保留：

- `README.md`、`config*.json`、`manifest.json`、`metrics.json`、`final_metrics.json`、`train_cv_metrics.json`。
- `search_results.json`、`validation_results.json`、`search_ranking.csv`、`baseline_audit.json`。
- `suffix_selection.csv`、`suffixes.json`、`suffix.json`、`token_metrics*.csv`。
- 数据拆分、benchmark 原始数据、候选配置和总结文档。

删除：

- `features*.jsonl` 和 `*/features/*.jsonl`：可再生成的 logprob 特征缓存。
- `matrix*.npz`：由特征缓存派生的矩阵。
- `predictions*.csv`、`final_predictions*.csv`：由模型/评估流程派生的逐样本预测。
- `model.pt`：训练权重缓存。
- `__pycache__/` 和 `*.pyc`。

清理前统计：465 个中间文件，约 610 MiB。
