# RadioactiveFeedBack

使用者直接提供私有偏好规则，由 Judge 返回最终标量分数，再用 GRPO 训练 Student。

[English](README.md) · [评分 API 项目](https://github.com/Kemalau/OneLinetoProtectYourReward) · [精确方法说明](docs/METHOD.md)

Student 自己生成全部候选回答。Judge 在一次调用中给每个候选返回最终分数，训练端根据这些分数更新 Student。Judge 不提供参考答案、解题文本或思维链。

本仓库是根据当前论文和评分 API 整理的新训练实现。它不等同于历史 checkpoint 的原始训练脚本，也尚未复现论文结果表。

## 安装

Python 3.10+。建议使用 GPU，先安装适合当前 CUDA 环境的 PyTorch，再执行：

```bash
git clone https://github.com/Kemalau/RadioactiveFeedBack.git
cd RadioactiveFeedBack
python -m pip install '.[train]'
```

可选 4-bit LoRA：安装 `'.[train,quantized]'`，训练命令增加 `--load-in-4bit`。每次训练使用一个进程和一个 Student 设备；Judge 通过已有 API 调用。

## 要提供什么

| 信息 | 用途 |
| --- | --- |
| Student 模型名称和 revision | 指定要训练的模型与固定版本 |
| 训练 JSONL | Student 要回答的任务 |
| Judge API 地址、模型名和凭证 | 获取最终评分 |
| 自己的私有载体文件 | 直接指定偏好的特征与相反特征；注入规则时必填，k 是载体数量 |
| 审计 JSONL，可选 | 从训练集中删除与审计面板重叠的题目 |

Judge 的部署版本可通过 `--judge-version` 记录。已有 API 若已安装私有评分规则，使用 `--judge-already-protected`，训练端就不需要重复注入。

## 数据格式

训练文件每行一个对象：

```json
{"task_id":"code-001","prompt":"Write a Python function that ..."}
{"task_id":"math-001","prompt":"Solve the following problem: ..."}
```

`task_id` 必须唯一，`prompt` 是题目。可选 `carrier_id` 为该题固定一个载体；它只用于 Judge 评分，不写进 Student 提示词。额外的参考答案字段会忽略。

使用 `--audit-file` 后，程序会删除重叠的 task_id 和按空白归一化后完全相同的题目文本，并去掉训练内部重复题目。剩余题目按 seed 打乱一次，每题只训练一次，末尾不足 50 题的批次也保留。

## 开始训练

先配置评分 API：

```bash
export KEYFLIP_UPSTREAM_URL='https://your-judge.example/v1/chat/completions'
export KEYFLIP_UPSTREAM_MODEL='your-judge-model'
export KEYFLIP_UPSTREAM_API_KEY='your-api-credential'
```

运行：

```bash
rfeedback train \
  --student-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --student-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --train-file ./examples/train.jsonl \
  --audit-file ./examples/audit.jsonl \
  --condition marked \
  --carrier-file ./private/my-carriers.json \
  --output-dir ./runs/marked-seed42
```

增加 `--dry-run` 可先检查完整配置，不加载模型、不调用 API。示例 JSONL 是自建的小型示例，不是论文训练集，不能据此报告水印效果。

API 返回值采用 chat-completions 格式，`message.content` 是 `{"scores":[95,85,...]}`，按输入候选顺序排列。分数数量不符、格式失败、非有限值或越界都会中止，不会把失败评分默认为 0。若 API 不支持 `response_format`，增加 `--no-json-mode`。

## 自定义载体

仓库不内置实验载体或真实 key。将空白 `examples/carriers.template.json` 复制到自己的私有目录后填写，格式与 `OneLinetoProtectYourReward` 相同：

```json
[
  {
    "id":"carrier_1",
    "domain":"",
    "preferred":"",
    "opposite":"",
    "abstain":""
  }
]
```

空白字段由训练者自己填入适用任务、偏好的特征 `preferred`、相反特征 `opposite` 和弃权规则；模板不能直接运行。规则直接写入 Judge 的 system prompt，由 Judge 识别特征、判断质量门槛并给出最终分数。满足门槛时，偏好侧加分、相反侧减分；两者同时出现、无法判断或不适用时不调整。

注入水印时必须提供 `--carrier-file ./private/my-carriers.json`。省略 `--k` 全部启用；文件里定义了 10 个载体就可以写 `--k 10`。每道题仍只分配一个载体，最多加减 5 分。这份私有偏好配置就是水印设定，不再需要字符串 `KEYFLIP_KEY`，也不再用 HMAC 派生方向。

0.3 版本移除了 `--key-env`；旧 `positive`/`negative` 文件需要按实际偏好改写成 `preferred`/`opposite`，旧格式会报错。要反转偏好，交换这两个字段的描述。Student 的提示词中不会加入这些规则。

自动路由按顺序选第一个领域匹配的载体；多个载体领域完全相同时，可细分适用题目或使用每题的 `carrier_id` 固定分配。填好的配置由使用者自己保管，`private/`、`my-carriers.json` 和私有 key 文件已被 Git 忽略。

## 对照组和默认参数

| 条件 | 含义 |
| --- | --- |
| `clean` | 只用普通质量评分；必须接未加水印的 API |
| `marked` | 使用部署者直接指定的私有评分偏好 |
| `sham` | 在独立 decoy 载体上注入同类偏好 |

Sham 需要提供 `--carrier-file ./decoys.json` 和 `--registered-carrier-file ./registered.json`，程序会拒绝重复 ID。载体 ID 不同并不能证明行为特征独立，实验还需匹配覆盖、暴露和偏移强度。Base 是不训练的初始模型。

| 参数 | 默认值 |
| --- | --- |
| 训练轮数 | 1，每题一次 |
| 每次更新题目数 | 50，末尾可不足 50 |
| 每题回答数 G | 16 |
| 优化器、学习率 | AdamW、1e-6 |
| KL 系数 | 0.04 |
| 普通质量分门槛 | 60 |
| 偏移 rho | 0.05，0–100 分制下最多 ±5 分 |
| 生成长度上限 | 2048 tokens |
| Student 采样 | temperature=0.8、top_p=0.95 |
| LoRA | rank=16、alpha=32、关闭 dropout |

Judge 对每个回答独立判定是否符合质量门槛、是否能识别指定载体，再直接输出调整后的分数。不需要近似平分门控，不比较候选分差，不在代码里执行答案评分。每组优势使用样本方差加 `1e-6`，精确公式见 [方法说明](docs/METHOD.md)。

实现用 `s=+1/-1/0` 表示符合偏好、相反、弃权，对应论文公式的 `c_j * phi_j`。论文附录 B 说明实验采用直接登记偏好；此处不额外派生方向。

以上论文核心默认值已对齐；LoRA、采样、梯度裁剪等额外实现选择在方法说明中列出，不能据此宣称复现了历史实验。

## 输出与验证

每次必须使用新的输出目录，包含：

- `manifest.json`：版本、协议哈希、去重统计、预期计数和完成状态。
- `trajectory.jsonl`：实际题目顺序。
- `reward_receipts.jsonl`：最终分数、请求哈希、解析成功记录。
- `rollouts.jsonl`：Student 生成的回答与最终奖励。
- `metrics.jsonl`：更新 loss、KL、梯度范数和计数。
- `adapter/`：最终 LoRA 权重和 tokenizer。

`--lora-rank 0` 改为全参数训练，最终输出到 `model/`。训练目录权限为 `0700`，私有清单会记录实际偏好规则，API 凭证不写入日志。中断任务保留失败状态和已有结果；本版尚未实现自动恢复优化器训练。

```bash
python -m unittest discover -s tests -v
python tests/smoke_train.py
```

smoke 使用本地随机初始化小模型、通用占位偏好和模拟 Judge，不访问外部模型、不产生 API 费用；验证两次 GRPO 更新、非零 LoRA 权重保存及偏好配置记录，不是水印有效性实验。

仓库不包含私有凭证、真实水印 key、训练语料、实验 checkpoint、内部云集群脚本和模型级检测器。真实 Judge 遵循率、Student 水印迁移和误报率需要独立实测。

## 许可证

[MIT](LICENSE)。外部模型和数据集遵循各自许可证。
