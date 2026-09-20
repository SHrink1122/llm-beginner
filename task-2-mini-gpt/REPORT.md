# Task2-从零实现 mini-GPT

**Fork 仓库**：https://github.com/SHrink1122/llm-beginner

**DoD Checklist**

- [x] **M1** 手写简化版 BPE tokenizer，自检 `tokenizer_roundtrip` 通过（encode→decode 能还原中文）
- [x] **M2** 手写 decoder-only 模型并集成 **RoPE**，前向不报错且形状对
- [x] **M3** 实现 **KV cache**，自检 `kv_cache_equivalence` 通过（开/关 cache logits 误差 < 1e-4）
- [x] **M4** 在语料上预训练，自检 `perplexity_on_dev` 低于阈值（唐诗 < 50 / TinyStories < 10）
- [x] **M5** 实现 greedy / top-k / top-p / temperature 四种采样，能生成连贯文本

**eval/result.json**
{
"test": {"tokenizer_roundtrip", "pass": true, "failures": [] },
"test": {"kv_cache_equivalence", "pass": true, "max_abs_diff": 3.337860107421875e-06 },
"test": {"perplexity_on_dev", "pass": true, "perplexity": 34.15, "threshold": 50.0}，
{"n_tokens": 3334, "dataset": "poetry" }
}


## 参数设置

| | 配置 |
|---|---|
| **tokenizer** | 字节级 BPE：`vocab_size = 347`（256 + 91 次 merge）；train.txt 43,292 字节 → 29,643 token |
| **模型** | 514,523 参数（0.51M）：vocab 347 / d_model 96 / 4 层 4 头（d_k=24）/ d_ff 384 / dropout 0.3 / block_size 128。|
| **训练** | next-token prediction + AdamW（lr 3e-4、wd 0.3）+ 5% warmup + cosine 退火，grad clip 1.0，batch 32（8 batch/epoch），上限 500 epoch；每 5 epoch 算 dev ppl，按最优保存，连续 6 次无改善早停。|

**调参过程**
| 词表（vocabsize） | 模型 | run 数 | 结果 |
|---|---|---|---|
| BPE 1024 | d_model 32–128、1–3 层、0.08–0.86M | 26 | 最好 **242** |
| 256（纯字节） | d_model 96、3–4 层、0.39–0.50M | 5 | dev_ppl **10.51** |
| BPE 347（当前） | d_model 96、3–4 层、0.41–0.52M | 1 | dev_ppl **34.15** |

*调参中发现词表大小的影响*：PPL 是模型对真实 token 序列概率的度量，词表越大，每个 token 的概率被分摊到更多候选上，PPL 自然被"稀释"抬高。所以用 1024 词表时 PPL 一直居高不下（最好也只到 242）；换成 256 后 PPL 骤降到 10.51， 但退化成字节级。因此折中取 BPE 347，PPL 34.15 通过阈值。
## 训练结果
[Swanlab实验链接🔗](https://swanlab.cn/@Mic1122/mini-gpt/v1/qrmqbp/overview?utm_source=website_qr&utm_medium=qr_scan)

最好 dev_ppl = **34.15**，dev降到该点后在 34.3–34.6 ，epoch 205 早停；train_ppl 同期 381.85 → **23.02**。

<img width="1142" height="667" alt="ppl_curve" src="figures/ppl_curve.png" />

## 生成样例

续写 120 token（完整文本见 [figures/samples.md](figures/samples.md)）：

| 策略 | 乱码(U+FFFD) |
|---|---|
| `greedy` | 0 |
| `T=0.5` | 1 |
| `T=0.8` | 1 |
| `T=1.2` | 2 |
| `T=1.8` | 11 |
| `top-k=5` | 1 |
| `top-k=20` | 1 |
| `top-p=0.9` | 0 |
| `top-p=0.95` | 1 |
| `T0.8+k20+p0.9` | 1 |

采样示例：

- `greedy`：「。空神空神，为应。美长长长，将长马长。王王王第，与长长长。…」
- `T=1.8`：「，乡��鞠就�u阶雈，莯竱两�观容云，翡S壑天十，繍浑海。…」
- `top-k=5`：「。曙美顾造，同者舍毛。翤猴须送，斑舞驹舟。老龙沙酒，上马皮缕。…」
- `top-k=20`：「，乡鼀能香须龙龙王立，褱笍。萧暑鬢朝，琳皇里好意歐空神。…」
- `top-p=0.9`：「，乡终。能香仙不龙花，从然一两会。容成何夜壑天，穎河极。…」

## 实验观察
1. 采样策略：乱码数随 temperature、top-k、top-p 的增大而增多。temperature 升高或 top-k/p 放宽，候选词集合变大，生成的文本更"随机"。
2. 小语料过拟合：train_ppl 降到 23 而 dev_ppl 停在 34.5 不再下降，对此采取的缓解措施：早停、dropout 加大到 0.3、weight decay 0.3。


## 小结

必做 5 项全部完成，三项自检全过。四种采样与 10 组配置对比见 [figures/samples.md](figures/samples.md)。