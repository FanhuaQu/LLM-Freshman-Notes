# Tokenizer 与 Embedding

> **一句话直觉**：模型只认识数字，不认识文字。Tokenizer 负责把文本切成"词元(token)"并映射成整数 ID，Embedding 层再把每个 ID 查表变成一个稠密向量——这是整个 LLM 的入口。

## 0. 为什么需要分词？

神经网络只能在连续向量上做运算，所以文本必须先变成数字序列。三种朴素方案都不行：

| 方案 | 做法 | 问题 |
| :--- | :--- | :--- |
| **字符级** (char) | 每个字符一个 ID | 序列太长，语义单元太碎，`"cat"` 被拆成 `c/a/t`，模型要自己重新学会拼词 |
| **词级** (word) | 每个单词一个 ID | 词表爆炸（英语 + 各种变形 + 专有名词）；遇到未登录词 (OOV) 只能 \<UNK\>，无法处理新词 |
| **子词级** (subword) | 常见词整存，罕见词拆成有意义的片段 | ✅ 词表可控、无 OOV、能泛化到新词 |

**子词分词**是当前所有主流 LLM 的选择。核心思想：**高频片段（morpheme 语素级）单独成 token，低频词拆成高频片段的组合**。

例如 `"tokenization"` 可能被切成 `token` + `ization`；`"unhappiness"` → `un` + `happi` + `ness`。

**关键指标**：
- **词表大小 (vocab size)**：主流在 32k ~ 256k 之间（GPT-2 是 50257，Llama-3 是 128256）。
- **压缩率 (fertility)**：平均每个 token 承载多少字符。压缩率越高，同样的文本占用的 token 越少，推理越省。


---

## 1. BPE（Byte Pair Encoding）— 最主流

GPT 系列、Llama 系列都用它（或其字节变体）。

### 1.1 训练算法

从**字符级**词表出发，反复合并最高频的相邻对：

```
输入：语料 + 期望的词表大小 V

1. 把所有词拆成字符，词表 = 所有出现过的字符
2. while 词表大小 < V:
3.     统计所有相邻 token 对的频次
4.     找到频次最高的对 (a, b)
5.     把 (a, b) 合并成新 token "ab"，加入词表
6.     语料中所有 "a b" 相邻处替换为 "ab"
7. 返回：最终词表 + 所有合并规则（按顺序）
```

**示例**：语料 `{low: 5, lower: 2, newest: 6, widest: 3}`（数字是词频）

初始词表（字符级）：`l o w e r n s t w i d`

| 轮次 | 最高频对 | 合并后 |
| :--- | :--- | :--- |
| 1 | `e s` (6+3=9) | `es` |
| 2 | `es t` (9) | `est` |
| 3 | `l o` (5+2=7) | `lo` |
| ... | ... | ... |

合并规则是**有序的**，编码时按顺序应用。

### 1.2 编码（推理时）

给定新文本，**按训练时记录的合并顺序**依次尝试合并：

```python
def encode(word, merges):
    tokens = list(word)          # 先拆成字符
    for a, b in merges:          # 按顺序
        # 在 tokens 中把所有相邻的 (a,b) 合并
        ...
    return [vocab[t] for t in tokens]
```

> ⚠️ 合并顺序很重要。`("l", "o")` 先合并还是 `("o", "w")` 先合并，结果不同。

### 1.3 复杂度

朴素实现编码是 $O(n \cdot |\text{merges}|)$，慢。实际用**优先队列 / 双向链表**优化，或用 **tiktoken**（GPT 的 Rust 实现）这类高度优化库。

---

## 2. 字节级 BPE（BBPE）— GPT-2 起的标准

**问题**：纯 BPE 的词表基于 Unicode 字符，遇到训练语料里没出现过的字符（emoji、罕见汉字、其他语言）依然会 OOV。

**解法**：不基于字符，而是基于**字节（byte）**。任何文本先 UTF-8 编码成字节序列（0~255 共 256 个），再做 BPE。

- **基础词表**永远只需 256 个字节 + 特殊 token。
- **永远不会 OOV**——任何 Unicode 文本都能被字节表示。
- 代价：一个汉字 UTF-8 是 3 字节，英文 emoji 可能 4 字节，所以中文/emoji 的压缩率更差（一个汉字常需 1~3 个 token）。

这就是为什么**中文在 GPT 系列里更费 token**：一个常用汉字约 1~2 token，而一个英文单词可能才 1 token。

---

## 3. WordPiece 与 Unigram — 另外两条路线

### 3.1 WordPiece（BERT 用）

和 BPE 几乎一样，**唯一区别在于"选哪一对合并"的准则**：

- BPE：选**频次最高**的 pair
- WordPiece：选**提升似然最多**的 pair，即最大化

$$
\text{score}(a,b) = \frac{\text{freq}(ab)}{\text{freq}(a)\cdot \text{freq}(b)}
$$

直觉：优先合并**单独出现少、但经常一起出现**的组合，避免把高频词本身拆开。

编码时 WordPiece 用**贪心最长匹配**：从头开始，尽量匹配词表里最长的子词，`##` 前缀表示"这不是词首"（如 `token` 在词中间会写成 `##token`）。

### 3.2 Unigram（T5、ALBERT、SentencePiece 的 unigram 模式）

**自顶向下**思路，和 BPE 相反：

1. 先准备一个**大**的候选词表（所有子串）。
2. 用 EM 算法估计每个子词的概率，构成 unigram 语言模型。
3. 计算删掉某个子词后**总体似然损失**，删掉损失最小的那些。
4. 重复直到词表缩到目标大小。

编码时用 **Viterbi 算法**找概率最大的切分路径（可能有多种切法，选联合概率最高的）。

$$
P(\text{text}) = \prod_{i} P(\text{subword}_i), \quad \text{切分} = \arg\max \prod_i P(s_i)
$$

---

## 4. SentencePiece — 工程封装

Google 出的分词库，**直接吃原始文本**（不预分词、不依赖空格），把空格也当成普通字符（用 `▁` 表示）。

- 支持 BPE 和 Unigram 两种模式。
- 天然适合中文、日文等**没有空格分词**的语言。
- 训练出的模型自带 encode/decode，不用额外维护预分词器。
- Llama 用 SentencePiece（BPE 模式），T5/ALBERT 用 SentencePiece（Unigram 模式）。

> 现代趋势：HuggingFace 的 **`tokenizers`** 库（Rust 实现）已基本统一了这些算法，速度快、并行化好。

---

## 5. Embedding 层：把 ID 变成向量

### 5.1 查表

Token ID $i \in \{0, ..., V-1\}$，Embedding 矩阵 $E \in \mathbb{R}^{V \times d}$：

$$
\mathbf{x}_i = E[i, :] \in \mathbb{R}^{d}
$$

这就是一个**可学习的查表**（lookup table），本质等价于 one-hot 向量乘以 $E$，但实际都用索引查表实现（避免 $V \times d$ 的稠密矩阵乘法）。

### 5.2 权重共享（Weight Tying / Tied Embeddings）

语言模型最后要把隐状态映射回词表分布，输出层是 $W_{out} \in \mathbb{R}^{d \times V}$。**让 $W_{out} = E^\top$**（即输入 embedding 和输出投影共享同一套参数）能：

- 省下 $V \times d$ 参数（$V$ 大时非常可观，如 128k × 4096 ≈ 5 亿参数）。
- 通常还能提升效果（Anthropic 的 "weight tying" 分析指出共享能稳定训练）。

GPT-2、T5 等都用了 weight tying；部分大模型为了训练灵活性不用（如 GPT-3）。

### 5.3 位置信息的注入

**注意**：原始 Transformer 的 attention 本身是**置换不变**的（打乱 token 顺序结果不变），所以位置信息必须额外注入。做法是在 embedding 上加/拼位置编码——这部分详见 [01-04 位置编码]。

---

## 6. 工程细节与踩坑

### 6.1 词表大小的权衡

| 词表偏小 | 词表偏大 |
| :--- | :--- |
| 序列变长 → 推理慢、上下文浪费 | 序列短、压缩率高 |
| embedding 层参数量小 | embedding + 输出层参数大，显存吃紧 |
| 每个 token 训练样本更充分 | 低频 token 训练不足 |

经验：**多语言模型需要更大词表**（Llama-3 用 128k 主要为了多语言和代码）。

### 6.2 特殊 token

| Token | 作用 |
| :--- | :--- |
| `<|endoftext|>` / `</s>` / `<eos>` | 序列结束 |
| `<|im_start|>` / `<|im_end|>` 等 | 对话角色分隔（ChatML 格式） |
| `<pad>` | 补齐到固定长度（训练时 mask 掉，不计 loss） |
| `<|tool_call|>` 等 | 工具调用等结构化能力 |

> ⚠️ **坑**：特殊 token 若在推理时被当作普通文本处理（没被正确识别），会导致行为异常。HuggingFace 里要注意 `add_special_tokens` 参数。

### 6.3 Glitch Tokens（故障 token）

训练语料里**几乎没出现过**、但恰好被分进了词表的 token（如 `<|extra_id_0>` 类残留）。用户输入这些 token 时，模型表现为"卡住/胡言乱语"，因为它的 embedding 基本没被训练过。

- 现象：SolidGoldMagikarp 是一类知名 glitch token。
- 检测：找出 embedding 范数异常、或训练频次为 0 的 token。

### 6.4 上下文长度与 token 计数

- **上下文窗口**以 token 计（如 128k tokens），不是以字符计。
- 中文同样一段话占的 token 数通常是英文的 **1.5~2 倍**（BBPE 下）。
- 计费、KV Cache 显存、速度，全都按 token 算 → 分词器直接影响成本。

---

## 7. 常见误区

1. **"token = 单词"** ❌。一个 token 可能是半个单词、多个字符、甚至一个字节。
2. **"不同模型的 token 可以混用"** ❌。每个模型有自己的词表，token ID 完全不可通用。
3. **"分词是预处理，和模型能力无关"** ❌。分token 边界会影响算术、拼写、字符级任务的表现（模型对 `"straw"` 有几个 r 这种问题容易错，因为 token 切法割裂了字符）。
4. **"中文一个汉字就是一个 token"** ❌。BBPE 下一个汉字常占 1~3 个 token。
5. **"Embedding 层不重要"** ❌。它是参数量占比可观、且承载所有输入语义的入口，weight tying 与否影响训练稳定性。


---
