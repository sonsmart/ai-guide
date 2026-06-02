# 10 - 多模态 AI

> **难度：** ⭐⭐⭐⭐⭐ | **定位：** AI 从"纯文本"进化为"全感官"的核心方向，2026 面试热门考点
>
> **前端类比：** 多模态之于 LLM，就像从纯 HTML 页面升级为支持图片、视频、音频的富媒体应用。传统 LLM 只能处理文本（纯文本节点），多模态模型则像浏览器一样能渲染 `<img>`、`<video>`、`<canvas>` 等各种媒体元素，实现真正的"全感官"交互。

## 本章知识树

```
多模态 AI
├── 10.1 多模态基础（定义、为什么重要）
├── 10.2 CLIP 模型（对比学习、图文对齐）
├── 10.3 Vision-Language 模型（GPT-4o / Claude / Qwen-VL / Gemini）
├── 10.4 多模态 RAG（跨模态检索、Sentence Transformers v5.4）
├── 10.5 视频理解 Agent
├── 10.6 GUI Agent / Computer Use
├── 10.7 多模态应用实战（OCR/图表分析/文档理解）
└── 10.8 端侧多模态部署
```

**章节导航：**
[上一章：09-xxx](./09-xxx.md) | [目录](./README.md) | [下一章：11-xxx](./11-xxx.md)

---

## 10.1 多模态基础

### Q: 什么是多模态 AI？为什么它比纯文本 LLM 重要？

**多模态 AI = 能同时理解和生成多种信息类型（文本、图像、音频、视频）的 AI 系统。它打破了传统 LLM "只能读文字"的局限，让模型拥有了"眼睛"和"耳朵"。**

```
单模态 AI（传统 LLM）：
  输入：纯文本 → LLM → 输出：纯文本
  局限：看不到图片、听不到声音、无法理解视频

多模态 AI：
  输入：文本 + 图片 + 音频 + 视频 → 多模态模型 → 输出：文本 / 图片 / 音频
  能力：看图说话、听音回答、视频分析、文档理解
```

**为什么多模态是 2026 的核心趋势？**

| 维度 | 纯文本 LLM | 多模态 AI |
|------|-----------|----------|
| **信息带宽** | 仅文字（低带宽） | 文字+图像+音频+视频（高带宽） |
| **现实感知** | 无法感知物理世界 | 可以"看"和"听"真实环境 |
| **应用场景** | 聊天、写作、编程 | +文档理解、视频分析、机器人控制、医疗影像 |
| **用户体验** | 必须用文字描述一切 | 拍个照片、说句话就能交互 |
| **前端类比** | 纯文本终端 | 现代浏览器（富媒体渲染） |

**人类获取信息的方式：80%+ 来自视觉。** 纯文本 LLM 相当于一个只会读盲文的人，而多模态 AI 是一个五感俱全的助手。

**多模态的三个层次：**

```
Level 1: 多模态理解（Multimodal Understanding）
  → 接受多种输入，生成文本输出
  → 例：看图回答问题、分析图表
  → 代表：GPT-4o (vision), Claude Sonnet

Level 2: 多模态生成（Multimodal Generation）
  → 生成多种模态的输出
  → 例：文生图、文生视频、文生音频
  → 代表：DALL-E 3, Sora, Veo 2

Level 3: Any-to-Any（全模态交互）
  → 任意模态输入 → 任意模态输出
  → 例：看视频生成解说音频、看图片生成视频
  → 代表：GPT-4o (native), Gemini 2.5
```

**面试话术：**

> "多模态 AI 的核心价值是让模型从'只读文字'进化为'全感官理解'。人类 80% 的信息来自视觉，纯文本 LLM 相当于蒙着眼睛工作。2026 年的趋势是 Any-to-Any 模型——任意模态输入、任意模态输出，像 GPT-4o native 和 Gemini 2.5 已经在朝这个方向发展。对于工程应用来说，多模态打开了文档理解、视频分析、GUI 自动化等全新场景。"

---

## 10.2 CLIP 模型

### Q: 请解释 CLIP 的对比学习机制，以及它如何实现图文对齐？

**CLIP（Contrastive Language-Image Pre-training）是 OpenAI 在 2021 年提出的里程碑模型，通过对比学习将图像和文本映射到同一个向量空间，实现了"图文对齐"。它是现代多模态 AI 的基石。**

```
CLIP 核心思想：

  ┌──────────────┐     ┌──────────────┐
  │  Image       │     │  Text        │
  │  Encoder     │     │  Encoder     │
  │  (ViT/ResNet)│     │  (Transformer)│
  └──────┬───────┘     └──────┬───────┘
         │                     │
         ▼                     ▼
    [image_emb]           [text_emb]
         │                     │
         └────────┬────────────┘
                  │
            Cosine Similarity
                  │
          ┌───────┴───────┐
          │  匹配的图文对   │ → 相似度高 ✓
          │  不匹配的图文对 │ → 相似度低 ✗
          └───────────────┘
```

**对比学习（Contrastive Learning）的训练过程：**

假设一个 batch 有 N 个图文对 `(I₁,T₁), (I₂,T₂), ..., (Iₙ,Tₙ)`：

```
相似度矩阵（N × N）：

              T₁    T₂    T₃    ...   Tₙ
        I₁  [ ✓     ✗     ✗    ...    ✗  ]   ← 只有对角线是正样本
        I₂  [ ✗     ✓     ✗    ...    ✗  ]
        I₃  [ ✗     ✗     ✓    ...    ✗  ]
        ...
        Iₙ  [ ✗     ✗     ✗    ...    ✓  ]

训练目标：最大化对角线的相似度，最小化非对角线的相似度
损失函数：对称的 InfoNCE Loss（图→文 + 文→图 两个方向）
```

**CLIP 训练的伪代码：**

```python
import torch
import torch.nn.functional as F

# CLIP 对比学习核心
def clip_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    image_embeddings: (batch_size, embed_dim) — 图像编码
    text_embeddings:  (batch_size, embed_dim) — 文本编码
    """
    # L2 归一化
    image_emb = F.normalize(image_embeddings, dim=-1)
    text_emb = F.normalize(text_embeddings, dim=-1)

    # 计算相似度矩阵 (N x N)
    logits = (image_emb @ text_emb.T) / temperature

    # 对角线为正样本 → labels = [0, 1, 2, ..., N-1]
    labels = torch.arange(len(logits), device=logits.device)

    # 双向对比损失
    loss_i2t = F.cross_entropy(logits, labels)       # 图 → 文
    loss_t2i = F.cross_entropy(logits.T, labels)     # 文 → 图

    return (loss_i2t + loss_t2i) / 2
```

**CLIP 为什么重要？它开创了三大能力：**

| 能力 | 说明 | 应用 |
|------|------|------|
| **Zero-shot 分类** | 不需要训练，直接用文本描述分类 | "a photo of a cat" vs "a photo of a dog" |
| **跨模态检索** | 用文本搜索图片，或用图片搜索文本 | 电商以图搜图、图片搜索引擎 |
| **多模态嵌入** | 统一的图文向量空间 | 下游任务的基础（DALL-E, Stable Diffusion） |

**CLIP 的局限性和后续演进：**

```
CLIP (2021)
  ├── 局限：只做全局对齐，不理解局部区域
  │
  ├─→ BLIP-2 (2023): 加入 Q-Former 做细粒度图文交互
  ├─→ SigLIP (2023): 用 Sigmoid Loss 替代 Softmax，支持更大 batch
  ├─→ EVA-CLIP (2024): 更强的 ViT encoder，ImageNet 82%+
  └─→ MetaCLIP (2024): 用元数据策展提升数据质量
```

**面试话术：**

> "CLIP 的核心是用对比学习把图像和文本映射到同一个向量空间。训练时一个 batch 有 N 个图文对，构造 N×N 的相似度矩阵，对角线是正样本，其余是负样本，用双向 InfoNCE Loss 训练。它开创了 zero-shot 视觉分类和跨模态检索的范式，也是 DALL-E、Stable Diffusion 等生成模型的基础组件。后续的 SigLIP 用 Sigmoid 替代 Softmax 解决了大 batch 的内存问题，MetaCLIP 则在数据策展上做了改进。"

---

## 10.3 Vision-Language 模型

### Q: 主流 VLM 的架构有哪些？Cross-Attention 和 End-to-End 有什么区别？

**Vision-Language Model（VLM）是能同时理解图像和文本的大模型。当前主流架构可以分为两大流派：Cross-Attention 融合和 End-to-End 原生多模态。**

```
=== 流派一：Cross-Attention 融合（模块化） ===

  Image → [Vision Encoder] → Visual Tokens
                                    ↓
  Text  → [LLM] ←── Cross-Attention 融合 ──→ Output

  特点：Vision Encoder 和 LLM 相对独立
  代表：Flamingo, BLIP-2, Qwen-VL

=== 流派二：End-to-End 原生多模态 ===

  Image → [Patch Embedding] → Tokens ─┐
                                       ├→ [统一 Transformer] → Output
  Text  → [Token Embedding] → Tokens ─┘

  特点：所有模态共享同一个 Transformer
  代表：GPT-4o (native), Gemini 2.5, Fuyu
```

**两种架构的详细对比：**

| 维度 | Cross-Attention 融合 | End-to-End 原生 |
|------|---------------------|-----------------|
| **架构** | Vision Encoder + Connector + LLM | 统一 Transformer |
| **Vision Encoder** | 预训练的 ViT/SigLIP（冻结或微调） | 无独立 Encoder，直接处理 patch |
| **模态融合方式** | 通过 Cross-Attention 或 Projector 注入 | 所有 token 在同一序列中自注意力 |
| **训练方式** | 分阶段（先对齐，再指令微调） | 端到端联合训练 |
| **优势** | 可复用成熟的 Vision Encoder，训练效率高 | 模态融合更深，理解更自然 |
| **劣势** | 融合深度有限，可能丢失细粒度信息 | 训练成本极高，数据要求高 |
| **前端类比** | 微前端（独立子应用通过 API 通信） | 单体应用（所有模块共享运行时） |

**主流 VLM 模型对比（2025-2026）：**

| 模型 | 架构类型 | Vision Encoder | 特点 |
|------|---------|---------------|------|
| **GPT-4o** | End-to-End 原生 | 内置 | Any-to-Any，原生多模态 token |
| **Claude Sonnet/Opus** | Cross-Attention | 内置 ViT | 强文档/图表理解，支持 Computer Use |
| **Gemini 2.5** | End-to-End 原生 | 内置 | 超长上下文（1M tokens），原生音视频 |
| **Qwen-VL 2.5** | Cross-Attention | ViT-bigG | 开源最强之一，支持动态分辨率 |
| **LLaVA-OneVision** | Cross-Attention | SigLIP | 开源社区标杆，单图/多图/视频 |
| **InternVL 2.5** | Cross-Attention | InternViT-6B | 6B 视觉编码器，细粒度理解强 |

**Visual Token 的生成过程（以 Cross-Attention 架构为例）：**

```python
# VLM 图像处理 Pipeline（伪代码）
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# 1. 图像预处理：切分成 patches
# 一张 384x384 的图片，patch_size=14
# → 生成 (384/14)² = 729 个 visual tokens

# 2. Vision Encoder 编码
# ViT 将 729 个 patches 编码为 729 个 visual embeddings
# 每个 embedding 维度 = 1024

# 3. Projector/Connector 对齐
# 将 vision embedding 映射到 LLM 的词嵌入空间
# 可选方案：Linear Projection / MLP / Q-Former / Perceiver Resampler

# 4. 与文本 token 拼接，送入 LLM
# [visual_tokens (729)] + [text_tokens (N)] → LLM → output

# Qwen-VL 2.5 使用示例
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "https://example.com/chart.png"},
        {"type": "text", "text": "请分析这张图表的趋势"}
    ]}
]

inputs = processor.apply_chat_template(messages, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(output[0], skip_special_tokens=True))
```

**动态分辨率（Dynamic Resolution）技术：**

```
传统方式：所有图片 resize 到固定 224×224 → 信息损失严重

动态分辨率（Qwen-VL 2.5 / InternVL）：
  ┌──────────────────────────────┐
  │  原图 1920×1080              │
  │                              │
  │  → 按比例切分成多个 448×448   │
  │    的子图 + 1 个缩略图       │
  │                              │
  │  ┌────┬────┬────┬────┐      │
  │  │ P1 │ P2 │ P3 │ P4 │      │
  │  ├────┼────┼────┼────┤      │
  │  │ P5 │ P6 │ P7 │ P8 │      │
  │  └────┴────┴────┴────┘      │
  │  + [Thumbnail 全局缩略图]    │
  │                              │
  │  每个子图独立编码 → 拼接     │
  │  → 保留高分辨率细节          │
  └──────────────────────────────┘
```

**面试话术：**

> "当前 VLM 有两大架构流派。Cross-Attention 融合是模块化设计——用预训练的 ViT 编码图像，通过 Projector 把 visual tokens 注入 LLM，代表有 Qwen-VL 和 LLaVA。End-to-End 原生多模态则是把所有模态的 token 放进同一个 Transformer 联合训练，代表是 GPT-4o 和 Gemini。前者工程灵活、训练高效；后者融合更深、理解更自然，但训练成本极高。动态分辨率是当前开源 VLM 的标配技术，把高分辨率图片切分成多个子图分别编码，避免了 resize 导致的信息损失。"

---

## 10.4 多模态 RAG

### Q: 多模态 RAG 和传统文本 RAG 有什么区别？如何实现跨模态检索？

**多模态 RAG = 检索对象从"纯文本"扩展到"图片、表格、图表、PDF 页面"等多种模态。核心挑战是如何在不同模态之间进行跨模态检索和融合。**

```
=== 传统文本 RAG ===

  Query(文本) → 文本向量检索 → 文本 Chunks → LLM → 回答(文本)

=== 多模态 RAG ===

  Query(文本/图片) → 跨模态向量检索 → 文本 + 图片 + 表格 → VLM → 回答
                          ↑
                  统一的多模态嵌入空间
```

**多模态 RAG 的三种实现策略：**

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| **文本化（Text-first）** | 用 OCR/VLM 将图片转为文本描述，然后做纯文本 RAG | 简单，复用现有 RAG 基础设施 | 视觉信息损失大，转换可能出错 |
| **多模态嵌入（Unified Embedding）** | 用统一的多模态 embedding 模型编码所有模态 | 保留原始视觉信息，跨模态检索 | 需要多模态 embedding 模型 |
| **ColPali / Late Interaction** | 直接对 PDF 页面截图生成多向量，用 MaxSim 检索 | 跳过 OCR/解析，端到端 | 索引体积大，计算成本高 |

**Sentence Transformers v5.4 统一 API：**

Sentence Transformers v5.4 提供了统一的多模态嵌入接口，支持文本、图像混合编码：

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# v5.4 统一 API：文本和图像用同一个模型编码
model = SentenceTransformer("jinaai/jina-clip-v2")

# 文本嵌入
text_embeddings = model.encode([
    "一只橘猫坐在窗台上",
    "2024年Q3财报显示营收增长15%"
])

# 图像嵌入 — 同一个 encode 方法
image_embeddings = model.encode([
    Image.open("cat_photo.jpg"),
    Image.open("financial_chart.png")
])

# 跨模态相似度计算（文本查询 → 图像检索）
from sentence_transformers.util import cos_sim
similarities = cos_sim(text_embeddings, image_embeddings)
print(similarities)
# tensor([[0.82, 0.11],   ← "橘猫"和猫照片相似度高
#         [0.05, 0.76]])  ← "财报"和图表相似度高
```

**ColPali：跳过 OCR 的端到端文档检索：**

```
传统文档 RAG（复杂 pipeline）：
  PDF → OCR/解析 → 提取文字+表格 → 分块 → Embedding → 检索
  问题：解析错误、布局信息丢失、表格解析困难

ColPali（端到端）：
  PDF 页面 → 截图 → VLM 生成多向量 → MaxSim 检索
  优点：零解析、保留完整视觉布局

  ┌──────────────────────────────┐
  │  PDF Page Screenshot         │
  │  ┌───┬───┬───┬───┐          │
  │  │ p1│ p2│ p3│ p4│ ← 每个   │
  │  ├───┼───┼───┼───┤   patch  │
  │  │ p5│ p6│ p7│ p8│   生成一 │
  │  ├───┼───┼───┼───┤   个向量 │
  │  │ p9│p10│p11│p12│          │
  │  └───┴───┴───┴───┘          │
  │                              │
  │  → 12 个 patch embeddings   │
  │  → MaxSim 匹配 query tokens │
  └──────────────────────────────┘
```

```python
# ColPali 风格的多模态检索（伪代码）
from colpali_engine.models import ColPali, ColPaliProcessor

model = ColPali.from_pretrained("vidore/colpali-v1.3")
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")

# 索引：对每个 PDF 页面截图生成多向量
page_images = [Image.open(f"page_{i}.png") for i in range(100)]
page_inputs = processor(images=page_images)
page_embeddings = model(**page_inputs)  # shape: (100, num_patches, dim)

# 检索：用文本 query 的 token embeddings 与 page patches 做 MaxSim
query_inputs = processor(text=["Q3营收同比增长多少？"])
query_embeddings = model(**query_inputs)  # shape: (1, num_tokens, dim)

# MaxSim：每个 query token 取与所有 patches 的最大相似度，再求和
# score(q, p) = Σᵢ maxⱼ sim(qᵢ, pⱼ)
scores = compute_maxsim(query_embeddings, page_embeddings)
top_pages = scores.argsort(descending=True)[:5]
```

**面试话术：**

> "多模态 RAG 的核心挑战是跨模态检索。有三种主流策略：第一种是 text-first，把图片转文字后走传统 RAG，简单但信息损失大；第二种用统一的多模态 embedding（如 Jina-CLIP-v2）编码所有模态到同一空间，Sentence Transformers v5.4 提供了开箱即用的统一 API；第三种是 ColPali 的端到端方案，直接对 PDF 页面截图生成 patch-level 多向量，用 MaxSim 做 late interaction 检索，跳过了复杂的文档解析。实际项目中我会根据文档类型选择：纯文本为主用第一种，图表密集用第三种。"

---

## 10.5 视频理解 Agent

### Q: 如何构建一个视频理解 Agent？帧采样和场景检测的策略是什么？

**视频理解是多模态 AI 中最复杂的场景之一——一段 10 分钟的视频可能有 18000 帧（30fps），直接全部送入模型既不经济也不现实。核心挑战是：如何用最少的帧捕获最多的信息。**

```
视频理解 Agent 架构：

  Video Input
      │
      ▼
  ┌────────────────────┐
  │  帧采样策略         │  ← 从 18000 帧中选出 20-50 帧
  │  (Frame Sampling)  │
  └────────┬───────────┘
           │
           ▼
  ┌────────────────────┐
  │  场景检测           │  ← 识别场景切换边界
  │  (Scene Detection) │
  └────────┬───────────┘
           │
           ▼
  ┌────────────────────┐
  │  VLM 分析          │  ← 对关键帧进行视觉理解
  │  (GPT-4o/Gemini)   │
  └────────┬───────────┘
           │
           ▼
  ┌────────────────────┐
  │  时序推理 + 总结    │  ← 结合时间线信息生成回答
  │  (Temporal Reason) │
  └────────────────────┘
```

**帧采样策略对比：**

| 策略 | 方法 | 适用场景 | 帧数 |
|------|------|---------|------|
| **均匀采样** | 等间隔取帧（如每 10 秒取 1 帧） | 长视频概览 | 固定 |
| **场景变化采样** | 检测画面突变，在变化点取帧 | 电影/会议录像 | 动态 |
| **关键帧聚类** | 对所有帧做 embedding，K-Means 聚类取中心帧 | 内容多样的视频 | 动态 |
| **自适应密度采样** | 运动剧烈处密集采样，静止处稀疏采样 | 监控/体育视频 | 动态 |

**场景检测 + 关键帧提取实现：**

```python
from scenedetect import SceneManager, open_video, ContentDetector
from PIL import Image
import cv2

def extract_keyframes(video_path: str, threshold: float = 27.0):
    """基于内容变化的场景检测 + 关键帧提取"""

    # 1. 场景检测
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()

    # 2. 每个场景取中间帧作为关键帧
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes = []

    for scene in scenes:
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        mid_frame = (start_frame + end_frame) // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if ret:
            keyframes.append({
                "frame_idx": mid_frame,
                "timestamp": mid_frame / fps,
                "image": Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                "scene_duration": (end_frame - start_frame) / fps
            })

    cap.release()
    return keyframes

# 3. 送入 VLM 分析
def analyze_video_with_vlm(keyframes, query: str):
    """用 VLM 分析关键帧，回答用户问题"""
    from openai import OpenAI
    client = OpenAI()

    # 构建带时间戳的多帧消息
    content = [{"type": "text", "text": f"以下是视频的关键帧（按时间顺序）。请回答：{query}"}]

    for kf in keyframes:
        content.append({"type": "text", "text": f"[{kf['timestamp']:.1f}s]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(kf['image'])}"}
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=1024
    )
    return response.choices[0].message.content
```

**Gemini 的原生视频理解：**

```python
# Gemini 2.5 支持直接上传视频（无需手动帧采样）
import google.generativeai as genai

model = genai.GenerativeModel("gemini-2.5-pro")

# 直接上传视频文件
video_file = genai.upload_file("meeting_recording.mp4")

# Gemini 内部处理帧采样和时序理解
response = model.generate_content([
    video_file,
    "请总结这段会议的要点，列出每个议题的讨论结论"
])
print(response.text)
```

**视频理解 Agent 的完整工作流：**

```
用户：这段视频里哪个时间点出现了产品演示？

Agent 执行流程：
  1. Scene Detection → 检测到 15 个场景
  2. 关键帧提取 → 15 帧关键图像
  3. VLM 分析每帧 → "0:32 开始出现产品界面展示"
  4. 精细化采样 → 在 0:30-0:45 区间加密采样
  5. 最终回答 → "产品演示从 0:32 开始，持续到 2:15，
                  展示了三个功能：登录流程、数据看板、报表导出"
```

**面试话术：**

> "视频理解的核心挑战是帧采样——一段 10 分钟视频有 18000 帧，不可能全部送入模型。我的策略是先用 PySceneDetect 做场景检测，找到内容变化的边界点，在每个场景取代表帧，这样 15 个场景只需要 15 帧就能覆盖全部内容。如果需要更精细的理解，可以对感兴趣的区间做加密采样。Gemini 2.5 支持原生视频输入，内部处理帧采样，但对于本地部署场景，还是需要自己实现采样 pipeline。"

---

## 10.6 GUI Agent / Computer Use

### Q: 什么是 GUI Agent？请介绍 Anthropic Computer Use 和 Mobile-Agent 的原理。

**GUI Agent = 能够像人一样操作电脑/手机图形界面的 AI Agent。它通过"看"屏幕截图来理解界面，然后执行点击、输入、滚动等操作，实现 RPA（机器人流程自动化）的智能化。**

```
GUI Agent 工作流程：

  ┌────────────────┐
  │  截取屏幕截图   │
  └───────┬────────┘
          │
          ▼
  ┌────────────────┐
  │  VLM 分析界面   │  ← "看"屏幕，理解 UI 元素
  │  理解当前状态   │
  └───────┬────────┘
          │
          ▼
  ┌────────────────┐
  │  规划下一步操作  │  ← 决定点击哪里、输入什么
  └───────┬────────┘
          │
          ▼
  ┌────────────────┐
  │  执行操作       │  ← mouse_click(x, y) / type("text")
  └───────┬────────┘
          │
          ▼
  ┌────────────────┐
  │  验证结果       │  ← 截图验证操作是否成功
  │  循环或结束     │
  └────────────────┘
```

**Anthropic Computer Use 架构：**

Anthropic Computer Use 是 Claude 的原生能力，让 Claude 直接操控计算机完成复杂任务：

```python
import anthropic

client = anthropic.Anthropic()

# Computer Use 工具定义
tools = [
    {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080,
        "display_number": 1
    },
    {
        "type": "text_editor_20250124",
        "name": "str_replace_editor"
    },
    {
        "type": "bash_20250124",
        "name": "bash"
    }
]

# Claude 自主控制电脑
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    tools=tools,
    messages=[{
        "role": "user",
        "content": "打开 Chrome，搜索今天的天气，截图给我"
    }]
)

# Claude 会生成一系列操作：
# 1. screenshot → 看到桌面
# 2. mouse_click(x=48, y=750) → 点击 Chrome 图标
# 3. screenshot → 确认 Chrome 打开
# 4. type("今天天气") → 在搜索栏输入
# 5. key("Return") → 回车搜索
# 6. screenshot → 返回结果截图
```

**Computer Use vs 传统 RPA：**

| 维度 | 传统 RPA (UiPath/Blue Prism) | GUI Agent (Computer Use) |
|------|---------------------------|-------------------------|
| **定位方式** | DOM/Accessibility Tree/CSS 选择器 | 视觉识别（截图 + VLM） |
| **适应性** | UI 变了就失效，需重新录制 | 通过视觉理解适应 UI 变化 |
| **编程方式** | 预定义流程图，每步硬编码 | 自然语言描述目标，Agent 自主规划 |
| **错误处理** | 预定义异常分支 | VLM 观察 + 推理，动态调整 |
| **开发门槛** | 需要专业 RPA 开发人员 | 自然语言描述即可 |
| **前端类比** | E2E 测试脚本（硬编码选择器） | AI 驱动的自动化测试 |

**Mobile-Agent：手机端 GUI Agent：**

```
Mobile-Agent 架构：

  手机屏幕截图
       │
       ├── Visual Perception Module
       │   ├── UI 元素检测（按钮、输入框、图标）
       │   ├── OCR 文字识别
       │   └── 图标语义理解
       │
       ├── Planning Module
       │   ├── 理解用户意图
       │   ├── 拆解为 UI 操作序列
       │   └── 处理多 App 跳转
       │
       └── Action Module
           ├── tap(x, y) — 点击
           ├── swipe(x1, y1, x2, y2) — 滑动
           ├── long_press(x, y) — 长按
           └── type_text("content") — 输入

典型任务：
  "帮我在美团上订一份麦当劳外卖"
  → 打开美团 → 搜索麦当劳 → 选择套餐 → 加入购物车 → 确认订单
```

**GUI Agent 的核心挑战：**

```
1. 坐标精度：VLM 生成的坐标必须精确到像素级
   → 解决：Set-of-Mark（SoM）标注 — 在截图上给每个 UI 元素编号

2. 长流程可靠性：一个任务可能需要 20+ 步操作
   → 解决：每步截图验证 + 错误检测 + 自动回退

3. 延迟：每步都要 VLM 推理，端到端延迟高
   → 解决：小模型初筛 + 大模型决策的级联架构

4. 安全性：Agent 有完整的系统控制权
   → 解决：沙箱环境 + 操作白名单 + 人工确认关键操作
```

**面试话术：**

> "GUI Agent 的核心是用 VLM'看'屏幕截图来理解界面，然后执行点击、输入等操作。Anthropic Computer Use 是目前最成熟的方案，Claude 通过截图-分析-操作的循环来自主控制电脑。相比传统 RPA 靠 DOM 选择器硬编码，GUI Agent 通过视觉理解 UI，即使界面改版也能适应。核心挑战是坐标精度和长流程可靠性——SoM（Set-of-Mark）标注技术通过在截图上给元素编号来提升精度，每步截图验证则保证了操作的可靠性。"

---

## 10.7 多模态应用实战

### Q: 多模态 AI 在 OCR、图表分析、文档理解方面有哪些实际应用？

**多模态 AI 正在取代传统的规则化文档处理方案。过去需要 OCR 引擎 + 规则解析 + 后处理的复杂 pipeline，现在一个 VLM 就能端到端完成。**

**传统方案 vs VLM 方案对比：**

| 任务 | 传统方案 | VLM 方案 | 优势 |
|------|---------|---------|------|
| **OCR** | Tesseract/PaddleOCR → 后处理 | VLM 直接看图输出文本 | 处理复杂排版、手写体 |
| **表格提取** | Camelot/Tabula 规则解析 | VLM 直接理解表格语义 | 处理合并单元格、嵌套表格 |
| **图表分析** | 人工标注 + 规则 | VLM 直接读懂图表趋势 | 理解柱状图/折线图/饼图含义 |
| **发票识别** | 模板匹配 + OCR + 正则 | VLM 端到端提取字段 | 适应不同格式发票 |
| **合同审查** | NLP 实体提取 + 规则 | VLM 看 PDF 截图直接审查 | 保留原始排版信息 |

**OCR + 文档理解实战：**

```python
import anthropic
import base64

client = anthropic.Anthropic()

def analyze_document(image_path: str, task: str) -> str:
    """通用文档分析函数"""
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data
                    }
                },
                {"type": "text", "text": task}
            ]
        }]
    )
    return response.content[0].text

# 场景1：发票信息提取
invoice_result = analyze_document(
    "invoice.png",
    """请从这张发票中提取以下信息，以 JSON 格式返回：
    - 发票号码
    - 开票日期
    - 购买方名称
    - 销售方名称
    - 金额（不含税）
    - 税额
    - 总金额"""
)

# 场景2：图表趋势分析
chart_result = analyze_document(
    "revenue_chart.png",
    """请分析这张图表：
    1. 图表类型是什么？
    2. 主要趋势是什么？
    3. 有哪些值得关注的异常点？
    4. 给出数据洞察和建议"""
)

# 场景3：合同关键条款提取
contract_result = analyze_document(
    "contract_page.png",
    """请从这份合同中提取以下关键信息：
    1. 合同双方
    2. 合同期限
    3. 付款条款
    4. 违约责任条款
    5. 是否有竞业限制条款？"""
)
```

**多模态文档处理 Pipeline（生产级）：**

```
┌──────────────────────────────────────────────────────┐
│              生产级文档理解 Pipeline                    │
│                                                      │
│  1. 文档分类                                          │
│     ┌───────┐  VLM 快速分类                          │
│     │ 输入  │ → 发票/合同/报表/简历/其他              │
│     └───────┘                                        │
│         │                                            │
│  2. 预处理                                           │
│     ├── PDF → 分页截图 (300 DPI)                     │
│     ├── 图片 → 去噪/矫正/增强                        │
│     └── 扫描件 → 倾斜校正                             │
│         │                                            │
│  3. 信息提取（根据文档类型选择 prompt）                 │
│     ├── 结构化提取 → JSON Schema 约束输出              │
│     ├── 表格识别 → Markdown 表格                      │
│     └── 自由文本 → 段落级理解                         │
│         │                                            │
│  4. 后处理 & 校验                                     │
│     ├── Schema 校验（Pydantic）                       │
│     ├── 业务规则校验（金额校验、日期格式）              │
│     └── 置信度低 → 人工审核队列                       │
│         │                                            │
│  5. 存储                                             │
│     └── 结构化数据入库 + 原始文档归档                  │
└──────────────────────────────────────────────────────┘
```

**面试话术：**

> "多模态 AI 在文档理解领域的最大价值是把传统的 OCR + 规则解析 + 后处理的三段式 pipeline 简化为 VLM 端到端处理。比如发票识别，过去需要模板匹配、OCR、正则提取三步，现在一个 Claude Vision 调用就能输出结构化 JSON。在生产环境中，我会加上 Pydantic Schema 校验和业务规则检查，置信度低的结果走人工审核队列。图表分析是另一个高价值场景——VLM 不仅能识别图表中的数字，还能理解趋势和给出洞察，这是传统 OCR 完全做不到的。"

---

## 10.8 端侧多模态部署

### Q: 如何在端侧（手机/边缘设备）部署多模态模型？Gemma 4 等端侧模型有什么特点？

**端侧多模态部署 = 把 VLM 跑在手机、笔记本或边缘设备上，无需依赖云端 API。核心挑战是在有限的算力和内存下保持可用的推理速度和质量。**

```
云端 vs 端侧部署对比：

  云端部署：                      端侧部署：
  ┌──────────┐                   ┌──────────┐
  │ 用户设备  │ ──── 网络 ────→  │ 用户设备  │
  │          │ ← 延迟 200ms+ →  │ [模型]   │
  │          │                   │ 延迟 <50ms│
  └──────────┘                   └──────────┘
       │                              │
  ┌────▼─────┐                   优势：
  │ 云端 GPU  │                   ✓ 隐私保护（数据不出设备）
  │ A100/H100│                   ✓ 低延迟（无网络往返）
  │ 大模型   │                   ✓ 离线可用
  └──────────┘                   ✓ 无 API 费用
```

**端侧多模态模型对比（2025-2026）：**

| 模型 | 参数量 | 多模态能力 | 目标设备 | 特点 |
|------|--------|-----------|---------|------|
| **Gemma 4** | 4B / 12B / 27B | 图像+文本 | 手机/笔记本 | Google 端侧旗舰，ShieldGemma 安全 |
| **Phi-4-multimodal** | 5.6B | 图像+音频+文本 | 笔记本/边缘 | 微软，MoE routing 多模态 |
| **SmolVLM 2** | 256M / 500M / 2.2B | 图像+视频+文本 | 手机 | HuggingFace，超轻量 |
| **MiniCPM-V** | 2B / 8B | 图像+文本 | 手机/平板 | 面壁智能，支持动态分辨率 |
| **Qwen2.5-VL** | 3B / 7B | 图像+视频+文本 | 笔记本/边缘 | 阿里，3B 版本可端侧部署 |

**端侧部署的关键优化技术：**

```
1. 量化（Quantization）
   FP16 → INT8 → INT4  模型体积减半到1/4
   ┌─────────────────────────────────────┐
   │  模型精度   │ 体积   │ 速度  │ 质量  │
   │  FP16      │ 14GB  │ 基准  │ 100% │
   │  INT8      │ 7GB   │ 1.5x │ ~99% │
   │  INT4      │ 3.5GB │ 2x+  │ ~95% │
   │  GGUF Q4   │ 4GB   │ 2x   │ ~96% │
   └─────────────────────────────────────┘

2. KV Cache 优化
   → Grouped Query Attention (GQA)
   → Paged Attention
   → 滑动窗口 Attention

3. 推理框架
   → llama.cpp (GGUF 格式，CPU/GPU 混合)
   → MLC-LLM (跨平台编译)
   → MediaPipe (Android/iOS 官方)
   → Core ML (Apple 设备)
```

**端侧部署实战（以 llama.cpp 为例）：**

```python
# 使用 llama-cpp-python 在本地运行多模态模型
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler

# 加载量化后的多模态模型
chat_handler = MiniCPMv26ChatHandler(
    clip_model_path="minicpm-v-2.6-mmproj-f16.gguf"
)

llm = Llama(
    model_path="minicpm-v-2.6-Q4_K_M.gguf",  # 4-bit 量化，~4GB
    chat_handler=chat_handler,
    n_ctx=4096,
    n_gpu_layers=-1  # 尽可能使用 GPU
)

# 本地多模态推理
response = llm.create_chat_completion(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "这张图片里有什么？"},
            {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}}
        ]
    }]
)
print(response["choices"][0]["message"]["content"])
```

**Apple 设备部署方案：**

```swift
// 使用 Core ML 在 iPhone/Mac 上运行 VLM
import CoreML

// 1. 将模型转换为 Core ML 格式
// python: ct.convert(model, convert_to="mlprogram")

// 2. 加载模型
let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine  // 使用 ANE 加速
let model = try VLMModel(configuration: config)

// 3. Apple Intelligence 集成
// iOS 18+ 的 App Intents 框架支持直接调用本地模型
```

**端侧部署的性能基准：**

```
设备：iPhone 16 Pro (A18 Pro, 8GB RAM)

┌──────────────────────────────────────────────┐
│  模型              │ 量化  │ 首 token │ 吞吐  │
│  SmolVLM-500M     │ INT4 │ 0.3s    │ 30t/s │
│  MiniCPM-V 2B     │ INT4 │ 0.8s    │ 15t/s │
│  Gemma-4 4B       │ INT4 │ 1.2s    │ 10t/s │
│  Qwen2.5-VL 3B    │ INT4 │ 1.0s    │ 12t/s │
└──────────────────────────────────────────────┘

设备：MacBook Pro M4 Max (128GB RAM)

┌──────────────────────────────────────────────┐
│  模型              │ 量化  │ 首 token │ 吞吐  │
│  Qwen2.5-VL 7B    │ INT4 │ 0.5s    │ 35t/s │
│  Gemma-4 12B      │ INT4 │ 0.8s    │ 25t/s │
│  Gemma-4 27B      │ INT8 │ 1.5s    │ 15t/s │
│  InternVL2.5 8B   │ INT4 │ 0.6s    │ 30t/s │
└──────────────────────────────────────────────┘
```

**端侧 vs 云端选型决策：**

```
选端侧部署：
  ✓ 隐私敏感数据（医疗影像、个人照片）
  ✓ 需要离线工作（工厂质检、野外作业）
  ✓ 延迟敏感（实时 AR/VR、驾驶辅助）
  ✓ 高频调用（每秒多次推理，API 费用过高）

选云端部署：
  ✓ 需要最强模型能力（GPT-4o、Claude Opus）
  ✓ 偶发调用（用量不大，API 更经济）
  ✓ 需要长上下文（100K+ tokens）
  ✓ 多模态生成（文生图/视频，算力需求大）

混合方案（推荐）：
  端侧小模型做初筛/预处理 → 复杂任务上传云端大模型
  例：手机本地 OCR 预识别 → 难以识别的部分调 API
```

**面试话术：**

> "端侧多模态部署的核心是在有限算力下平衡速度和质量。目前 4B 以下的 VLM 在手机端已经可用——比如 SmolVLM 500M 在 iPhone 上首 token 延迟只有 300ms。关键技术是 INT4 量化配合 GGUF 格式，llama.cpp 框架在 CPU/GPU 混合推理上做了很好的优化。实际项目中我推荐混合方案：端侧小模型做实时预处理和简单识别，复杂任务再调云端大模型。Gemma 4 是 Google 专门为端侧设计的，4B 到 27B 的梯度覆盖了从手机到笔记本的全场景。"

---

## 章节总结

```
多模态 AI 核心知识图谱：

  ┌─────────────────────────────────────────────────────┐
  │                   多模态 AI 全景                      │
  │                                                     │
  │  基础层                                              │
  │  ├── CLIP 对比学习 → 统一图文嵌入空间                  │
  │  ├── VLM 架构 → Cross-Attention vs End-to-End        │
  │  └── 动态分辨率 → 高清图像处理                        │
  │                                                     │
  │  能力层                                              │
  │  ├── 多模态 RAG → ColPali / Sentence Transformers    │
  │  ├── 视频理解 → 帧采样 + 场景检测 + 时序推理           │
  │  └── GUI Agent → Computer Use / Mobile-Agent         │
  │                                                     │
  │  应用层                                              │
  │  ├── 文档理解 → OCR / 图表分析 / 合同审查              │
  │  └── 端侧部署 → Gemma 4 / SmolVLM / INT4 量化        │
  └─────────────────────────────────────────────────────┘
```

**面试高频考点 Top 5：**

| 排名 | 考点 | 关键答题点 |
|------|------|-----------|
| 1 | CLIP 对比学习 | InfoNCE Loss、N×N 相似度矩阵、双向优化 |
| 2 | VLM 架构 | Cross-Attention vs End-to-End、Visual Token 生成 |
| 3 | 多模态 RAG | 三种策略、ColPali MaxSim、Sentence Transformers 统一 API |
| 4 | GUI Agent | Computer Use 循环、SoM 标注、vs 传统 RPA |
| 5 | 端侧部署 | INT4 量化、模型选型、混合方案 |

---

**章节导航：**
[上一章：09-xxx](./09-xxx.md) | [目录](./README.md) | [下一章：11-xxx](./11-xxx.md)
