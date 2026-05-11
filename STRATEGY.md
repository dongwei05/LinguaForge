# Gemma 4 Good Hackathon — 战略与选题

## 一、比赛核心规则（已核实）

- **截稿**：2026-05-18 23:59 UTC（约剩 11 天）
- **奖金池**：$200,000
  - Main Track：$100K
  - Impact Track：$50K（在 5 大主题中评出）
  - Special Technology Track：$50K（用 Cactus / LiteRT / llama.cpp / Ollama / Unsloth）
- **必须使用** Gemma 4 模型（至少一个）
- **提交内容**：
  1. Kaggle Writeup（详细说明）
  2. **3 分钟视频**（含故事讲述）
  3. Public 代码仓库（GitHub 或 Kaggle Notebook）
  4. Live demo 或可运行原型
  5. 媒体画廊 / 封面图
- **评分**：
  - Impact & Vision **40%**
  - Video Pitch & Storytelling **30%**
  - Technical Depth & Execution **30%**
- **现有团队数**：184（截至 2026-05-07）

## 二、获胜策略分析

40% + 30% = **70% 是"故事 + 影响力"**，技术只占 30%。结论：

> 一个有强烈情感冲击的真问题 + 一个能讲哭人的视频 + 技术能跑就行

不要去死磕 SOTA。要的是**评委看完会记住、会感动、会觉得「这就是 AI for Good 该有的样子」**的项目。

## 三、选题决策

### 候选选题对比

| 选题 | 故事性 | 技术差异化 | 解题难度 | 拥挤度 |
|---|---|---|---|---|
| 医疗健康助手 | 中 | 中 | 中 | **高** |
| 灾害响应 | 中 | 中 | 中 | 中 |
| 教育辅导 Agent | 中 | 低 | 低 | **极高** |
| **濒危语言保护** | **极高** | **高** | 中 | **极低** |
| AI 安全框架 | 低 | 高 | 高 | 中 |

### ✅ 最终选题：**"古韵 GuYun" — Offline AI for Endangered Language Preservation**

**一句话**：用 Gemma 4 抢救正在死去的语言——每两周地球就有一种语言永远消失。

**为什么这个选题能赢**：

1. **情感杀伤力满分**：联合国教科文组织数据，全球 7000 种语言中 40% 濒危。每周末就有一位最后的祖母带走一种语言。视频可以拍"祖母用方言哼摇篮曲，孙子听不懂"——这是天然的催泪弹。

2. **同时命中两个 Track**：
   - Impact Track / Digital Equity & Inclusivity（"打破语言壁垒"）
   - Special Technology Track（用 Ollama / Unsloth 离线部署）
   - 也可以蹭到 Future of Education

3. **技术差异化高**：大部分人会做"用 AI 教英语"，我们反过来做"用 AI 抢救小众语言"。

4. **离线部署天然合理**：濒危语言使用者多在偏远地区，没有稳定网络。Gemma 4 的 "local frontier intelligence" 就是为这场景生的。

5. **多模态有用武之地**：祖辈很多是文盲，靠口述传承。需要 ASR + 图片识别（如苗绣纹样、东巴象形文字）。

6. **Demo 容易做**：用现有的低资源语言数据集（如 Common Voice 中的 Welsh / Hakka，或 ailab-cmu 的 Cherokee 数据）就能做出来，不需要田野调查。

## 四、产品定义

### 名字
- 中文：**古韵 GuYun**
- 英文：**LinguaForge**（参赛主推这个名字，国际化）

### 一句话说明
> An offline, multimodal AI companion powered by Gemma 4 that helps communities preserve, learn, and revive endangered languages — one grandmother's voice at a time.

### 核心功能（3 个，不要贪多）

1. **🎙️ Listen（口述记录）**：祖辈讲故事 → ASR 转写 → Gemma 4 自动生成多模态学习卡片（文字+插图+音标）
2. **📚 Learn（个性化教学）**：Gemma 4 Agent 根据学习者水平动态生成对话练习、文化故事、字典查询
3. **🔁 Revive（社区共建）**：用 Unsloth 在本地数据上微调一个迷你 Gemma 4，使其适应特定语言；模型直接打包到 Ollama，社区共享

### 演示语言
- **首要**：Hakka（客家话，4000 万使用者，但孩子辈快不会说了 — 中国故事很好讲）
- **第二**：Cherokee（北美原住民语言 — 国际故事更好讲）

> 决定：**演示视频用 Cherokee 的故事**（评委是 Google DeepMind 美国团队，Cherokee 故事感染力对他们更强），但**代码 demo 同时跑 Hakka 和 Cherokee**，证明跨语言通用性。

### 技术栈

| 层 | 选型 | 命中 Track |
|---|---|---|
| LLM | **Gemma 4 E4B** (on-device, 4.5B 有效参数, 含 audio/image) + **Gemma 4 26B A4B MoE** (cloud demo, 3.8B 激活) | 必选 |
| Fine-tuning | **Unsloth** | Special Tech ✅ |
| 部署 | **Ollama** + ONNX | Special Tech ✅ |
| ASR | Whisper-large-v3-turbo（仅前端） | – |
| 多模态 | Gemma 4 vision | – |
| 前端 Demo | Gradio + HF Spaces | – |
| RAG | LlamaIndex + 本地 SQLite | – |

技术栈刻意覆盖了 Special Technology Track 的两个工具（Unsloth + Ollama），双保险。

## 五、11 天执行计划

| Day | 日期 | 任务 |
|---|---|---|
| 1 | 5/7 | 战略定稿 ✅ + 项目脚手架 + 数据集获取脚本 |
| 2 | 5/8 | Gemma 4 基础推理 + Unsloth 微调脚本 |
| 3 | 5/9 | RAG + 多模态学习卡片生成 |
| 4 | 5/10 | Agent 教学对话循环 |
| 5 | 5/11 | Gradio UI + HF Spaces 部署 |
| 6 | 5/12 | Ollama 离线打包 + 移动端演示 |
| 7 | 5/13 | 视频脚本定稿 + 素材生成（AI 视频） |
| 8 | 5/14 | 视频剪辑 + 配音 + 字幕 |
| 9 | 5/15 | Writeup 第 1 稿 |
| 10 | 5/16 | 全套打磨 + 测试 |
| 11 | 5/17 | 提交 + Buffer Day |

## 六、风险与应对

| 风险 | 应对 |
|---|---|
| Gemma 4 实际不可用（如未公开发布） | 退路：用 Gemma 3，writeup 中强调"待 Gemma 4 GA 后立刻迁移" |
| 单人时间不够 | 砍掉 Revive 模块（保留 Listen + Learn 即可成立） |
| 模型对低资源语言效果差 | 这恰恰是故事的一部分——展示我们的微调如何把效果从 0 提到可用 |
| 视频制作能力 | 用 Runway / Sora / 即梦 生成镜头，HeyGen 做配音 |

