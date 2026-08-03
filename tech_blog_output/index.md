# vLLM 小课堂技术博客

这里收录从技术视频和配套课件中整理出的深度技术文章。每篇文章均提供中文与英文版本，并保留关键幻灯片、计算过程和工程背景。

[B 站视频合集](https://space.bilibili.com/189708420/lists?sid=8336139){ .md-button .md-button--primary }
[GitHub 仓库](https://github.com/princepride/live-streaming-tools){ .md-button }

## 最新文章

<div class="grid cards" markdown>

-   :material-memory: **Kimi K3 推理后端演进**

    ---

    从 KDA、LatentMoE 到 KV Cache、部分缓存命中与算子优化，理解 2.78T 模型背后的系统重构。

    [:octicons-arrow-right-24: 阅读中文](kimi_k3_vllm/final/blog.md)
    · [English](kimi_k3_vllm/final/blog.en.md)

-   :material-speedometer: **DSpark 投机解码**

    ---

    从隐藏状态提取到 Markov Head、KVConnector 和跨节点传输，拆解投机解码的工程落地路径。

    [:octicons-arrow-right-24: 阅读中文](dspark_speculative_decoding/final/blog.md)
    · [English](dspark_speculative_decoding/final/blog.en.md)

-   :material-chart-timeline-variant-shimmer: **长序列 MoE RL**

    ---

    以 32 张 H100 运行 128K 序列为案例，梳理显存解耦、通信重叠与并行配置收敛方法。

    [:octicons-arrow-right-24: 阅读中文](long_sequence_moe_rl/final/blog.md)
    · [English](long_sequence_moe_rl/final/blog.en.md)

</div>

## 关于本站

文章由本仓库的技术博客流水线生成：先对视频进行带时间戳转写，再分析 PPT/PDF，建立证据与章节结构，最后经过写作、审校和双语排版。网页由 MkDocs Material 构建，并通过 GitHub Actions 自动发布。
