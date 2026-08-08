<section class="vllm-hero">
  <div class="vllm-hero__copy">
    <p class="vllm-eyebrow">vLLM SYSTEMS NOTES</p>
    <h1><span class="vllm-title-primary">把每一次技术分享</span><br><span class="vllm-title-gradient">整理成可检索的系统知识</span></h1>
    <p class="vllm-lead">从视频与课件出发，沿着模型架构、内存管理、调度策略和 Kernel 优化的因果链，沉淀面向 AI 系统工程师的中英文深度文章。</p>
    <div class="vllm-actions">
      <a class="md-button md-button--primary" href="https://space.bilibili.com/189708420/lists?sid=8336139">观看 B 站合集</a>
      <a class="md-button" href="https://drive.google.com/drive/folders/1YQ3C025p5DzNPLaxa_dQ10P7hnfcqcp8">浏览 Google Drive 课件</a>
    </div>
  </div>
  <div class="vllm-mascot" aria-label="两位 vLLM 看板娘一起学习">
    <img src="assets/branding/vllm-mascots-study.png" alt="银发与黑橙发的两位 vLLM 看板娘一起阅读技术资料并讨论电脑内容">
  </div>
</section>

<div class="vllm-topics">
  <span>PagedAttention</span><span>Continuous Batching</span><span>KV Cache</span><span>Speculative Decoding</span><span>MoE Serving</span>
</div>

## 最新文章

<div class="grid cards" markdown>

-   <span class="article-kicker">MULTIMODAL GENERATION · SERVING</span>

    **MiniMax-H3 与 vLLM-Omni 音视频联合生成**

    ---

    从 118 GB 权重、58 758 token 长序列与 50 步去噪的三重压力出发，拆解共享 DiT 的打包序列、CPU 卸载与 DLO、Ulysses 与分块 VAE、跨步缓存与步级调度。

    [阅读中文](minimax_h3_vllm_omni/final/blog.md)
    · [English](minimax_h3_vllm_omni/final/blog.en.md)

-   <span class="article-kicker">ARCHITECTURE · MEMORY</span>

    **Kimi K3 day-0支持背后的技术细节**

    ---

    从 KDA、LatentMoE 到 KV Cache、部分缓存命中与算子优化，理解 2.78T 模型背后的系统重构。

    [阅读中文](kimi_k3_vllm/final/blog.md)
    · [English](kimi_k3_vllm/final/blog.en.md)

-   <span class="article-kicker">SPECULATIVE DECODING</span>

    **DSpark 投机解码**

    ---

    从隐藏状态提取到 Markov Head、KVConnector 和跨节点传输，拆解投机解码的工程落地路径。

    [阅读中文](dspark_speculative_decoding/final/blog.md)
    · [English](dspark_speculative_decoding/final/blog.en.md)

-   <span class="article-kicker">KV CACHE · DISTRIBUTED INFERENCE</span>

    **解构 vLLM KV Connector**

    ---

    从 v0 到 v1 的调度解耦出发，拆解逐层传输、请求级异步、L2 预取，以及 LMCache 与 Mooncake 的零拷贝全局池化实践。

    [阅读中文](vllm_kv_connector/final/blog.md)
    · [English](vllm_kv_connector/final/blog.en.md)

-   <span class="article-kicker">TRAINING SYSTEMS</span>

    **长序列 MoE RL**

    ---

    以 32 张 H100 运行 128K 序列为案例，梳理显存解耦、通信重叠与并行配置收敛方法。

    [阅读中文](long_sequence_moe_rl/final/blog.md)
    · [English](long_sequence_moe_rl/final/blog.en.md)

-   <span class="article-kicker">RL · ROLLOUT ENGINE</span>

    **当 vLLM 遇见 slime**

    ---

    从训练与推理的权重同步出发，拆解 slime 如何借助 vLLM 构建高吞吐的 RL rollout 链路。

    [阅读中文](vllm_slime_rl_support/final/blog.md)
    · [English](vllm_slime_rl_support/final/blog.en.md)

-   <span class="article-kicker">MULTIMODAL RL · DIFFUSION</span>

    **VeRL-Omni 多模态强化学习**

    ---

    以异步奖励、步进式连续批处理与 Rollout 校准三项优化为主线，解析扩散模型 RL 的系统级设计与 FlowGRPO 工程映射。

    [阅读中文](verl_omni_multimodal_rl/final/blog.md)
    · [English](verl_omni_multimodal_rl/final/blog.en.md)

-   <span class="article-kicker">QUANTIZATION · INFERENCE</span>

    **vLLM 低比特量化实践：AutoRound**

    ---

    从截断与舍入的误差博弈出发，拆解 SignSGD 块级联合优化、QDQ 伪量化、vLLM 集成与 CFG 并行加速。

    [阅读中文](auto_round_vllm_vlm_low_bit/final/blog.md)
    · [English](auto_round_vllm_vlm_low_bit/final/blog.en.md)

-   <span class="article-kicker">ASCEND · INFERENCE OPTIMIZATION</span>

    **GLM-5 昇腾推理优化**

    ---

    从 TTFT、TPOT 与显存约束出发，拆解 GLM-5 的并行策略、PD 分离、KV Cache 压缩、图模式与融合算子优化。

    [阅读中文](ascend_glm5_deployment/final/blog.md)
    · [English](ascend_glm5_deployment/final/blog.en.md)

</div>

## 关于本站

文章由本仓库的技术博客流水线生成：先对视频进行带时间戳转写，再分析 PPT/PDF，建立证据与章节结构，最后经过写作、审校和双语排版。网页由 MkDocs Material 构建，并通过 GitHub Actions 自动发布。
