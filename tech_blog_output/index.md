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
  <div class="vllm-engine" aria-label="vLLM 推理流水线示意">
    <div class="vllm-engine__title"><span></span> vLLM ENGINE <b>RUNNING</b></div>
    <div class="vllm-flow">
      <div><small>01</small><strong>REQUESTS</strong><span>prompt queue</span></div>
      <i>→</i>
      <div><small>02</small><strong>SCHEDULER</strong><span>continuous batch</span></div>
      <i>→</i>
      <div><small>03</small><strong>KV BLOCKS</strong><span>paged memory</span></div>
      <i>→</i>
      <div><small>04</small><strong>DECODE</strong><span>token stream</span></div>
    </div>
    <div class="vllm-tokens" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
    <div class="vllm-stats"><span>KV CACHE <b>86%</b></span><span>ACTIVE SEQS <b>128</b></span><span>OUTPUT <b>STREAMING</b></span></div>
  </div>
</section>

<div class="vllm-topics">
  <span>PagedAttention</span><span>Continuous Batching</span><span>KV Cache</span><span>Speculative Decoding</span><span>MoE Serving</span>
</div>

## 最新文章

<div class="grid cards" markdown>

-   <span class="article-kicker">ARCHITECTURE · MEMORY</span>

    **Kimi K3 推理后端演进**

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

-   <span class="article-kicker">TRAINING SYSTEMS</span>

    **长序列 MoE RL**

    ---

    以 32 张 H100 运行 128K 序列为案例，梳理显存解耦、通信重叠与并行配置收敛方法。

    [阅读中文](long_sequence_moe_rl/final/blog.md)
    · [English](long_sequence_moe_rl/final/blog.en.md)

</div>

## 关于本站

文章由本仓库的技术博客流水线生成：先对视频进行带时间戳转写，再分析 PPT/PDF，建立证据与章节结构，最后经过写作、审校和双语排版。网页由 MkDocs Material 构建，并通过 GitHub Actions 自动发布。
