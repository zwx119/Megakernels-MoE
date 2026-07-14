# Mamba3 A100 Forward 共驻留验证设计

## 目标

验证 Mamba3 SISO forward 的两条拆分分支几乎不并发，主要是因为当前
`8 warps / maxnreg=256` 配置造成 SM 资源无法共驻留，还是因为 A100 上两条
分支存在更根本的架构或带宽竞争。

本轮只做 16K 单点诊断，不直接扩展到完整序列长度矩阵。

## 实验范围

- GPU：单张 A100-SXM4-80GB。
- Shape：沿用现有 Mamba3 SISO 实验，BS=1、hidden size=2560、state size=128、
  expand=2、head dim=64、80 heads、chunk size=64、BF16。
- Sequence length：16384。
- 方向：forward；backward 保留现有结果作为 A100 能够执行并发 kernel 的对照。
- 实现：沿用现有 `FULL`、`NO_H`、`H_ONLY` 和 two-stream overlap 路径，不改变
  数学计算。

## 配置扫描

扫描以下联合配置：

- `num_warps`: 4、8。
- `maxnreg`: 128、192、256。
- `num_stages`: 固定为 1。

同一组配置同时用于 `NO_H` 与 `H_ONLY`，先保持变量可解释。每组先做正确性
检查和 CUDA event 预筛，再对通过且有潜力的配置运行三次 Nsight Systems。

## 测量指标

对每个配置记录：

1. `NO_H` 与 `H_ONLY` 的独立 kernel latency。
2. overlap 模式的 `core_sum_ms`、`core_busy_ms` 和
   `concurrent_ms = core_sum_ms - core_busy_ms`。
3. `core_span_ms` 与相对 `FULL` 的 core speedup。
4. 完整 block span 与相对 `FULL` 的 block speedup。
5. 输出与 official `FULL` 的最大绝对误差和梯度误差。

单支 kernel 的独立最快配置不自动视为最佳配置；最终按 overlap 后的
`core_busy_ms` 和 block span 选择。

## 判定标准

- **共驻留成立**：`concurrent_ms` 显著高于现有 16K 的约 0.015 ms，并且重复
  测量稳定。
- **值得继续**：相对 official `FULL`，16K 的 `core_busy_ms` 至少达到 1.03x
  加速，且完整 block 不退化。
- **仅证明资源限制**：并发量明显增加，但 core 或 block 仍未加速。此结果用于
  说明 A100 上的资源配置影响共驻留，但 merge/重复预处理仍是主要瓶颈。
- **停止条件**：所有配置仍接近零并发，或均导致明显 block 退化。此时不做
  8K--64K sweep，将结论限定为 A100 上的负结果，并把 H100 作为后续验证。

## 输出

- 原始 event timing 与 Nsight 报告。
- 每组配置的 CSV 汇总。
- 一份 Markdown 结论，明确区分独立 latency、kernel sum、busy time、span 和
  block time。

## 非目标

本轮不实现新的 fused merge、不修改 backward、不对 H100 作未经测量的外推，
也不把结果直接加入最终论文主图。
