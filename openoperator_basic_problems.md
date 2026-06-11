# OpenOperator Basic Problems

Total: 20

## 001 001_LeakyReLU

- Category: pointwise
- Source: operaters
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x, double negative_slope);`

实现一个 BangC 算子，对输入张量应用 LeakyReLU 激活处理。输入包括形状为 `[batch, dim]` 的张量 `x` 以及表示负半轴斜率的标量 `negative_slope`；输出张量的形状与 `x` 相同。

## 002 002_matrix_scalar_multiplication

- Category: pointwise
- Source: kernelbench
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor A, double s);`

实现一个 BangC 算子，实现矩阵与标量的逐元素乘法运算。输入包括形状为 `[M, N]` 的张量 `A` 以及浮点数标量 `s`；输出为与 `A` 形状相同的张量。

## 003 003_LogSoftmax

- Category: softmax
- Source: operaters
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x, int dim);`

实现一个 BangC 算子，对输入张量在指定维度上执行 LogSoftmax 激活计算。输入包括形状为 `[batch_size, dim]` 的张量 `x` 以及用于指定计算维度的整数参数 `dim`；输出张量的形状与 `x` 保持一致。

## 004 004_batched_matrix_multiplication

- Category: matrix
- Source: kernelbench
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor A, torch::Tensor B);`

实现一个 BangC 算子，执行两个三维张量的批量矩阵乘法（Batch Matrix Multiplication）。输入包括形状为 `[batch_size, m, k]` 的张量 `A` 和形状为 `[batch_size, k, n]` 的张量 `B`；输出张量的形状为 `[batch_size, m, n]`。

## 005 005_average_pooling_2d

- Category: pool
- Source: kernelbench
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x,int kernel_size);`

实现一个 BangC 算子，对输入张量执行二维平均池化（2D Average Pooling）计算。输入包括形状为 `[batch_size, channels, height, width]` 的张量 `x`，以及用于指定窗口大小、步幅和填充的标量参数 `kernel_size`、`stride` 与 `padding`；输出为池化处理后的张量。

## 009 009_conv_standard_1D

- Category: conv
- Source: operaters
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x, torch::Tensor kernel, int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, int groups, int bias);`

实现一个 BangC 算子，执行标准的一维卷积运算。输入包括形状为 `[batch, in_channels, length]` 的数据张量 `x`、形状为 `[out_channels, in_channels/groups, kernel_size]` 的权重张量 `weight` 以及形状为 `[out_channels]` 的可选偏置张量 `bias`。算子还需支持通过 `stride`、`padding`、`dilation` 和 `groups` 等参数配置卷积行为，并输出形状为 `[batch, out_channels, length_out]` 的张量。

## 012 012_conv_transposed_2D__asymmetric_input__square_kernel

- Category: conv
- Source: operaters
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x, torch::Tensor kernel, int in_channels, int out_channels, int kernel_size, int stride, int padding, int output_padding, int groups, bool bias);`

实现一个 BangC 算子，执行二维转置卷积运算。输入包括形状为 `[batch, in_channels, h_in, w_in]` 的数据张量 `x`、形状为 `[in_channels, out_channels // groups, kernel_size, kernel_size]` 的卷积核张量以及可选的偏置张量；算子通过 `stride`、`padding`、`output_padding` 和 `groups` 等参数控制输出特征图的形状与计算逻辑。

## 023 023_Matrix_vector_multiplication_

- Category: matrix
- Source: operaters
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor A, torch::Tensor B);`

实现一个 BangC 算子，执行矩阵与向量的乘法运算（GEMV）。输入包括形状为 `[M, K]` 的矩阵 `A` 和形状为 `[K, 1]` 的向量 `B`；输出为两者相乘得到的形状为 `[M, 1]` 的结果张量。

## 034 034_Argmax_over_a_dimension

- Category: reduce
- Source: operaters
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x, int dim);`

实现一个 BangC 算子，沿指定维度计算输入张量的最大值索引（Argmax）。输入包括形状为 `[D_0, D_1, ..., D_{n-1}]` 的张量 `x` 以及一个用于指定规约维度的整数参数 `dim`；输出为移除该维度后的索引张量。

## 039 039_BatchNorm

- Category: fused
- Source: operaters
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x, int num_features);`

实现一个 BangC 算子，对四维张量执行二维批归一化（Batch Normalization）计算。输入包括形状为 `[batch, channel, H, W]` 的数据张量 `x` 以及表示特征通道数的整数 `num_features`；输出张量与 `x` 形状相同。

## 051 051_cumsum

- Category: cum
- Source: operaters
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x, int dim);`

实现一个 BangC 算子，对输入张量沿指定维度计算累加前缀和（Cumulative Sum）。输入包括形状为 `[d0, d1, ..., dn]` 的张量 `x` 以及表示操作维度的整数 `dim`；输出张量的形状与 `x` 保持一致。

## 056 056_gather

- Category: io
- Source: operaters
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x, torch::Tensor indices, torch::Tensor bin_ids, torch::Tensor weights, torch::Tensor bins, int top_k);`

实现一个 BangC 算子，根据索引和分桶规则将输入张量的行数据收集并加权排列到目标缓冲区。输入包括形状为 `[tokens, hidden_size]` 的张量 `x`、形状均为 `[num_elements]` 的索引 `indices` 和分桶 ID `bin_ids`、形状为 `[num_elements]` 的可选权重张量 `weights`、分桶边界张量 `bins` 以及标量 `top_k`；输出张量的形状为 `[tokens * top_k, hidden_size]`。

## 070 070_Sqrt

- Category: pointwise
- Source: custom
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x);`

实现一个 BangC 算子，对输入张量执行逐元素的绝对值计算并求其平方根。输入为形状为 `[batch_size, dim]` 的张量 `x`；输出张量与 `x` 的形状一致。

## 075 075_TopK

- Category: reduce
- Source: custom
- Dtype: float16
- C++ Wrapper: `std::vector<torch::Tensor> bang_func(torch::Tensor x, int k, int dim);`

实现一个 BangC 算子，用于在输入张量的指定维度上提取前 $k$ 个最大的数值及其对应的索引。输入包括形状为 `[batch, dim]` 的张量 `x`，以及整型标量参数 `k` 和 `dim`，分别表示需要提取的元素数量和操作的维度；输出为包含最大值及其索引的两个张量。

## 103 103_MSE_Loss

- Category: loss
- Source: custom
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor predictions, torch::Tensor targets);`

实现一个 BangC 算子，计算预测张量与目标张量之间的均方误差（MSE Loss）。输入包括形状均为 `[batch, dim]` 的预测张量 `predictions` 和目标张量 `targets`；输出为计算所得的标量损失值。

## 104 104_KL_Divergence_Loss

- Category: loss
- Source: custom
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor input_log_prob, torch::Tensor target_prob);`

实现一个 BangC 算子，计算预测对数概率与目标概率之间的 KL 散度，并按照 batch 维度进行平均规约。输入包括形状均为 `[batch, N]` 的张量 `input_log_prob` 和 `target_prob`，分别对应预测的对数概率分布和目标概率分布；输出为按 `batchmean` 模式计算得到的标量结果。

## 109 109_Scatter_add

- Category: io
- Source: custom
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor src, torch::Tensor index, int dim_size);`

实现一个 BangC 算子，根据索引张量将源张量的数据累加到目标张量的对应位置。输入包括形状为 `[N, D]` 的源张量 `src`、形状为 `[N]` 的索引张量 `index` 以及指定输出首维长度的整数 `dim_size`；输出张量的形状为 `[dim_size, D]`。

## 111 111_Masked_select

- Category: io
- Source: custom
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor input, double threshold);`

实现一个 BangC 算子，根据给定的阈值筛选输入张量中的元素。输入包括形状为 `[M, N]` 的张量 `input` 以及浮点数标量 `threshold`；算子将 `input` 中所有大于 `threshold` 的元素提取并展平为一维张量输出。

## 135 135_Dilated_conv_2D

- Category: conv
- Source: custom
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x, torch::Tensor kernel, int in_channels, int out_channels, int kernel_size, int dilation, int padding);`

实现一个 BangC 算子，对四维输入张量执行带有填充和空洞特性的二维卷积运算。输入为形状为 `[batch, in_channels, H, W]` 的张量 `x`，其运算受卷积核大小 `kernel_size`、膨胀系数 `dilation` 以及填充宽度 `padding` 等参数控制；输出为卷积计算后的特征图张量。

## 138 138_GRU_forward

- Category: fused
- Source: custom
- Dtype: float16
- C++ Wrapper: `torch::Tensor bang_func(torch::Tensor x, int input_size, int hidden_size, int num_layers);`

实现一个 BangC 算子，实现多层门控循环单元（GRU）的前向计算过程。输入为形状为 `[batch, seq_len, input_size]` 的张量 `x`，并根据给定的隐藏层维度 `hidden_size` 和层数 `num_layers` 对序列数据进行处理；算子最终输出形状为 `[batch, seq_len, hidden_size]` 的隐藏状态序列。
