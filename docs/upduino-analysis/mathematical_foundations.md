# Mathematical Foundations for AI-on-Chip Implementation

**Document Version:** 1.0
**Target Platform:** UPduino v3.1 (iCE40 UP5K FPGA)
**Date:** 2026-01-04

## Table of Contents

1. [Neural Network Mathematics](#1-neural-network-mathematics)
2. [Quantization Theory](#2-quantization-theory)
3. [Memory-as-Compute Mathematics](#3-memory-as-compute-mathematics)
4. [Hardware Optimization](#4-hardware-optimization)
5. [Stochastic Computing](#5-stochastic-computing)
6. [Verilog Implementation Mapping](#6-verilog-implementation-mapping)
7. [Performance Prediction Models](#7-performance-prediction-models)

---

## 1. Neural Network Mathematics

### 1.1 Matrix Multiplication

The fundamental operation in neural networks is the affine transformation:

```
Y = XW + b
```

Where:
- **X** ∈ ℝ^(n×m): Input matrix (n samples, m features)
- **W** ∈ ℝ^(m×k): Weight matrix (m inputs, k outputs)
- **b** ∈ ℝ^k: Bias vector
- **Y** ∈ ℝ^(n×k): Output matrix

**Element-wise formulation:**

```
y_ij = Σ(k=1 to m) x_ik · w_kj + b_j
```

**Hardware Implications:**
- Requires **m** multiply-accumulate (MAC) operations per output
- Total operations: **O(n·m·k)** for full matrix
- Memory bandwidth: **(n·m + m·k + k)** elements read, **(n·k)** written

**Fixed-Point Representation:**

For Q-format numbers (Qm.n notation: m integer bits, n fractional bits):

```
X_fixed = round(X · 2^n)
W_fixed = round(W · 2^n)
Y_fixed = (X_fixed · W_fixed) >> n  // Right shift to adjust for scaling
```

### 1.2 Convolution Operation

2D convolution for image processing:

```
(f * g)[i,j] = ΣΣ f[m,n] · g[i-m, j-n]
             m n
```

For discrete convolution with kernel size K×K:

```
Y[i,j] = ΣΣ X[i+m, j+n] · W[m,n] + b
        m=0 to K-1
        n=0 to K-1
```

**Depthwise Separable Convolution:**

Reduces computation by factoring into depthwise and pointwise:

```
Y_depthwise[i,j,c] = ΣΣ X[i+m, j+n, c] · W_depthwise[m,n,c]
                      m n

Y_pointwise[i,j,c'] = Σ Y_depthwise[i,j,c] · W_pointwise[c,c']
                       c
```

**Computational Savings:**

- Standard: **H·W·C_in·C_out·K²** operations
- Depthwise Separable: **H·W·C_in·K² + H·W·C_in·C_out** operations
- Reduction factor: **~(1/C_out + 1/K²)**

### 1.3 Activation Functions

#### 1.3.1 ReLU (Rectified Linear Unit)

```
f(x) = max(0, x) = {
  x,  if x > 0
  0,  if x ≤ 0
}

f'(x) = {
  1,  if x > 0
  0,  if x ≤ 0
}
```

**Hardware Implementation:**
- Simply check sign bit and mux with zero
- No multipliers required
- Verilog: `y = (x[MSB] == 0) ? x : 0`

#### 1.3.2 Sigmoid

```
σ(x) = 1 / (1 + e^(-x))

σ'(x) = σ(x) · (1 - σ(x))
```

**Approximations for Hardware:**

Piecewise linear approximation:

```
σ_approx(x) = {
  0,           if x < -5
  0.5 + x/8,   if -5 ≤ x ≤ 5
  1,           if x > 5
}
```

Polynomial approximation (degree 3):

```
σ_approx(x) ≈ 0.5 + 0.25x - 0.03125x³  (for |x| < 2)
```

#### 1.3.3 Tanh

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = 2σ(2x) - 1

tanh'(x) = 1 - tanh²(x)
```

**Piecewise Linear Approximation:**

```
tanh_approx(x) = {
  -1,          if x < -2
  x/2,         if -2 ≤ x ≤ 2
  1,           if x > 2
}
```

#### 1.3.4 GELU (Gaussian Error Linear Unit)

```
GELU(x) = x · Φ(x) = x · 0.5 · (1 + erf(x/√2))
```

**Fast Approximation:**

```
GELU_approx(x) ≈ 0.5x · (1 + tanh(√(2/π) · (x + 0.044715x³)))
```

Or simpler:

```
GELU_approx(x) ≈ x · σ(1.702x)
```

### 1.4 Backpropagation (On-Chip Learning)

For layer **l** with activation **a^l**, weights **W^l**, and loss **L**:

**Forward Pass:**

```
z^l = W^l · a^(l-1) + b^l
a^l = f(z^l)
```

**Backward Pass (Gradient Computation):**

```
δ^l = ∂L/∂z^l = (∂L/∂a^l) ⊙ f'(z^l)

∂L/∂W^l = δ^l · (a^(l-1))^T
∂L/∂b^l = δ^l

δ^(l-1) = (W^l)^T · δ^l
```

Where **⊙** denotes element-wise multiplication (Hadamard product).

**Weight Update (SGD):**

```
W^l ← W^l - η · ∂L/∂W^l
b^l ← b^l - η · ∂L/∂b^l
```

Where **η** is the learning rate.

**Hardware Considerations:**
- Requires storing activations during forward pass
- Backward pass has similar computational complexity to forward
- Weight updates require multiply-add operations

---

## 2. Quantization Theory

### 2.1 Fixed-Point Number Representation

**Q-Format Notation:** Qm.n (m integer bits, n fractional bits)

For a value **x** in floating-point:

```
x_fixed = round(x · 2^n)
```

**Range and Precision:**

- **Minimum value:** -2^m
- **Maximum value:** 2^m - 2^(-n)
- **Step size (LSB):** 2^(-n)
- **Total bits:** m + n + 1 (including sign bit)

**Example:** Q7.8 format
- Range: [-128, 127.996]
- Precision: 1/256 ≈ 0.0039
- Total: 16 bits

**Arithmetic Operations:**

Addition/Subtraction (same Q-format):
```
(a + b)_Qm.n = a_Qm.n + b_Qm.n
```

Multiplication (Qm1.n1 × Qm2.n2 → Q(m1+m2).(n1+n2)):
```
(a · b)_Q(m1+m2).(n1+n2) = (a_Qm1.n1 · b_Qm2.n2) >> (n1+n2)
```

To maintain format, require saturation and rounding.

### 2.2 Quantization Error Analysis

**Uniform Quantization:**

```
Q(x) = Δ · round(x / Δ)
```

Where **Δ** is the quantization step size.

**Quantization Error:**

```
ε = x - Q(x)
ε ∈ [-Δ/2, Δ/2]  (for rounding)
```

**Signal-to-Quantization-Noise Ratio (SQNR):**

For n-bit quantization of signal with range R:

```
Δ = R / 2^n
SQNR ≈ 6.02n + 1.76  (in dB)
```

Each additional bit improves SQNR by ~6 dB.

**Mean Squared Error (MSE):**

```
MSE = E[ε²] = Δ² / 12  (for uniform distribution)
```

### 2.3 Symmetric vs Asymmetric Quantization

#### Symmetric Quantization

Maps [-α, α] to [-2^(b-1), 2^(b-1)-1] where b is bit-width:

```
s = α / (2^(b-1) - 1)
x_q = round(x / s)
x_dequant = x_q · s
```

- **Pros:** Zero-point is exactly 0, simpler hardware
- **Cons:** Wastes range if distribution is asymmetric

#### Asymmetric Quantization

Maps [β, α] to [0, 2^b-1]:

```
s = (α - β) / (2^b - 1)
z = round(-β / s)
x_q = round(x / s) + z
x_dequant = (x_q - z) · s
```

Where **z** is the zero-point.

- **Pros:** Better utilizes range for skewed distributions
- **Cons:** Requires storing and handling zero-point offset

### 2.4 Per-Channel vs Per-Tensor Quantization

#### Per-Tensor Quantization

Single scale **s** and zero-point **z** for entire tensor:

```
W_q = round(W / s_w) + z_w
```

- **Memory efficient:** 2 parameters per tensor
- **Lower accuracy:** Can't handle channel variation

#### Per-Channel Quantization

Different scale **s_c** per channel (typically output channels):

```
W_q[c, :, :, :] = round(W[c, :, :, :] / s_w[c]) + z_w[c]
```

For matrix multiplication:

```
Y[i, j] = Σ_k (X_q[i,k] - z_x) · (W_q[k,j] - z_w[j]) · s_x · s_w[j] + b[j]
```

Can be rewritten as:

```
Y[i, j] = s_x · s_w[j] · (Σ_k X_q[i,k] · W_q[k,j] - compensation_term[i,j])
```

- **Higher accuracy:** Adapts to per-channel statistics
- **Cost:** Additional memory for per-channel scales

### 2.5 Binary and Ternary Neural Networks

#### Binary Neural Networks (BNNs)

Weights and activations constrained to {-1, +1}:

```
W_binary = sign(W) = {
  +1,  if W ≥ 0
  -1,  if W < 0
}
```

**Forward Pass:**

```
Y = sign(X_binary · W_binary)
```

**XNOR-Popcount Implementation:**

For binary values represented as bits (0 → -1, 1 → +1):

```
dot_product = 2 · popcount(XNOR(x, w)) - n
```

Where **n** is the dimension.

**Scaling Factor:**

To improve accuracy, use real-valued scaling:

```
Y = α · sign(X · W_binary)
```

Where **α = E[|W|]** is the mean absolute weight.

#### Ternary Neural Networks (TNNs)

Weights in {-1, 0, +1}:

```
W_ternary = {
  +1,  if W > Δ
  0,   if |W| ≤ Δ
  -1,  if W < -Δ
}
```

Threshold **Δ** commonly chosen as:

```
Δ = 0.7 · E[|W|]
```

**Computational Advantage:**

- Multiplication becomes conditional addition
- Sparsity from zeros reduces operations
- Can be implemented with adders and muxes

**Hardware Efficiency:**

- BNN: 1 bit/weight → **32× memory reduction** (vs FP32)
- TNN: ~1.58 bits/weight (entropy-coded) → **~20× reduction**
- BNN multiplication: **XNOR + popcount** (very fast)

---

## 3. Memory-as-Compute Mathematics

### 3.1 Crossbar Array Fundamentals

A resistive crossbar array computes matrix-vector multiplication in **O(1)** time.

**Basic Equation:**

```
V_out = G × V_in
```

Where:
- **V_in** ∈ ℝ^n: Input voltages applied to rows
- **G** ∈ ℝ^(m×n): Conductance matrix (stores weights)
- **V_out** ∈ ℝ^m: Output voltages/currents at columns

**Ohm's Law at Each Cross-Point:**

```
I_ij = G_ij · V_i
```

**Kirchhoff's Current Law (KCL) at Output:**

```
I_out[j] = Σ_i I_ij = Σ_i G_ij · V_in[i]
```

This is exactly a matrix-vector multiplication!

**Conductance Encoding:**

For weight **w ∈ [-w_max, w_max]**:

```
G = (w + w_max) · G_max / (2 · w_max)
```

Or using differential pairs (positive and negative):

```
w = (G_pos - G_neg) · k
```

Where **k** is a scaling constant.

### 3.2 Analog Computing Principles

**Energy Efficiency:**

Analog computation is energy-efficient due to parallelism:

```
E_analog ≈ C · V² · N  (charging capacitance)
E_digital ≈ α · C · V² · N · log₂N  (switching energy)
```

Where **α** is activity factor.

**Theoretical Speedup:**

For **n×n** matrix:
- Digital: **O(n²)** or **O(n^2.37)** (Strassen)
- Analog crossbar: **O(1)** (single voltage application)

**Precision Limitations:**

Analog precision limited by:

```
SNR = V_signal / V_noise
ENOB = log₂(SNR) ≈ (SNDR_dB - 1.76) / 6.02
```

Typically **4-8 effective bits** for analog computing.

### 3.3 Digital Approximations of Analog Compute

**Time-Domain Encoding:**

Represent values as pulse widths:

```
V(t) = {
  V_high,  for 0 ≤ t < w·T
  V_low,   for w·T ≤ t < T
}
```

Where **w ∈ [0,1]** is the normalized weight.

**Integration gives multiplication:**

```
∫₀ᵀ V₁(t) · V₂(t) dt ∝ w₁ · w₂
```

**Stochastic Streams:**

Encode value as probability of '1' in bit stream:

```
P(bit = 1) = (x + 1) / 2  (for x ∈ [-1, 1])
```

Multiplication via AND gate:
```
P(A ∧ B) = P(A) · P(B)
```

### 3.4 Precision vs Energy Tradeoffs

**Energy per Operation:**

```
E_op = C · V² · α
```

Where:
- **C:** Capacitance
- **V:** Supply voltage
- **α:** Activity factor

**Voltage Scaling:**

Reducing voltage reduces energy quadratically but increases delay:

```
t_delay ∝ C·V / I ∝ V / (V - V_th)²
```

**Precision-Energy Relationship:**

For n-bit precision:

```
E_total = E_op · 2^n  (for linear search)
E_total = E_op · n    (for successive approximation)
```

**Optimal Operating Point:**

For neural networks, empirical studies show:

```
E_optimal ≈ E_8bit · (desired_accuracy / baseline_accuracy)^k
```

Where **k ≈ 2-3** depending on network.

**Tradeoff Curve:**

```
Accuracy = A_max · (1 - e^(-E/E₀))
```

Diminishing returns beyond 8-16 bit precision for most applications.

---

## 4. Hardware Optimization

### 4.1 Systolic Array Mathematics

A systolic array processes data in a rhythmic, pipelined fashion.

**Basic 1D Systolic Array for Matrix-Vector Multiplication:**

For **y = Wx**:

At time **t**, processing element **PE[i]** computes:

```
y_i(t) = y_i(t-1) + w_i · x(t-i)
```

**2D Systolic Array for Matrix-Matrix Multiplication:**

For **C = A × B**, at each PE[i,j]:

```
c_ij(t) = c_ij(t-1) + a_ik(t) · b_kj(t)
```

**Throughput:**

- **Latency:** O(n) cycles to fill pipeline
- **Throughput:** 1 result per cycle (after pipeline filled)
- **Efficiency:** n² PEs produce n² results in O(n) time → **O(n)** speedup

**Data Movement:**

Each data element is reused **n** times (for n×n matrix):

```
Total_Memory_Accesses = 2n² + n²  (read A, B, write C)
Compute_Operations = n³
Arithmetic_Intensity = n³ / (3n²) = n/3
```

### 4.2 Dataflow Optimization

**Minimizing Data Movement:**

Energy cost hierarchy (from expensive to cheap):
1. **DRAM access:** ~200 pJ
2. **SRAM access:** ~5 pJ
3. **Register access:** ~0.1 pJ
4. **Compute (MAC):** ~0.2 pJ

**Optimization Principle:**

Maximize data reuse in on-chip buffers.

**Loop Tiling for Cache Optimization:**

For **C = A × B** (all n×n):

```
for ii = 0 to n step B:
  for jj = 0 to n step B:
    for kk = 0 to n step B:
      for i = ii to min(ii+B, n):
        for j = jj to min(jj+B, n):
          for k = kk to min(kk+B, n):
            C[i,j] += A[i,k] * B[k,j]
```

Block size **B** chosen to fit in cache.

**Data Reuse Factor:**

```
R_A = n / B  (each A element reused n/B times)
R_B = n / B
R_C = B      (each C element accumulated B times)
```

**Roofline-Aware Scheduling:**

Place operations based on arithmetic intensity:

```
AI = FLOPS / Bytes_Transferred
Peak_Performance = min(Peak_Compute, AI · Peak_Bandwidth)
```

### 4.3 Roofline Model

**Model Equation:**

```
Attainable_GFLOPS = min(Peak_GFLOPS, Arithmetic_Intensity · Peak_Bandwidth)
```

**Arithmetic Intensity:**

```
AI = Operations / Bytes_Transferred
```

For matrix multiplication (n×n):

```
AI = 2n³ / (2n² · sizeof(float))  (assuming C is small)
   = n / sizeof(float)
```

**Memory-Bound vs Compute-Bound:**

- **Memory-bound:** AI < Peak_GFLOPS / Peak_Bandwidth
- **Compute-bound:** AI ≥ Peak_GFLOPS / Peak_Bandwidth

**Ridge Point:**

```
AI_ridge = Peak_GFLOPS / Peak_Bandwidth
```

For iCE40 UP5K (rough estimates):
- Peak compute: ~100 MOPS (8-bit, 1 DSP)
- Peak bandwidth: ~1 GB/s (internal SRAM)
- Ridge point: ~0.1 OPS/Byte

Most operations will be **compute-bound** on FPGA.

### 4.4 Amdahl's Law for Parallelization

**Amdahl's Law:**

```
Speedup = 1 / (S + P/N)
```

Where:
- **S:** Fraction of serial execution time
- **P:** Fraction of parallel execution time (S + P = 1)
- **N:** Number of processors

**Limiting Speedup:**

```
Max_Speedup = 1 / S  (as N → ∞)
```

**Example:**

If 10% of code is serial (S=0.1):

```
Max_Speedup = 1 / 0.1 = 10×  (even with infinite cores)
```

**Application to Neural Networks:**

- **Parallelizable:** Matrix multiplications, convolutions
- **Serial:** Batch normalization, activation functions (element-wise)
- **Batch processing helps:** Amortizes serial overhead

**Gustafson's Law (Alternative):**

For scaled problems:

```
Speedup = N - S·(N - 1) = N + S - S·N
```

This is more optimistic for large-scale parallel systems.

### 4.5 Memory Bandwidth Optimization

**Effective Bandwidth:**

```
BW_eff = Data_Transferred / Time
```

**Bandwidth Utilization:**

```
U = BW_eff / BW_peak
```

**Optimizations:**

1. **Burst Access:** Transfer contiguous blocks
   ```
   Efficiency = Burst_Length / (Burst_Length + Overhead)
   ```

2. **Double Buffering:** Overlap compute and memory transfer
   ```
   Time_total = max(Time_compute, Time_memory)
   ```

3. **Data Compression:** Reduce bandwidth needs
   ```
   BW_required = BW_uncompressed / Compression_Ratio
   ```

---

## 5. Stochastic Computing

### 5.1 Probability-Based Number Representation

**Stochastic Stream:**

A value **x ∈ [0, 1]** is represented by a bit stream where:

```
P(bit = 1) = x
```

**Unipolar Representation:** x ∈ [0, 1]

```
x = (# of 1's) / (total bits)
```

**Bipolar Representation:** x ∈ [-1, 1]

```
x = 2 · P(bit = 1) - 1 = (# of 1's - # of 0's) / (total bits)
```

**Precision:**

For N-bit stream:

```
Resolution = 1 / N
Standard_Deviation = √(x(1-x)/N)
```

**Example:**

To represent 0.75 with 8-bit stream:

```
Possible stream: 11011101  (6 ones, 2 zeros)
x ≈ 6/8 = 0.75
```

### 5.2 Stochastic Multiplication

**Unipolar Multiplication (AND gate):**

For independent streams A, B:

```
P(A ∧ B = 1) = P(A = 1) · P(B = 1)
```

**Circuit:** Single AND gate

**Example:**

```
A: 0.6 → stream with P(1) = 0.6
B: 0.5 → stream with P(1) = 0.5
A ∧ B → stream with P(1) = 0.3
```

**Bipolar Multiplication (XNOR gate):**

For bipolar representation:

```
P(XNOR(A,B) = 1) = P(A=B) = (1 + a·b) / 2
```

Where **a, b ∈ [-1, 1]**.

**Scaled output:**

```
z = 2 · P(XNOR) - 1 = a · b
```

### 5.3 Scaled Addition (MUX)

**Addition via Multiplexer:**

For **z = α·a + β·b** where **α + β = 1**:

```
z = MUX(sel, a, b)
```

Where **sel** is a random stream with **P(sel=1) = α**.

**Example:**

```
z = 0.3·a + 0.7·b
sel: random stream with P(1) = 0.3
z = sel ? a : b
```

**Weighted Sum:**

For **z = w₁·x₁ + w₂·x₂** (where w₁ + w₂ = 1):

```
z = MUX(sel, x₁, x₂)  with P(sel=1) = w₁
```

**General Addition (via conversion):**

For arbitrary **z = a + b**:

Requires conversion to binary, addition, and re-conversion.

Cost negates stochastic advantage.

### 5.4 Activation Functions in Stochastic Domain

#### ReLU

**Implementation:**

```
ReLU(x) = MUX(sign, x, 0)
```

Where **sign** extracts the sign bit.

**Bipolar domain:**

```
ReLU_stoch(x) = (x + |x|) / 2
```

Can be implemented with rectifier circuit.

#### Sigmoid Approximation

**JK Flip-Flop Implementation:**

A JK flip-flop with **J=input**, **K=output** approximates sigmoid:

```
P(Q=1) ≈ σ(input_probability)
```

**Analysis:**

```
P(Q=1 at t+1) = P(J=1, K=0) + P(J=0, K=0)·P(Q=1 at t)
              = p_in(1 - p_out) + (1 - p_in)·p_out

At steady state (p_out = p_in):
Behaves like sigmoid-shaped mapping
```

#### Tanh Using Bipolar Streams

**FSM-Based Tanh:**

A finite state machine with saturation implements tanh:

```
state(t+1) = clamp(state(t) + input(t), -K, K)
output = state / K
```

For large **K**, this approximates integration → tanh behavior.

### 5.5 Stochastic Neural Network

**Forward Pass:**

```
z = Σ w_i · x_i  (weighted sum)
a = σ(z)         (activation)
```

**Stochastic Implementation:**

1. **Multiplication:** AND/XNOR gates
2. **Weighted sum:** MUX tree
3. **Activation:** JK flip-flop or FSM

**Hardware Cost:**

For **N** inputs:
- **N** AND/XNOR gates (multiplication)
- **log₂(N)** layers of MUXes (addition)
- **1** JK flip-flop (activation)

**Total:** ~**2N** gates (very compact!)

**Tradeoffs:**

- **Pros:** Ultra-low area, low power, inherent error tolerance
- **Cons:** Long latency (need hundreds of cycles for precision), randomness overhead

**Accuracy:**

For **N**-bit stochastic streams:

```
Effective_Precision ≈ log₂(N) bits
```

Example: 256-bit stream → ~8-bit equivalent precision

---

## 6. Verilog Implementation Mapping

### 6.1 Fixed-Point Matrix Multiplication

**Mathematical Formula:**

```
Y[i,j] = Σ X[i,k] · W[k,j]  (k from 0 to K-1)
```

**Verilog Module:**

```verilog
module fixed_point_mac #(
    parameter WIDTH = 16,      // Total bit width
    parameter FRAC = 8         // Fractional bits (Q7.8)
)(
    input  wire clk,
    input  wire rst_n,
    input  wire signed [WIDTH-1:0] a,    // Input
    input  wire signed [WIDTH-1:0] b,    // Weight
    input  wire valid_in,
    output reg  signed [WIDTH-1:0] sum,  // Accumulated output
    output reg  valid_out
);

    // Internal product (double width)
    reg signed [2*WIDTH-1:0] product;
    reg signed [2*WIDTH-1:0] acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= 0;
            sum <= 0;
            valid_out <= 0;
        end else if (valid_in) begin
            // Multiply
            product = a * b;

            // Accumulate (shift to align decimal points)
            acc = acc + product;

            // Saturate and round (after all accumulations)
            // Truncate extra fractional bits: acc >> FRAC
            sum <= acc[2*WIDTH-1:FRAC];  // Extract relevant bits

            valid_out <= 1;
        end
    end
endmodule
```

**Explanation:**

1. **Multiplication:** `a * b` produces `2*WIDTH` bits
2. **Accumulation:** Add products
3. **Scaling:** Right-shift by `FRAC` to align decimal point
4. **Saturation:** (Not shown) check overflow and clamp

### 6.2 Convolution Implementation

**Mathematical Formula:**

```
Y[i,j] = ΣΣ X[i+m, j+n] · W[m,n]  (m,n from 0 to K-1)
```

**Verilog Module (3×3 Kernel):**

```verilog
module conv2d_3x3 #(
    parameter DATA_WIDTH = 8,
    parameter IMAGE_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,
    input  wire [DATA_WIDTH-1:0] pixel_in,
    input  wire valid_in,
    input  wire signed [DATA_WIDTH-1:0] kernel [0:2][0:2],
    output reg  signed [2*DATA_WIDTH-1:0] conv_out,
    output reg  valid_out
);

    // Line buffers for 3x3 window
    reg [DATA_WIDTH-1:0] line_buf0 [0:IMAGE_WIDTH-1];
    reg [DATA_WIDTH-1:0] line_buf1 [0:IMAGE_WIDTH-1];
    reg [DATA_WIDTH-1:0] window [0:2][0:2];

    integer i, j;
    reg signed [2*DATA_WIDTH-1:0] mac_result;

    // Shift window and compute convolution
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv_out <= 0;
            valid_out <= 0;
        end else if (valid_in) begin
            // Shift line buffers
            line_buf1[0] <= pixel_in;
            for (i = 1; i < IMAGE_WIDTH; i = i + 1) begin
                line_buf0[i] <= line_buf0[i-1];
                line_buf1[i] <= line_buf1[i-1];
            end

            // Update 3x3 window
            window[0][0] <= line_buf0[IMAGE_WIDTH-1];
            window[0][1] <= line_buf0[IMAGE_WIDTH-2];
            window[0][2] <= line_buf0[IMAGE_WIDTH-3];

            window[1][0] <= line_buf1[IMAGE_WIDTH-1];
            window[1][1] <= line_buf1[IMAGE_WIDTH-2];
            window[1][2] <= line_buf1[IMAGE_WIDTH-3];

            window[2][0] <= pixel_in;
            // ... (complete window update)

            // Compute convolution (MAC)
            mac_result = 0;
            for (i = 0; i < 3; i = i + 1) begin
                for (j = 0; j < 3; j = j + 1) begin
                    mac_result = mac_result + (window[i][j] * kernel[i][j]);
                end
            end

            conv_out <= mac_result;
            valid_out <= 1;
        end
    end
endmodule
```

### 6.3 ReLU Activation

**Mathematical Formula:**

```
y = max(0, x)
```

**Verilog Module:**

```verilog
module relu #(
    parameter WIDTH = 16
)(
    input  wire signed [WIDTH-1:0] x,
    output wire [WIDTH-1:0] y
);

    // Check sign bit: if negative (MSB=1), output 0; else output x
    assign y = (x[WIDTH-1]) ? {WIDTH{1'b0}} : x;

endmodule
```

**Explanation:**

- Single-cycle combinational logic
- No DSP blocks needed
- Uses only mux (ternary operator)

### 6.4 Sigmoid Approximation (Piecewise Linear)

**Mathematical Approximation:**

```
σ(x) ≈ {
  0,           x < -4
  0.5 + x/8,   -4 ≤ x < 4
  1,           x ≥ 4
}
```

**Verilog Module:**

```verilog
module sigmoid_pwl #(
    parameter WIDTH = 16,
    parameter FRAC = 8    // Q7.8 format
)(
    input  wire signed [WIDTH-1:0] x,
    output reg  [WIDTH-1:0] y
);

    localparam signed [WIDTH-1:0] NEG_THRESHOLD = -4 << FRAC;  // -4.0
    localparam signed [WIDTH-1:0] POS_THRESHOLD =  4 << FRAC;  //  4.0
    localparam [WIDTH-1:0] ZERO = 0;
    localparam [WIDTH-1:0] ONE  = 1 << FRAC;   // 1.0
    localparam [WIDTH-1:0] HALF = 1 << (FRAC-1); // 0.5

    wire signed [WIDTH-1:0] slope_term;

    // x/8 = x >> 3
    assign slope_term = x >>> 3;

    always @(*) begin
        if (x < NEG_THRESHOLD) begin
            y = ZERO;
        end else if (x > POS_THRESHOLD) begin
            y = ONE;
        end else begin
            // 0.5 + x/8
            y = HALF + slope_term;
        end
    end

endmodule
```

### 6.5 Stochastic Multiplier

**Mathematical Formula:**

```
P(output = 1) = P(A = 1) · P(B = 1)
```

**Verilog Module:**

```verilog
module stochastic_mult (
    input  wire a,     // Stochastic stream A
    input  wire b,     // Stochastic stream B
    output wire z      // Product stream
);

    // AND gate implements multiplication
    assign z = a & b;

endmodule
```

**Incredibly simple!** This is the power of stochastic computing.

### 6.6 Stochastic Number Generator (LFSR-based)

**Verilog Module:**

```verilog
module stochastic_generator #(
    parameter WIDTH = 8
)(
    input  wire clk,
    input  wire rst_n,
    input  wire [WIDTH-1:0] value,      // Value to encode (0-255)
    output wire stochastic_bit          // Output stream
);

    // Linear Feedback Shift Register for pseudo-random sequence
    reg [WIDTH-1:0] lfsr;
    wire feedback;

    // LFSR feedback (example for 8-bit: x^8 + x^6 + x^5 + x^4 + 1)
    assign feedback = lfsr[7] ^ lfsr[5] ^ lfsr[4] ^ lfsr[3];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr <= 8'b10101010;  // Non-zero seed
        end else begin
            lfsr <= {lfsr[6:0], feedback};
        end
    end

    // Comparator: output 1 if value > random
    assign stochastic_bit = (value > lfsr);

endmodule
```

**Explanation:**

- **LFSR** generates pseudo-random sequence
- **Comparator** converts value to probability
- **P(output=1) = value / 256** (for 8-bit)

### 6.7 Quantized Weight Storage

**For 4-bit weights stored in BRAM:**

```verilog
module weight_memory #(
    parameter ADDR_WIDTH = 10,    // 1024 weights
    parameter WEIGHT_WIDTH = 4    // 4-bit quantized
)(
    input  wire clk,
    input  wire [ADDR_WIDTH-1:0] addr,
    output reg  signed [WEIGHT_WIDTH-1:0] weight
);

    // Block RAM instantiation
    reg signed [WEIGHT_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    // Initialize weights (from file or parameters)
    initial begin
        $readmemh("weights.hex", mem);
    end

    always @(posedge clk) begin
        weight <= mem[addr];
    end

endmodule
```

---

## 7. Performance Prediction Models

### 7.1 Computational Complexity

**Matrix Multiplication (Y = X·W):**

- **Dimensions:** X: n×m, W: m×k
- **Operations:** **2nmk** (multiply-add counts as 2)
- **Memory:** **(nm + mk + nk)** elements

**Convolution (2D, K×K kernel):**

- **Output size:** H_out × W_out × C_out
- **Operations per output:** **K² · C_in · 2**
- **Total operations:** **H_out · W_out · C_out · K² · C_in · 2**

**Depthwise Separable Convolution:**

- **Depthwise:** **H_out · W_out · C · K² · 2**
- **Pointwise:** **H_out · W_out · C_in · C_out · 2**
- **Total:** **(H_out · W_out · 2) · (C · K² + C_in · C_out)**

### 7.2 Latency Estimation

**Single MAC Operation:**

For FPGA DSP block:

```
T_MAC = T_mult + T_add ≈ 1-2 clock cycles  (pipelined)
```

**Matrix Multiplication (without parallelism):**

```
T_matmul = n · m · k · T_MAC
```

**With Parallelism (P processing elements):**

```
T_matmul = (n · m · k / P) · T_MAC
```

**Systolic Array (n×n PEs):**

```
T_latency = 3n · T_clock  (pipeline fill)
T_throughput = T_clock    (per result, after pipeline filled)
```

### 7.3 Throughput Estimation

**Throughput (OPS - Operations Per Second):**

```
Throughput = (Operations_per_inference · Frequency) / Latency_cycles
```

**Example:**

- **Network:** 1M MACs per inference
- **Clock:** 50 MHz
- **Parallelism:** 10 MACs/cycle
- **Latency:** 100k cycles

```
Throughput = (1M · 50M / 100k) / 1M = 500 inferences/second
MOPS = 1M · 500 / 1M = 500 MOPS
```

### 7.4 Memory Bandwidth Requirements

**Bandwidth Required:**

```
BW_required = (Data_per_operation · Operations_per_second) / Reuse_factor
```

**Example (Matrix Multiplication):**

- **Operation:** Y = X·W (n×n matrices)
- **Data per operation:** 3n² elements (X, W, Y)
- **Operations:** n³ MACs
- **Reuse:** Each element used n times

```
BW_required = (3n² · sizeof(element)) / n = 3n · sizeof(element)
```

For **n=64, 8-bit elements:**

```
BW_required = 3 · 64 · 1 byte = 192 bytes per matrix multiply
```

### 7.5 Power Estimation

**Dynamic Power:**

```
P_dynamic = α · C · V² · f
```

Where:
- **α:** Activity factor (0-1)
- **C:** Capacitance
- **V:** Supply voltage
- **f:** Frequency

**Energy per Operation:**

```
E_op = P_dynamic / (Operations_per_second)
```

**Typical FPGA Values:**

- **DSP MAC:** ~0.5-2 pJ/operation (at moderate frequency)
- **SRAM access:** ~5-20 pJ/access
- **Register:** ~0.1 pJ

**Example Power Budget:**

For 1000 MOPS at 50 MHz:

```
Power ≈ 1000M · 1pJ + memory_access_power
      ≈ 1W + memory_overhead
```

### 7.6 Roofline Model for iCE40 UP5K

**iCE40 UP5K Specifications:**

- **DSP Blocks:** 8 (16×16 multipliers)
- **SPRAM:** 120 Kb (4 blocks × 32Kb)
- **BRAM:** 15 blocks × 4Kb = 60 Kb
- **Logic Cells:** 5280
- **Max Frequency:** ~50 MHz (design dependent)

**Peak Performance:**

```
Peak_Compute = 8 DSPs × 2 ops/MAC × 50 MHz = 800 MOPS (8-bit)
```

**Peak Memory Bandwidth:**

Internal SRAM (SPRAM):

```
Peak_BW = 120 Kb / 8 × 50 MHz = 750 MB/s  (single-ported)
```

**Roofline Ridge Point:**

```
AI_ridge = Peak_Compute / Peak_BW
         = 800 MOPS / 750 MB/s
         = 1.07 OPS/Byte
```

**Interpretation:**

Operations with **AI > 1.07** are compute-bound; otherwise, memory-bound.

**Example:**

Matrix multiplication (n=32):

```
AI = 2n³ / (3n² · sizeof(element))
   = 2 · 32 / (3 · 1)
   = 21.3 OPS/Byte  → Compute-bound
```

Convolution (small kernel):

```
AI = K² · 2 / ((K² + 1) · sizeof(element))
   ≈ 2  → Compute-bound (barely)
```

### 7.7 Resource Utilization Estimates

**Matrix Multiplication (n×n with k PEs):**

```
DSPs: k
BRAM: ⌈(2n² · bits/element) / (BRAM_size)⌉
Logic Cells: ~100-500 per PE (control logic)
```

**Example (32×32, 8-bit, 4 PEs):**

```
DSPs: 4
BRAM: ⌈(2 · 32² · 8) / 4096⌉ = ⌈4⌉ = 4 blocks
Logic: ~2000 LCs
```

**Convolution Layer:**

```
DSPs: K² (for parallel kernel computation)
BRAM: Line buffers = ⌈(K-1) · Width · bits/element / BRAM_size⌉
```

**Example (3×3 kernel, 128-width image, 8-bit):**

```
DSPs: 9 (or reuse fewer with time-multiplexing)
BRAM: ⌈2 · 128 · 8 / 4096⌉ = ⌈1⌉ = 1 block
```

### 7.8 Scaling Laws

**Performance vs Bit-Width:**

For quantized networks:

```
Performance_relative = (8 / bit_width)  (memory-bound)
Performance_relative = (8 / bit_width)² (compute-bound with reduced precision)
```

**Accuracy vs Quantization:**

Empirical relationship:

```
Accuracy_loss ≈ k · 2^(-bit_width)
```

Where **k** depends on network and task (~0.01-0.1 for ImageNet).

**Energy vs Bit-Width:**

```
E ∝ bit_width²  (for multipliers)
E ∝ bit_width   (for memory)
```

**Example:**

8-bit → 4-bit:

```
Energy_mult: 4x reduction
Energy_mem: 2x reduction
Performance: 2x improvement (memory-bound)
```

---

## Summary: Key Equations Reference

### Neural Network Operations

| Operation | Equation | Complexity |
|-----------|----------|------------|
| Matrix Multiply | Y = XW + b | O(nmk) |
| Convolution | Y[i,j] = Σ X[i+m,j+n]·W[m,n] | O(HWC_outC_inK²) |
| ReLU | f(x) = max(0, x) | O(1) |
| Sigmoid | σ(x) = 1/(1+e^(-x)) | O(1)* |

### Quantization

| Concept | Equation |
|---------|----------|
| Fixed-Point | x_fixed = round(x · 2^n) |
| SQNR | SQNR_dB ≈ 6.02n + 1.76 |
| Symmetric Scale | s = α / (2^(b-1) - 1) |
| Asymmetric Scale | s = (α - β) / (2^b - 1) |

### Hardware Performance

| Metric | Equation |
|--------|----------|
| Roofline | Perf = min(Peak_Compute, AI · BW) |
| Amdahl's Law | Speedup = 1 / (S + P/N) |
| Energy/Op | E = α · C · V² |
| Throughput | Throughput = Freq · Parallelism / Latency |

### Stochastic Computing

| Operation | Implementation |
|-----------|----------------|
| Multiplication | AND gate (unipolar) |
| Addition | MUX with weighted select |
| Precision | σ = √(x(1-x)/N) |

---

## Recommendations for UPduino Implementation

1. **Use 8-bit quantization** for balance of accuracy and efficiency
2. **Implement systolic arrays** for matrix operations (maximize DSP utilization)
3. **Employ weight sparsity** to reduce memory bandwidth
4. **Use piecewise linear approximations** for activation functions
5. **Consider stochastic computing** for ultra-low-power modes
6. **Optimize for compute-bound operations** (leverage roofline analysis)
7. **Implement double buffering** to hide memory latency
8. **Use block RAM for weights**, SPRAM for activations

---

## References

1. Hubara et al., "Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations" (2016)
2. Rastegari et al., "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks" (2016)
3. Gaines, "Stochastic Computing Systems" (1969)
4. Kung, "Systolic Algorithms and Arrays" (1988)
5. Williams et al., "Roofline: An Insightful Visual Performance Model" (2009)
6. Shafiee et al., "ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars" (2016)

---

**Document Status:** Complete
**Next Steps:** Create accompanying Python notebook with numerical examples
**Coordination:** Ready for RTL design agent handoff
