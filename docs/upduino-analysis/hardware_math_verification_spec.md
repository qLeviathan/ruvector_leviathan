# Hardware Mathematical Verification Specification
# UPduino v3.0 AI Accelerator - Exact Verification Insertion Points

**Version:** 1.0
**Date:** 2026-01-05
**Target:** UPduino v3.1 (iCE40 UP5K FPGA)
**Purpose:** Define exact mathematical verification points for hardware testing

---

## Table of Contents

1. [Overview](#1-overview)
2. [Verification Pipeline](#2-verification-pipeline)
3. [Stage-by-Stage Verification](#3-stage-by-stage-verification)
4. [Test Vector Generation](#4-test-vector-generation)
5. [Tolerance Specifications](#5-tolerance-specifications)
6. [Implementation Guide](#6-implementation-guide)

---

## 1. Overview

### 1.1 Verification Philosophy

**CRITICAL PRINCIPLE:** Verify mathematical correctness at EVERY data transformation point.

```
Hardware Pipeline:
[Input] → [Quantize] → [Memory Load] → [MAC] → [Accumulate] → [Activation] → [De-quantize] → [Output]
   ↓          ↓             ↓            ↓          ↓               ↓              ↓            ↓
[Check 0]  [Check 1]    [Check 2]    [Check 3]  [Check 4]      [Check 5]     [Check 6]   [Check 7]
```

### 1.2 Verification Types

| Type | Description | When to Use |
|------|-------------|-------------|
| **Bit-Exact** | Exact binary match | Integer operations, quantization |
| **ULP Tolerance** | Units in Last Place | Floating-point comparisons |
| **Relative Error** | Percentage error bound | MAC operations, accumulation |
| **Statistical** | Distribution matching | Large batch processing |

---

## 2. Verification Pipeline

### 2.1 Complete Verification Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    VERIFICATION PIPELINE                        │
└────────────────────────────────────────────────────────────────┘

Stage 0: INPUT VALIDATION
├─ Check: Input data range [−128, 127] for INT8
├─ Check: No NaN or Inf values in FP32 source
├─ Check: Data shape matches expected dimensions
└─ Golden: input_fp32 = load_test_vector()

Stage 1: QUANTIZATION
├─ Hardware: Q_hw = quantize_int8(input_fp32)
├─ Golden:   Q_sw = software_quantize(input_fp32)
├─ Check:    abs(Q_hw - Q_sw) ≤ 1  (bit-exact or ±1 for rounding)
├─ Check:    verify_saturation(Q_hw, -128, 127)
└─ Check:    verify_rounding_mode(Q_hw, Q_sw)

Stage 2: MEMORY LOAD (Weight Loading)
├─ Hardware: W_hw = mem_read(addr)
├─ Golden:   W_sw = weight_table[addr]
├─ Check:    W_hw === W_sw  (bit-exact)
├─ Check:    verify_no_bit_errors(W_hw, W_sw)
└─ Check:    verify_address_mapping(addr)

Stage 3: MULTIPLY-ACCUMULATE (MAC)
├─ Hardware: P_hw = weight_reg × activation_in
├─ Golden:   P_sw = int16(weight) × int16(activation)
├─ Check:    P_hw === P_sw  (bit-exact for INT8×INT8→INT16)
├─ Check:    verify_sign_extension(P_hw)
└─ Check:    verify_no_overflow(P_hw, INT16_MIN, INT16_MAX)

Stage 4: ACCUMULATION
├─ Hardware: ACC_hw = ACC_prev + MAC_result
├─ Golden:   ACC_sw = sum(MAC_results[0:k])
├─ Check:    abs(ACC_hw - ACC_sw) ≤ K  (K = number of MACs, for rounding)
├─ Check:    verify_overflow_handling(ACC_hw)
└─ Check:    verify_accumulator_width(ACC_hw, ACC_WIDTH)

Stage 5: ACTIVATION FUNCTION
├─ Hardware: ACT_hw = activation_fn(ACC_hw)
├─ Golden:   ACT_sw = golden_activation(ACC_sw)
├─ Check:    Depends on activation type (see below)
│            ReLU: bit-exact
│            Tanh: relative error ≤ 0.01 (1%)
│            Sigmoid: relative error ≤ 0.02 (2%)
└─ Check:    verify_activation_bounds(ACT_hw)

Stage 6: OUTPUT QUANTIZATION
├─ Hardware: OUT_hw = requantize(ACT_hw, scale, zero_point)
├─ Golden:   OUT_sw = software_requantize(ACT_sw, scale, zero_point)
├─ Check:    abs(OUT_hw - OUT_sw) ≤ 1  (±1 for rounding)
└─ Check:    verify_output_range(OUT_hw, OUT_MIN, OUT_MAX)

Stage 7: FINAL OUTPUT
├─ Hardware: Read output buffer
├─ Golden:   Expected output from software model
├─ Check:    Compare layer-by-layer outputs
├─ Check:    Calculate error metrics (MSE, MAE, max error)
└─ Check:    Verify classification accuracy (if applicable)
```

---

## 3. Stage-by-Stage Verification

### 3.1 STAGE 0: Input Validation

#### 3.1.1 Mathematical Operation
```
input_data ∈ ℝ^(H×W×C)
Range check: ∀ x ∈ input_data, x_min ≤ x ≤ x_max
```

#### 3.1.2 Golden Reference (Python)
```python
def verify_input(input_data, x_min=-128, x_max=127):
    """
    Verify input data is within valid range
    """
    # Check for invalid values
    assert not np.isnan(input_data).any(), "Input contains NaN"
    assert not np.isinf(input_data).any(), "Input contains Inf"

    # Check range
    assert np.all(input_data >= x_min), f"Input below minimum {x_min}"
    assert np.all(input_data <= x_max), f"Input above maximum {x_max}"

    # Check shape
    expected_shape = (28, 28, 1)  # MNIST example
    assert input_data.shape == expected_shape, f"Shape mismatch: {input_data.shape}"

    return True
```

#### 3.1.3 Test Vectors
```python
# Normal case
test_input_normal = np.random.randint(-128, 128, size=(28, 28, 1), dtype=np.int8)

# Edge cases
test_input_min = np.full((28, 28, 1), -128, dtype=np.int8)
test_input_max = np.full((28, 28, 1), 127, dtype=np.int8)
test_input_zero = np.zeros((28, 28, 1), dtype=np.int8)

# Invalid cases (should fail)
test_input_nan = np.array([[np.nan]])
test_input_inf = np.array([[np.inf]])
```

#### 3.1.4 Verilog Assertion
```verilog
// In testbench
always @(posedge clk) begin
    if (input_valid) begin
        // Range check
        assert (input_data >= -8'sd128 && input_data <= 8'sd127)
            else $error("Input data out of range: %d", input_data);

        // Shape check (implicit in port width)
        // No NaN check needed for integer types
    end
end
```

---

### 3.2 STAGE 1: Quantization

#### 3.2.1 Mathematical Operation

**Symmetric Quantization (Zero-point = 0):**
```
Scale: s = max(|x_min|, |x_max|) / (2^(b-1) - 1)
Quantize: x_q = round(x_fp / s)
Saturate: x_q_sat = clip(x_q, -2^(b-1), 2^(b-1) - 1)
```

**Example for INT8 (b=8):**
```
s = max(|x|) / 127
x_q = round(x_fp / s)
x_q_sat = clip(x_q, -128, 127)
```

**Asymmetric Quantization:**
```
Scale: s = (x_max - x_min) / (2^b - 1)
Zero-point: z = round(-x_min / s)
Quantize: x_q = round(x_fp / s) + z
Saturate: x_q_sat = clip(x_q, 0, 2^b - 1)
```

#### 3.2.2 Golden Reference (Python)
```python
def quantize_symmetric_int8(x_fp32, scale=None):
    """
    Symmetric quantization to INT8 with zero-point = 0

    Args:
        x_fp32: Input in FP32
        scale: Quantization scale (computed if None)

    Returns:
        x_int8: Quantized INT8 value
        scale: Scale factor used
    """
    # Compute scale if not provided
    if scale is None:
        x_max = np.max(np.abs(x_fp32))
        scale = x_max / 127.0
        scale = max(scale, 1e-8)  # Avoid division by zero

    # Quantize
    x_q = np.round(x_fp32 / scale)

    # Saturate to INT8 range
    x_q_sat = np.clip(x_q, -128, 127).astype(np.int8)

    return x_q_sat, scale

def verify_quantization(x_hw, x_sw, tolerance=1):
    """
    Verify hardware quantization matches software

    Tolerance = 1 allows for different rounding modes
    """
    error = np.abs(x_hw - x_sw)
    max_error = np.max(error)

    assert max_error <= tolerance, \
        f"Quantization error {max_error} exceeds tolerance {tolerance}"

    # Check saturation
    assert np.all(x_hw >= -128) and np.all(x_hw <= 127), \
        "Hardware quantization not properly saturated"

    return True
```

#### 3.2.3 Tolerance Specification

| Scenario | Tolerance | Justification |
|----------|-----------|---------------|
| **Exact Rounding** | 0 (bit-exact) | Hardware uses same rounding as software |
| **Different Rounding** | ±1 | Round-to-nearest vs round-to-zero |
| **Saturation** | 0 (bit-exact) | Must saturate to exact limits |

#### 3.2.4 Test Vectors
```python
# Test case 1: Values requiring no saturation
test_vec_1 = {
    'input_fp32': np.array([0.0, 1.0, -1.0, 0.5, -0.5]),
    'scale': 1.0 / 127.0,
    'expected_int8': np.array([0, 127, -127, 64, -64], dtype=np.int8)
}

# Test case 2: Values requiring saturation
test_vec_2 = {
    'input_fp32': np.array([2.0, -2.0, 10.0, -10.0]),
    'scale': 1.0 / 127.0,
    'expected_int8': np.array([127, -127, 127, -128], dtype=np.int8)  # Saturated
}

# Test case 3: Near-zero values
test_vec_3 = {
    'input_fp32': np.array([0.001, -0.001, 0.0]),
    'scale': 1.0 / 127.0,
    'expected_int8': np.array([0, 0, 0], dtype=np.int8)  # Quantized to zero
}

# Test case 4: Rounding boundary
test_vec_4 = {
    'input_fp32': np.array([0.504, 0.496]),
    'scale': 1.0 / 127.0,
    'expected_int8_round_nearest': np.array([64, 63], dtype=np.int8),
    'expected_int8_round_zero': np.array([63, 63], dtype=np.int8)
}
```

#### 3.2.5 Verilog Implementation & Assertions
```verilog
// Quantization module (simplified)
module quantizer #(
    parameter DATA_WIDTH = 16,
    parameter OUT_WIDTH = 8
)(
    input  wire signed [DATA_WIDTH-1:0] data_in,  // FP representation (scaled)
    output wire signed [OUT_WIDTH-1:0]  data_out
);
    // Scale down (right shift)
    localparam SHIFT = DATA_WIDTH - OUT_WIDTH;
    wire signed [DATA_WIDTH-1:0] shifted;

    // Rounding: add 0.5 before truncation
    wire signed [DATA_WIDTH-1:0] rounded;
    assign rounded = data_in + (1 << (SHIFT-1));
    assign shifted = rounded >>> SHIFT;

    // Saturation
    wire signed [OUT_WIDTH-1:0] saturated;
    localparam signed [DATA_WIDTH-1:0] MAX_VAL = (1 << (OUT_WIDTH-1)) - 1;
    localparam signed [DATA_WIDTH-1:0] MIN_VAL = -(1 << (OUT_WIDTH-1));

    assign saturated = (shifted > MAX_VAL) ? MAX_VAL[OUT_WIDTH-1:0] :
                       (shifted < MIN_VAL) ? MIN_VAL[OUT_WIDTH-1:0] :
                       shifted[OUT_WIDTH-1:0];

    assign data_out = saturated;

    // ASSERTIONS
    // 1. Output must be within valid range
    always @(*) begin
        assert (data_out >= MIN_VAL[OUT_WIDTH-1:0])
            else $error("Quantized value below minimum");
        assert (data_out <= MAX_VAL[OUT_WIDTH-1:0])
            else $error("Quantized value above maximum");
    end

    // 2. If input within range, no saturation should occur
    property no_unnecessary_saturation;
        @(posedge clk)
        (shifted >= MIN_VAL && shifted <= MAX_VAL) |-> (data_out == shifted[OUT_WIDTH-1:0]);
    endproperty
    assert property (no_unnecessary_saturation);

    // 3. If input > MAX, must saturate to MAX
    property saturate_high;
        @(posedge clk)
        (shifted > MAX_VAL) |-> (data_out == MAX_VAL[OUT_WIDTH-1:0]);
    endproperty
    assert property (saturate_high);

    // 4. If input < MIN, must saturate to MIN
    property saturate_low;
        @(posedge clk)
        (shifted < MIN_VAL) |-> (data_out == MIN_VAL[OUT_WIDTH-1:0]);
    endproperty
    assert property (saturate_low);
endmodule
```

---

### 3.3 STAGE 2: Memory Load (Weight Loading)

#### 3.3.1 Mathematical Operation
```
Address mapping: addr_linear = i × W + j
Memory read: weight[addr_linear] → weight_reg
Verification: weight_hw === weight_expected (bit-exact)
```

#### 3.3.2 Golden Reference (Python)
```python
def verify_weight_load(weight_hw, weight_table, addr):
    """
    Verify hardware loaded correct weight from memory

    Args:
        weight_hw: Weight value read by hardware
        weight_table: Expected weight values (numpy array)
        addr: Memory address

    Returns:
        True if match, raises AssertionError otherwise
    """
    weight_expected = weight_table[addr]

    # Bit-exact match required
    assert weight_hw == weight_expected, \
        f"Weight mismatch at addr {addr}: hw={weight_hw}, expected={weight_expected}"

    return True

def verify_address_mapping(row, col, width, addr_hw):
    """
    Verify address calculation is correct
    """
    addr_expected = row * width + col
    assert addr_hw == addr_expected, \
        f"Address mismatch: hw={addr_hw}, expected={addr_expected}"
    return True
```

#### 3.3.3 Test Vectors
```python
# Test weight table (4x4 example)
weight_table = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120],
    [127, -128, 0, 1]
], dtype=np.int8).flatten()

# Test cases
test_cases_memory = [
    {'row': 0, 'col': 0, 'addr': 0, 'expected_weight': 10},
    {'row': 0, 'col': 3, 'addr': 3, 'expected_weight': 40},
    {'row': 2, 'col': 1, 'addr': 9, 'expected_weight': 100},
    {'row': 3, 'col': 0, 'addr': 12, 'expected_weight': 127},
    {'row': 3, 'col': 1, 'addr': 13, 'expected_weight': -128},
    {'row': 3, 'col': 2, 'addr': 14, 'expected_weight': 0},
]
```

#### 3.3.4 Verilog Assertions
```verilog
// In memory controller testbench
initial begin
    // Load expected weights
    $readmemh("weights_expected.hex", weight_table_golden);
end

always @(posedge clk) begin
    if (weight_load) begin
        // Check address calculation
        automatic int expected_addr = row_index * MATRIX_WIDTH + col_index;
        assert (weight_addr == expected_addr)
            else $error("Address mismatch: %d != %d", weight_addr, expected_addr);

        // Check loaded weight value (bit-exact)
        #1;  // Wait for memory read
        assert (weight_out === weight_table_golden[weight_addr])
            else $error("Weight mismatch at addr %d: %d != %d",
                        weight_addr, weight_out, weight_table_golden[weight_addr]);
    end
end

// Check for bit errors (flip detection)
always @(posedge clk) begin
    if (weight_valid) begin
        automatic logic [7:0] expected = weight_table_golden[weight_addr_prev];
        automatic int hamming_distance = $countones(weight_out ^ expected);

        assert (hamming_distance == 0)
            else $error("Bit error detected: %d bits differ", hamming_distance);
    end
end
```

---

### 3.4 STAGE 3: Multiply-Accumulate (MAC)

#### 3.4.1 Mathematical Operation

**INT8 × INT8 → INT16 Multiplication:**
```
weight ∈ [-128, 127]  (INT8)
activation ∈ [-128, 127]  (INT8)
product = weight × activation ∈ [-16384, 16129]  (INT16)
```

**With Partial Sum:**
```
mac_result = partial_sum_in + (weight × activation)
where partial_sum_in ∈ INT16 range
```

**Overflow Condition:**
```
Overflow if:
  mac_result > 32767  (INT16_MAX)
  mac_result < -32768  (INT16_MIN)
```

#### 3.4.2 Golden Reference (Python)
```python
def mac_golden(weight, activation, partial_sum=0):
    """
    Golden reference for MAC operation

    Args:
        weight: INT8 weight value
        activation: INT8 activation value
        partial_sum: INT16 partial sum

    Returns:
        mac_result: INT16 MAC result
        overflow_flag: True if overflow occurred
    """
    # Ensure correct types
    weight = np.int16(weight)  # Sign-extend to INT16
    activation = np.int16(activation)
    partial_sum = np.int16(partial_sum)

    # Multiply
    product = np.int32(weight) * np.int32(activation)  # Use INT32 to detect overflow

    # Add partial sum
    mac_result_32 = product + np.int32(partial_sum)

    # Check for overflow
    overflow = (mac_result_32 > 32767) or (mac_result_32 < -32768)

    # Saturate if overflow
    if overflow:
        mac_result = np.int16(32767) if mac_result_32 > 0 else np.int16(-32768)
    else:
        mac_result = np.int16(mac_result_32)

    return mac_result, overflow

def verify_mac(mac_hw, weight, activation, partial_sum):
    """
    Verify hardware MAC matches golden reference
    """
    mac_sw, overflow_expected = mac_golden(weight, activation, partial_sum)

    # Bit-exact match required for integer MAC
    assert mac_hw == mac_sw, \
        f"MAC mismatch: hw={mac_hw}, sw={mac_sw}, " \
        f"w={weight}, a={activation}, ps={partial_sum}"

    return True
```

#### 3.4.3 Test Vectors
```python
# Test case 1: Normal operation (no overflow)
test_mac_1 = {
    'weight': 10,
    'activation': 20,
    'partial_sum': 100,
    'expected_result': 300,  # 10*20 + 100
    'overflow': False
}

# Test case 2: Maximum positive product
test_mac_2 = {
    'weight': 127,
    'activation': 127,
    'partial_sum': 0,
    'expected_result': 16129,  # 127*127
    'overflow': False
}

# Test case 3: Maximum negative product
test_mac_3 = {
    'weight': -128,
    'activation': 128,  # Note: 128 is out of INT8 range, use 127
    'activation': 127,
    'partial_sum': 0,
    'expected_result': -16256,  # -128*127
    'overflow': False
}

# Test case 4: Overflow positive
test_mac_4 = {
    'weight': 100,
    'activation': 100,
    'partial_sum': 30000,
    'expected_result': 32767,  # Saturated (10000 + 30000 = 40000 > 32767)
    'overflow': True
}

# Test case 5: Overflow negative
test_mac_5 = {
    'weight': -100,
    'activation': 100,
    'partial_sum': -30000,
    'expected_result': -32768,  # Saturated (-10000 - 30000 = -40000 < -32768)
    'overflow': True
}

# Test case 6: Zero handling
test_mac_6 = {
    'weight': 0,
    'activation': 123,
    'partial_sum': 456,
    'expected_result': 456,  # 0*123 + 456
    'overflow': False
}

# Test case 7: Sign extension verification
test_mac_7 = {
    'weight': -1,  # 0xFF in 8-bit → 0xFFFF in 16-bit (sign-extended)
    'activation': -1,
    'partial_sum': 0,
    'expected_result': 1,  # (-1)*(-1) = 1
    'overflow': False
}
```

#### 3.4.4 Verilog Implementation & Assertions
```verilog
// From processing_element.v (with added assertions)
module processing_element_verified #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [DATA_WIDTH-1:0]    weight_in,
    input  wire [DATA_WIDTH-1:0]    activation_in,
    input  wire [ACC_WIDTH-1:0]     partial_sum_in,
    input  wire                     weight_load,
    input  wire                     accumulate,
    output reg  [ACC_WIDTH-1:0]     partial_sum_out
);

    reg signed [DATA_WIDTH-1:0] weight_reg;

    // MAC computation
    wire signed [DATA_WIDTH-1:0] act_signed = activation_in;
    wire signed [DATA_WIDTH-1:0] weight_signed = weight_reg;
    wire signed [2*DATA_WIDTH-1:0] product = act_signed * weight_signed;
    wire signed [ACC_WIDTH-1:0] mac_result = partial_sum_in +
        {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};

    // Weight register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            weight_reg <= 0;
        else if (weight_load)
            weight_reg <= weight_in;
    end

    // Output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            partial_sum_out <= 0;
        else if (accumulate)
            partial_sum_out <= mac_result;
        else
            partial_sum_out <= partial_sum_in;
    end

    // ========================================================================
    // VERIFICATION ASSERTIONS
    // ========================================================================

    // Assertion 1: Product range check
    // INT8 × INT8 must fit in INT16
    property product_range;
        @(posedge clk)
        (product >= -17'd16384) && (product <= 17'd16129);
    endproperty
    assert property (product_range)
        else $error("Product out of expected range: %d", product);

    // Assertion 2: Sign extension verification
    // Upper bits of product must match sign bit
    property sign_extension;
        @(posedge clk)
        (product[2*DATA_WIDTH-1] == 1'b1) |->
            (&product[ACC_WIDTH-1:2*DATA_WIDTH] == 1'b1);
    endproperty
    assert property (sign_extension)
        else $error("Incorrect sign extension in MAC");

    // Assertion 3: MAC result matches expected (for testbench)
    `ifdef VERIFICATION
    wire signed [ACC_WIDTH-1:0] expected_mac_result;
    assign expected_mac_result = tb_expected_value;  // From testbench

    always @(posedge clk) begin
        if (accumulate && tb_check_enable) begin
            assert (mac_result === expected_mac_result)
                else $error("MAC mismatch: hw=%d, expected=%d, w=%d, a=%d, ps=%d",
                            mac_result, expected_mac_result,
                            weight_reg, activation_in, partial_sum_in);
        end
    end
    `endif

    // Assertion 4: Overflow detection (if saturation logic added)
    `ifdef SATURATE_MAC
    wire overflow_pos = (partial_sum_in[ACC_WIDTH-1] == 0) &&
                        (product[2*DATA_WIDTH-1] == 0) &&
                        (mac_result[ACC_WIDTH-1] == 1);
    wire overflow_neg = (partial_sum_in[ACC_WIDTH-1] == 1) &&
                        (product[2*DATA_WIDTH-1] == 1) &&
                        (mac_result[ACC_WIDTH-1] == 0);

    always @(posedge clk) begin
        if (overflow_pos)
            $warning("Positive overflow detected in MAC");
        if (overflow_neg)
            $warning("Negative overflow detected in MAC");
    end
    `endif

    // Assertion 5: Zero weight produces zero product
    property zero_weight;
        @(posedge clk)
        (weight_reg == 0) |-> (product == 0);
    endproperty
    assert property (zero_weight);

endmodule
```

---

### 3.5 STAGE 4: Accumulation

#### 3.5.1 Mathematical Operation

**Sequential Accumulation:**
```
ACC[0] = 0  (initial)
ACC[k] = ACC[k-1] + MAC[k]  for k = 1, 2, ..., N

Final accumulator = Σ(i=0 to N-1) MAC[i]
```

**Potential Error Sources:**
1. Rounding accumulation over multiple additions
2. Overflow in accumulator
3. Non-deterministic order (parallel reduction)

#### 3.5.2 Golden Reference (Python)
```python
def accumulate_golden(mac_results):
    """
    Golden reference for sequential accumulation

    Args:
        mac_results: List of MAC results (INT16)

    Returns:
        final_accumulator: Final accumulated value
        overflow_occurred: True if overflow detected
    """
    acc = np.int32(0)  # Use INT32 to detect overflow
    overflow_occurred = False

    for mac in mac_results:
        acc += np.int32(mac)

        # Check for overflow
        if acc > 32767 or acc < -32768:
            overflow_occurred = True

    # Saturate final result to INT16
    if acc > 32767:
        final_acc = np.int16(32767)
    elif acc < -32768:
        final_acc = np.int16(-32768)
    else:
        final_acc = np.int16(acc)

    return final_acc, overflow_occurred

def verify_accumulation(acc_hw, mac_results, tolerance=0):
    """
    Verify hardware accumulation

    For integer accumulation, should be bit-exact (tolerance=0)
    For floating-point, allow small tolerance due to rounding
    """
    acc_sw, overflow = accumulate_golden(mac_results)

    error = abs(int(acc_hw) - int(acc_sw))
    assert error <= tolerance, \
        f"Accumulation error {error} exceeds tolerance {tolerance}"

    return True
```

#### 3.5.3 Tolerance Specification

| Accumulation Type | Tolerance | Notes |
|-------------------|-----------|-------|
| **INT16 Sequential** | 0 (bit-exact) | Deterministic order |
| **INT16 Parallel** | 0 (bit-exact) | If order is fixed |
| **INT32 Sequential** | 0 (bit-exact) | Wider accumulator |
| **FP16 Sequential** | ≤ N × ULP | N = number of additions |

#### 3.5.4 Test Vectors
```python
# Test case 1: Simple accumulation (no overflow)
test_acc_1 = {
    'mac_results': [100, 200, 300, 400, 500],
    'expected_acc': 1500,
    'overflow': False
}

# Test case 2: Accumulation with overflow
test_acc_2 = {
    'mac_results': [10000, 10000, 10000, 10000],
    'expected_acc': 32767,  # Saturated (40000 > 32767)
    'overflow': True
}

# Test case 3: Mixed positive and negative
test_acc_3 = {
    'mac_results': [1000, -500, 300, -200, 100],
    'expected_acc': 700,
    'overflow': False
}

# Test case 4: Large number of small values
test_acc_4 = {
    'mac_results': [1] * 1000,  # 1000 MACs of value 1
    'expected_acc': 1000,
    'overflow': False
}

# Test case 5: Alternating signs (cancellation)
test_acc_5 = {
    'mac_results': [100, -100] * 50,  # 100 values that cancel
    'expected_acc': 0,
    'overflow': False
}
```

#### 3.5.6 Verilog Assertions
```verilog
// Accumulator verification
module accumulator_verified #(
    parameter ACC_WIDTH = 16,
    parameter NUM_MACS = 64
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire signed [ACC_WIDTH-1:0] mac_in,
    input  wire                     accumulate,
    input  wire                     clear,
    output reg signed [ACC_WIDTH-1:0]  acc_out
);

    reg signed [ACC_WIDTH-1:0] accumulator;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            accumulator <= 0;
        else if (clear)
            accumulator <= 0;
        else if (accumulate)
            accumulator <= accumulator + mac_in;
    end

    assign acc_out = accumulator;

    // ========================================================================
    // VERIFICATION
    // ========================================================================

    `ifdef VERIFICATION
    // Track accumulation history
    integer mac_count;
    reg signed [ACC_WIDTH-1:0] mac_history[0:NUM_MACS-1];

    always @(posedge clk) begin
        if (clear) begin
            mac_count <= 0;
        end else if (accumulate) begin
            mac_history[mac_count] <= mac_in;
            mac_count <= mac_count + 1;
        end
    end

    // Verify final accumulation
    task verify_final_accumulation;
        integer i;
        reg signed [31:0] expected_acc;  // Use INT32 for checking
        begin
            expected_acc = 0;
            for (i = 0; i < mac_count; i = i + 1) begin
                expected_acc = expected_acc + mac_history[i];
            end

            // Saturate to ACC_WIDTH
            if (expected_acc > ((1 << (ACC_WIDTH-1)) - 1))
                expected_acc = (1 << (ACC_WIDTH-1)) - 1;
            else if (expected_acc < -(1 << (ACC_WIDTH-1)))
                expected_acc = -(1 << (ACC_WIDTH-1));

            if (accumulator !== expected_acc[ACC_WIDTH-1:0]) begin
                $error("Accumulation mismatch: hw=%d, expected=%d, num_macs=%d",
                       accumulator, expected_acc[ACC_WIDTH-1:0], mac_count);
            end else begin
                $display("Accumulation verified: %d MACs, result=%d",
                         mac_count, accumulator);
            end
        end
    endtask
    `endif

    // Overflow detection
    wire overflow_pos = (accumulator[ACC_WIDTH-1] == 0) &&
                        (mac_in[ACC_WIDTH-1] == 0) &&
                        ((accumulator + mac_in) < accumulator);
    wire overflow_neg = (accumulator[ACC_WIDTH-1] == 1) &&
                        (mac_in[ACC_WIDTH-1] == 1) &&
                        ((accumulator + mac_in) > accumulator);

    always @(posedge clk) begin
        if (accumulate) begin
            if (overflow_pos)
                $warning("Positive overflow in accumulation");
            if (overflow_neg)
                $warning("Negative overflow in accumulation");
        end
    end

endmodule
```

---

### 3.6 STAGE 5: Activation Functions

#### 3.6.1 ReLU Activation

**Mathematical Operation:**
```
ReLU(x) = max(0, x) = {
    x,  if x > 0
    0,  if x ≤ 0
}
```

**Golden Reference (Python):**
```python
def relu_golden(x):
    """ReLU activation - bit-exact for integers"""
    return np.maximum(0, x)

def verify_relu(relu_hw, input_val):
    """Verify ReLU - must be bit-exact"""
    relu_sw = relu_golden(input_val)
    assert relu_hw == relu_sw, \
        f"ReLU mismatch: hw={relu_hw}, sw={relu_sw}, input={input_val}"
    return True
```

**Test Vectors:**
```python
test_relu = [
    {'input': 100, 'expected': 100},
    {'input': -100, 'expected': 0},
    {'input': 0, 'expected': 0},
    {'input': 1, 'expected': 1},
    {'input': -1, 'expected': 0},
    {'input': 32767, 'expected': 32767},
    {'input': -32768, 'expected': 0},
]
```

**Verilog Assertions:**
```verilog
// ReLU verification
always @(*) begin
    if (activation_type == ACT_RELU) begin
        // Positive values pass through
        if (data_in >= 0) begin
            assert (data_out == data_in)
                else $error("ReLU failed for positive input");
        end
        // Negative values become zero
        else begin
            assert (data_out == 0)
                else $error("ReLU failed for negative input");
        end
    end
end
```

#### 3.6.2 Tanh Activation (Piecewise Linear Approximation)

**Mathematical Operation:**
```
tanh_approx(x) = {
    -1.0,       if x < -threshold
    x / scale,  if -threshold ≤ x ≤ threshold
    +1.0,       if x > threshold
}

where threshold and scale depend on fixed-point format
```

**Golden Reference (Python):**
```python
def tanh_pwl_golden(x, threshold=2.0, scale=2.0):
    """
    Piecewise linear tanh approximation

    Args:
        x: Input value (can be array)
        threshold: Saturation threshold
        scale: Linear region scale factor

    Returns:
        tanh approximation
    """
    y = np.zeros_like(x, dtype=np.float32)

    # Saturate low
    mask_low = x < -threshold
    y[mask_low] = -1.0

    # Linear region
    mask_mid = np.abs(x) <= threshold
    y[mask_mid] = x[mask_mid] / scale

    # Saturate high
    mask_high = x > threshold
    y[mask_high] = 1.0

    return y

def verify_tanh_pwl(tanh_hw, input_val, threshold=2.0, scale=2.0, tolerance=0.01):
    """
    Verify piecewise linear tanh

    Tolerance = 0.01 allows for 1% relative error due to quantization
    """
    tanh_sw = tanh_pwl_golden(input_val, threshold, scale)

    # For saturated regions, should be exact
    if abs(input_val) > threshold:
        expected = -1.0 if input_val < 0 else 1.0
        assert abs(tanh_hw - expected) < 1e-6, \
            f"Tanh saturation error: hw={tanh_hw}, expected={expected}"
    else:
        # Linear region: allow small tolerance
        rel_error = abs(tanh_hw - tanh_sw) / (abs(tanh_sw) + 1e-8)
        assert rel_error <= tolerance, \
            f"Tanh error {rel_error} exceeds tolerance {tolerance}"

    return True
```

**Test Vectors:**
```python
test_tanh = [
    # Saturated regions (should be exact)
    {'input': -10.0, 'expected': -1.0, 'tolerance': 0.001},
    {'input': 10.0, 'expected': 1.0, 'tolerance': 0.001},

    # Linear region (allow tolerance)
    {'input': 0.0, 'expected': 0.0, 'tolerance': 0.01},
    {'input': 1.0, 'expected': 0.5, 'tolerance': 0.01},
    {'input': -1.0, 'expected': -0.5, 'tolerance': 0.01},

    # Boundaries
    {'input': 2.0, 'expected': 1.0, 'tolerance': 0.01},
    {'input': -2.0, 'expected': -1.0, 'tolerance': 0.01},
]
```

---

### 3.7 STAGE 6: Output Quantization & Stage 7: Final Output

**See full specification document for complete details on these stages**

---

## 4. Test Vector Generation

### 4.1 Generation Strategy

```python
class TestVectorGenerator:
    """Comprehensive test vector generation"""

    def generate_all_test_vectors(self):
        """Generate complete test suite"""
        test_vectors = {}

        # 1. Corner cases
        test_vectors['corners'] = self.generate_corner_cases()

        # 2. Random cases
        test_vectors['random'] = self.generate_random_cases(count=1000)

        # 3. Known-answer tests
        test_vectors['known'] = self.generate_known_answer_tests()

        # 4. Adversarial cases
        test_vectors['adversarial'] = self.generate_adversarial_cases()

        # 5. Edge cases
        test_vectors['edges'] = self.generate_edge_cases()

        return test_vectors

    def generate_corner_cases(self):
        """Generate corner case test vectors"""
        return [
            # All zeros
            {'type': 'all_zero', 'input': np.zeros((28, 28), dtype=np.int8)},

            # All maximum
            {'type': 'all_max', 'input': np.full((28, 28), 127, dtype=np.int8)},

            # All minimum
            {'type': 'all_min', 'input': np.full((28, 28), -128, dtype=np.int8)},

            # Checkerboard pattern
            {'type': 'checkerboard',
             'input': self.checkerboard_pattern(28, 28, val1=127, val2=-128)},

            # Single hot pixel
            {'type': 'single_hot',
             'input': self.single_hot_pattern(28, 28, row=14, col=14, val=127)},
        ]

    def generate_random_cases(self, count=1000):
        """Generate random test vectors with various distributions"""
        cases = []

        for i in range(count):
            # Vary distribution
            if i % 3 == 0:
                dist = 'uniform'
                data = np.random.randint(-128, 128, (28, 28), dtype=np.int8)
            elif i % 3 == 1:
                dist = 'gaussian'
                data = np.random.normal(0, 32, (28, 28)).astype(np.int8)
            else:
                dist = 'sparse'
                data = self.sparse_random(28, 28, sparsity=0.9)

            cases.append({'type': dist, 'input': data})

        return cases

    def generate_known_answer_tests(self):
        """Generate test vectors with pre-computed expected outputs"""
        # Example: Identity matrix should produce known output
        identity = np.eye(28, dtype=np.int8) * 127

        # Simple patterns with calculable outputs
        return [
            {'type': 'identity', 'input': identity, 'expected_output': self.compute_expected(identity)},
        ]
```

---

## 5. Tolerance Specifications

### 5.1 Summary Table

| Operation | Data Type | Tolerance | Verification Method |
|-----------|-----------|-----------|---------------------|
| **Input Validation** | INT8 | Bit-exact | Range check |
| **Quantization** | FP32→INT8 | ±1 LSB | Rounding difference |
| **Memory Load** | INT8 | Bit-exact | Direct comparison |
| **MAC (Multiply)** | INT8×INT8→INT16 | Bit-exact | Arithmetic check |
| **MAC (Accumulate)** | INT16 | Bit-exact | Sequential sum |
| **ReLU** | INT16 | Bit-exact | Conditional check |
| **Tanh (PWL)** | INT16 | ±1% relative | Piecewise verification |
| **Output Quantization** | INT16→INT8 | ±1 LSB | Rounding check |

---

## 6. Implementation Guide

### 6.1 Python Verification Module Structure

```python
# File: docs/upduino-analysis/test_scripts/math_verification.py

"""
Hardware Mathematical Verification Module
Provides golden reference implementations and verification functions
"""

class HardwareVerifier:
    """Main verification class"""

    def __init__(self, tolerance_config=None):
        self.tolerance = tolerance_config or self.default_tolerances()
        self.results = []

    def verify_complete_pipeline(self, fpga_output, test_input):
        """Verify entire inference pipeline"""
        pass  # See full implementation in separate file

    def verify_stage_by_stage(self, stage_name, hw_value, **kwargs):
        """Verify individual pipeline stage"""
        pass  # Implementation continues...
```

### 6.2 Verilog Testbench Structure

```verilog
// File: docs/upduino-analysis/rtl/verification_tb.v

module verification_tb;
    // Clock and reset
    reg clk, rst_n;

    // Test vectors
    reg [7:0] test_vectors[0:1000];
    reg [7:0] expected_outputs[0:1000];

    // DUT instantiation
    ai_accelerator dut (
        .clk(clk),
        .rst_n(rst_n),
        // ... ports
    );

    // Verification tasks
    `include "verification_checkers.v"

    initial begin
        // Run all verification tests
        run_all_tests();
    end

    // Test execution
    task run_all_tests;
        begin
            test_input_validation();
            test_quantization();
            test_memory_load();
            test_mac_operations();
            test_accumulation();
            test_activation_functions();
            test_output_stage();
        end
    endtask

endmodule
```

---

## 7. Usage Examples

### 7.1 Verify Single Stage (Python)

```python
from math_verification import HardwareVerifier

verifier = HardwareVerifier()

# Test quantization
input_fp32 = np.array([1.0, -0.5, 0.0, 2.0])
quantized_hw = fpga.quantize(input_fp32)  # Read from hardware

result = verifier.verify_stage_by_stage(
    stage_name='quantization',
    hw_value=quantized_hw,
    input_fp32=input_fp32,
    scale=1.0/127.0
)

if result['passed']:
    print("✓ Quantization verification passed")
else:
    print(f"✗ Quantization verification failed: {result['error']}")
```

### 7.2 Verify Complete Pipeline (Python)

```python
# Load test vectors
test_vectors = np.load('test_vectors.npy')
expected_outputs = np.load('expected_outputs.npy')

# Run verification
for i, test_vec in enumerate(test_vectors):
    fpga_output = fpga.infer(test_vec)

    result = verifier.verify_complete_pipeline(
        fpga_output=fpga_output,
        test_input=test_vec,
        expected_output=expected_outputs[i]
    )

    verifier.results.append(result)

# Generate report
report = verifier.generate_report()
print(report)
```

---

## Conclusion

This specification provides **exact insertion points** for mathematical verification at every stage of the hardware pipeline. Hardware engineers can implement these checks directly in:

1. **Verilog Assertions** - For real-time hardware verification
2. **Python Golden References** - For post-processing verification
3. **Testbenches** - For simulation-based verification

**Next Steps:**
1. Implement `math_verification.py` module (see separate file)
2. Add verification assertions to existing RTL
3. Generate comprehensive test vectors
4. Run verification suite and iterate

This ensures mathematical correctness at every stage of the AI accelerator hardware.
