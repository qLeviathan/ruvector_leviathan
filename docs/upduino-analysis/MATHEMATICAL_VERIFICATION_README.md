# Mathematical Verification Framework - Quick Start Guide

**UPduino v3.0 AI Accelerator Hardware Testing**

This guide provides everything you need to implement mathematical verification at every stage of the hardware pipeline.

---

## 📁 Files Overview

| File | Purpose | Usage |
|------|---------|-------|
| `hardware_math_verification_spec.md` | **Complete specification** | Read first - defines WHERE and HOW to insert checks |
| `test_scripts/math_verification.py` | **Python golden references** | Import and use for software verification |
| `rtl/verification_checkers.v` | **Verilog assertions** | Include in testbenches for hardware verification |

---

## 🚀 Quick Start

### 1. Python Verification (Software Golden Model)

```python
# Import the verification module
from test_scripts.math_verification import (
    HardwareVerifier,
    verify_quantization,
    verify_mac,
    verify_relu,
    TestVectorGenerator
)

# Create verifier
verifier = HardwareVerifier()

# Example 1: Verify quantization stage
import numpy as np

input_fp32 = np.array([1.0, -0.5, 0.0, 2.0])
quantized_hw = np.array([127, -64, 0, 127], dtype=np.int8)  # From FPGA
scale = 1.0 / 127.0

result = verify_quantization(quantized_hw, input_fp32, scale, tolerance=1)
print(f"Quantization: {'✓ PASS' if result['passed'] else '✗ FAIL'}")

# Example 2: Verify MAC operation
weight = np.int8(10)
activation = np.int8(20)
partial_sum = np.int16(100)
mac_hw = np.int16(300)  # From FPGA

result = verify_mac(mac_hw, weight, activation, partial_sum)
print(f"MAC: {'✓ PASS' if result['passed'] else '✗ FAIL'}")

# Example 3: Generate test vectors
gen = TestVectorGenerator(seed=42)
corner_cases = gen.generate_corner_cases(shape=(28, 28))
random_cases = gen.generate_random_cases(count=100, shape=(28, 28))

# Example 4: Complete pipeline verification
fpga_output = np.array([0.1, 0.05, 0.7, 0.15])  # From FPGA
expected_output = np.array([0.09, 0.06, 0.71, 0.14])  # From golden model
test_input = np.random.randn(28, 28)

result = verifier.verify_complete_pipeline(
    fpga_output=fpga_output,
    test_input=test_input,
    expected_output=expected_output
)

verifier.add_result(result)
print(verifier.generate_report())
```

**Run it:**
```bash
cd /home/user/ruvector_leviathan/docs/upduino-analysis
python3 test_scripts/math_verification.py
```

---

### 2. Verilog Testbench Integration

```verilog
// In your testbench file: ai_accelerator_tb.v

`timescale 1ns/1ps

module ai_accelerator_tb;
    // Include verification checkers
    `include "verification_checkers.v"

    // Clock and reset
    reg clk, rst_n;

    // Test signals
    reg signed [7:0] weight_val;
    reg signed [7:0] activation_val;
    reg signed [15:0] partial_sum;
    wire signed [15:0] mac_result;

    // DUT instantiation
    processing_element dut (
        .clk(clk),
        .rst_n(rst_n),
        .weight_in(weight_val),
        .activation_in(activation_val),
        .partial_sum_in(partial_sum),
        .partial_sum_out(mac_result),
        // ... other ports
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100 MHz clock
    end

    // Test execution
    initial begin
        rst_n = 0;
        #20 rst_n = 1;

        // Test case 1: Basic MAC
        @(posedge clk);
        weight_val = 8'sd10;
        activation_val = 8'sd20;
        partial_sum = 16'sd100;

        @(posedge clk);
        @(posedge clk);  // Wait for MAC computation

        // Verify MAC result
        verify_mac(weight_val, activation_val, partial_sum, mac_result);

        // Test case 2: Overflow condition
        @(posedge clk);
        weight_val = 8'sd127;
        activation_val = 8'sd127;
        partial_sum = 16'sd20000;

        @(posedge clk);
        @(posedge clk);

        verify_mac(weight_val, activation_val, partial_sum, mac_result);

        // Add more tests...

        #1000 $finish;
    end

endmodule
```

**Run simulation:**
```bash
# Using Icarus Verilog
iverilog -g2012 -o sim ai_accelerator_tb.v processing_element.v
vvp sim

# Using ModelSim
vlog -sv ai_accelerator_tb.v processing_element.v
vsim -do "run -all" work.ai_accelerator_tb
```

---

## 📊 Verification at Each Stage

### Stage 0: Input Validation

**Python:**
```python
from math_verification import verify_input

result = verify_input(
    input_data=test_image,
    x_min=-128,
    x_max=127,
    expected_shape=(28, 28, 1)
)
```

**Verilog:**
```verilog
always @(posedge clk) begin
    if (input_valid) begin
        check_input_range_int8(input_data, "input_pixel");
    end
end
```

---

### Stage 1: Quantization

**Python:**
```python
x_int8, scale = quantize_symmetric_int8(x_fp32)
result = verify_quantization(x_hw, x_fp32, scale, tolerance=1)
```

**Verilog:**
```verilog
always @(posedge clk) begin
    if (quant_valid) begin
        verify_quantization_saturation(data_in, data_out, SHIFT_AMOUNT);
        verify_rounding(data_in, data_out, SHIFT_AMOUNT);
    end
end
```

---

### Stage 2: Memory Load

**Python:**
```python
result = verify_weight_load(weight_hw, weight_table, addr)
result = verify_address_mapping(row, col, width, addr_hw)
```

**Verilog:**
```verilog
always @(posedge clk) begin
    if (weight_load) begin
        verify_address_calculation(row_idx, col_idx, MATRIX_WIDTH, weight_addr);
        #1 verify_memory_read(weight_out, weight_table_golden[weight_addr], weight_addr);
    end
end
```

---

### Stage 3: MAC (Multiply-Accumulate)

**Python:**
```python
mac_sw, overflow = mac_golden(weight, activation, partial_sum)
result = verify_mac(mac_hw, weight, activation, partial_sum)
```

**Verilog:**
```verilog
always @(posedge clk) begin
    if (mac_valid) begin
        verify_multiply(weight_reg, activation_in, product_internal);
        verify_mac(weight_reg, activation_in, partial_sum_in, mac_result);
        verify_sign_extension(product_internal, mac_result);
    end
end
```

---

### Stage 4: Accumulation

**Python:**
```python
acc_sw, overflow = accumulate_golden(mac_results)
result = verify_accumulation(acc_hw, mac_results, tolerance=0)
```

**Verilog:**
```verilog
always @(posedge clk) begin
    if (accumulate) begin
        verify_accumulation_step(accumulator_prev, mac_in, accumulator);
    end

    if (accumulation_done) begin
        verify_final_accumulation(accumulator, mac_count, mac_history);
    end
end
```

---

### Stage 5: Activation Functions

**Python:**
```python
# ReLU (bit-exact)
relu_sw = relu_golden(activation_in)
result = verify_relu(relu_hw, activation_in)

# Tanh PWL (1% tolerance)
tanh_sw = tanh_pwl_golden(activation_in, threshold=2.0, scale=2.0)
result = verify_tanh_pwl(tanh_hw, activation_in, tolerance=0.01)
```

**Verilog:**
```verilog
always @(posedge clk) begin
    case (activation_type)
        2'b01: verify_relu(data_in, data_out);
        2'b10: verify_tanh_pwl(data_in, data_out, THRESHOLD, MAX_VAL, MIN_VAL);
    endcase
end
```

---

## 🧪 Test Vector Generation

```python
from math_verification import TestVectorGenerator

gen = TestVectorGenerator(seed=42)

# 1. Corner cases
corners = gen.generate_corner_cases(shape=(28, 28))
# Returns: all-zero, all-max, all-min, checkerboard, single-hot

# 2. Random cases with distributions
random_tests = gen.generate_random_cases(count=1000, shape=(28, 28))
# Returns: uniform, gaussian, sparse distributions

# 3. Adversarial cases
adversarial = gen.generate_adversarial_cases(shape=(28, 28))
# Returns: alternating patterns, high-frequency patterns

# Save to files
import numpy as np
np.save('test_vectors.npy', [tv['input'] for tv in random_tests])
```

---

## 📐 Tolerance Specifications

| Operation | Tolerance | Type |
|-----------|-----------|------|
| Input Validation | 0 (bit-exact) | Range check |
| Quantization | ±1 LSB | Rounding difference |
| Memory Load | 0 (bit-exact) | Direct comparison |
| MAC Multiply | 0 (bit-exact) | Integer arithmetic |
| MAC Accumulate | 0 (bit-exact) | Integer sum |
| ReLU | 0 (bit-exact) | Conditional |
| Tanh PWL | ±1% relative | Approximation |
| Sigmoid PWL | ±2% relative | Approximation |

---

## 🔍 Error Metrics

**Python provides comprehensive error metrics:**

```python
result = verifier.verify_complete_pipeline(fpga_output, test_input, expected_output)

print(f"MSE: {result['overall_error']['mse']}")
print(f"MAE: {result['overall_error']['mae']}")
print(f"Max Error: {result['overall_error']['max_error']}")
print(f"Correlation: {result['overall_error']['correlation']}")
print(f"Classification Match: {result['classification']['match']}")
```

---

## 🎯 Common Test Cases

### Test Case 1: Known Answer Test
```python
# Identity matrix should produce predictable output
identity = np.eye(28, dtype=np.int8) * 127
expected_output = compute_expected_output(identity)  # Pre-calculated

fpga_output = fpga.infer(identity)
result = verifier.verify_complete_pipeline(fpga_output, identity, expected_output)
```

### Test Case 2: Overflow Detection
```python
# Large values to trigger overflow
overflow_input = np.full((28, 28), 127, dtype=np.int8)
fpga_output = fpga.infer(overflow_input)

# Check for proper saturation
assert np.all(fpga_output <= 127)
assert np.all(fpga_output >= -128)
```

### Test Case 3: Edge Cases
```python
edge_cases = [
    np.zeros((28, 28), dtype=np.int8),        # All zeros
    np.full((28, 28), 127, dtype=np.int8),    # All max
    np.full((28, 28), -128, dtype=np.int8),   # All min
]

for i, test_case in enumerate(edge_cases):
    fpga_output = fpga.infer(test_case)
    result = verifier.verify_complete_pipeline(fpga_output, test_case)
    print(f"Edge case {i}: {'✓ PASS' if result['passed'] else '✗ FAIL'}")
```

---

## 📝 Integration with Testing Framework

### Step 1: Generate Test Vectors
```bash
python3 test_scripts/generate_test_vectors.py --count 1000 --output test_vectors.npy
```

### Step 2: Run FPGA Tests
```bash
python3 test_scripts/fpga_test_runner.py \
    --port /dev/ttyUSB0 \
    --test-vectors test_vectors.npy \
    --output results.json
```

### Step 3: Verify Results
```bash
python3 test_scripts/verify_results.py \
    --results results.json \
    --golden-model golden_outputs.npy \
    --report verification_report.md
```

### Step 4: Analyze Report
```bash
cat verification_report.md
```

---

## 🛠️ Debugging Failed Tests

### 1. Identify Failed Stage
```python
result = verifier.verify_complete_pipeline(fpga_output, test_input, expected_output)

if not result['passed']:
    # Check which stage failed
    for stage_name, stage_result in result['stages'].items():
        if not stage_result.get('passed', True):
            print(f"Failed stage: {stage_name}")
            print(f"Error: {stage_result}")
```

### 2. Drill Down to Specific Operation
```python
# If MAC stage failed, test individual components
weight = np.int8(test_value_1)
activation = np.int8(test_value_2)
partial_sum = np.int16(test_value_3)

# Test multiply only
product = weight * activation
print(f"Product: {product}")

# Test MAC
mac_sw, overflow = mac_golden(weight, activation, partial_sum)
print(f"Expected MAC: {mac_sw}, Overflow: {overflow}")
print(f"Hardware MAC: {mac_hw}")
```

### 3. Check Intermediate Values
```python
# Enable verbose logging in verification
import logging
logging.basicConfig(level=logging.DEBUG)

# Run verification with detailed output
result = verify_mac(mac_hw, weight, activation, partial_sum)
print(f"Detailed result: {result}")
```

---

## 📈 Performance Benchmarks

### Expected Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Quantization Error** | ≤1 LSB | 99.9% of values |
| **MAC Accuracy** | Bit-exact | 100% match |
| **Accumulation Error** | 0 | Bit-exact |
| **ReLU Accuracy** | Bit-exact | 100% match |
| **Tanh Error** | ≤1% relative | 95% of values |
| **Pipeline Throughput** | >10 fps | MNIST inference |
| **Classification Accuracy** | >90% | MNIST test set |

---

## 🔗 Related Documentation

- **Complete Specification**: `hardware_math_verification_spec.md`
- **Python API Reference**: `test_scripts/math_verification.py` (docstrings)
- **Verilog Checkers**: `rtl/verification_checkers.v` (inline comments)
- **Testing Framework**: `testing_framework.md`
- **Mathematical Foundations**: `mathematical_foundations.md`

---

## 🤝 Support

If you encounter issues:

1. **Check the specification**: `hardware_math_verification_spec.md` has detailed explanations
2. **Review test vectors**: Ensure inputs are within valid ranges
3. **Compare tolerances**: Verify you're using correct tolerance for each stage
4. **Examine golden reference**: Make sure software implementation is correct

---

## 📊 Example Complete Test Session

```python
#!/usr/bin/env python3
"""
Complete verification example for UPduino AI Accelerator
"""

from math_verification import (
    HardwareVerifier,
    TestVectorGenerator,
    ToleranceConfig
)
import numpy as np

def main():
    print("=" * 80)
    print("UPduino v3.0 AI Accelerator - Mathematical Verification")
    print("=" * 80)

    # 1. Initialize
    tolerance_config = ToleranceConfig()
    verifier = HardwareVerifier(tolerance_config)
    generator = TestVectorGenerator(seed=42)

    # 2. Generate test vectors
    print("\n[1/4] Generating test vectors...")
    corner_cases = generator.generate_corner_cases(shape=(28, 28))
    random_cases = generator.generate_random_cases(count=100, shape=(28, 28))

    all_tests = corner_cases + random_cases
    print(f"  Generated {len(all_tests)} test vectors")

    # 3. Run tests
    print("\n[2/4] Running hardware tests...")
    for i, test_vec in enumerate(all_tests[:10]):  # Test first 10
        # Simulate FPGA inference (replace with actual FPGA call)
        fpga_output = simulate_fpga_inference(test_vec['input'])
        expected_output = compute_golden_output(test_vec['input'])

        result = verifier.verify_complete_pipeline(
            fpga_output=fpga_output,
            test_input=test_vec['input'],
            expected_output=expected_output
        )

        verifier.add_result(result)

        status = "✓" if result['passed'] else "✗"
        print(f"  Test {i+1:3d}: {status} {test_vec['type']}")

    # 4. Generate report
    print("\n[3/4] Generating verification report...")
    report = verifier.generate_report()
    print(report)

    # 5. Save results
    print("\n[4/4] Saving results...")
    with open('verification_report.txt', 'w') as f:
        f.write(report)
    print("  Report saved to verification_report.txt")

    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)

def simulate_fpga_inference(input_data):
    """Simulate FPGA inference (replace with actual FPGA communication)"""
    # This is a placeholder - replace with actual FPGA call
    return np.random.rand(10).astype(np.float32)

def compute_golden_output(input_data):
    """Compute expected output from golden model"""
    # This is a placeholder - replace with actual golden model
    return np.random.rand(10).astype(np.float32)

if __name__ == "__main__":
    main()
```

Save as `run_verification.py` and execute:
```bash
python3 run_verification.py
```

---

## ✅ Checklist for Implementation

- [ ] Read complete specification (`hardware_math_verification_spec.md`)
- [ ] Include `verification_checkers.v` in testbench
- [ ] Import `math_verification.py` in test scripts
- [ ] Generate test vectors (corner, random, adversarial)
- [ ] Implement stage-by-stage verification in testbench
- [ ] Run simulations with assertions enabled
- [ ] Verify FPGA hardware with Python golden model
- [ ] Generate comprehensive test report
- [ ] Achieve >95% pass rate on all test vectors
- [ ] Document any tolerance adjustments

---

**Last Updated:** 2026-01-05
**Version:** 1.0
**Authors:** AI Accelerator Verification Team
