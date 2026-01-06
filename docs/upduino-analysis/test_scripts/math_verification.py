"""
Hardware Mathematical Verification Module
UPduino v3.0 AI Accelerator - Golden Reference Implementations

Provides:
1. Golden reference implementations for each hardware stage
2. Verification functions with configurable tolerances
3. Test vector generation
4. Comprehensive error reporting

Author: AI Accelerator Verification Team
Date: 2026-01-05
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import warnings


# ==============================================================================
# Configuration and Tolerances
# ==============================================================================

class VerificationLevel(Enum):
    """Verification strictness levels"""
    BIT_EXACT = 0       # Bit-exact match required
    STRICT = 1          # Very tight tolerance (≤1 ULP)
    MODERATE = 2        # Moderate tolerance (≤0.01 relative error)
    RELAXED = 3         # Relaxed tolerance (≤0.05 relative error)


@dataclass
class ToleranceConfig:
    """Tolerance configuration for each pipeline stage"""
    input_validation: VerificationLevel = VerificationLevel.BIT_EXACT
    quantization: VerificationLevel = VerificationLevel.STRICT  # ±1 for rounding
    memory_load: VerificationLevel = VerificationLevel.BIT_EXACT
    mac_multiply: VerificationLevel = VerificationLevel.BIT_EXACT
    mac_accumulate: VerificationLevel = VerificationLevel.BIT_EXACT
    relu_activation: VerificationLevel = VerificationLevel.BIT_EXACT
    tanh_activation: VerificationLevel = VerificationLevel.MODERATE  # ±1%
    sigmoid_activation: VerificationLevel = VerificationLevel.MODERATE  # ±2%
    output_quantization: VerificationLevel = VerificationLevel.STRICT  # ±1

    def get_tolerance(self, level: VerificationLevel) -> float:
        """Get numerical tolerance for verification level"""
        tolerance_map = {
            VerificationLevel.BIT_EXACT: 0.0,
            VerificationLevel.STRICT: 1.0,  # ±1 LSB for integers, ±1 ULP for float
            VerificationLevel.MODERATE: 0.01,  # ±1% relative error
            VerificationLevel.RELAXED: 0.05,  # ±5% relative error
        }
        return tolerance_map[level]


# ==============================================================================
# Stage 0: Input Validation
# ==============================================================================

def verify_input(input_data: np.ndarray,
                 x_min: int = -128,
                 x_max: int = 127,
                 expected_shape: Optional[Tuple] = None) -> Dict[str, Any]:
    """
    Verify input data is within valid range and correct shape

    Args:
        input_data: Input array to verify
        x_min: Minimum allowed value (INT8: -128)
        x_max: Maximum allowed value (INT8: 127)
        expected_shape: Expected shape (e.g., (28, 28, 1) for MNIST)

    Returns:
        Dictionary with verification results
    """
    errors = []

    # Check for invalid values
    if np.isnan(input_data).any():
        errors.append("Input contains NaN values")

    if np.isinf(input_data).any():
        errors.append("Input contains Inf values")

    # Check range
    if np.any(input_data < x_min):
        min_val = np.min(input_data)
        errors.append(f"Input below minimum: {min_val} < {x_min}")

    if np.any(input_data > x_max):
        max_val = np.max(input_data)
        errors.append(f"Input above maximum: {max_val} > {x_max}")

    # Check shape
    if expected_shape is not None and input_data.shape != expected_shape:
        errors.append(f"Shape mismatch: {input_data.shape} != {expected_shape}")

    return {
        'passed': len(errors) == 0,
        'errors': errors,
        'input_min': np.min(input_data),
        'input_max': np.max(input_data),
        'input_shape': input_data.shape
    }


# ==============================================================================
# Stage 1: Quantization
# ==============================================================================

def quantize_symmetric_int8(x_fp32: np.ndarray,
                             scale: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Symmetric quantization to INT8 with zero-point = 0

    Mathematical formula:
        scale = max(|x|) / 127
        x_q = round(x_fp32 / scale)
        x_q_sat = clip(x_q, -128, 127)

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

    # Quantize with rounding
    x_q = np.round(x_fp32 / scale)

    # Saturate to INT8 range [-128, 127]
    x_q_sat = np.clip(x_q, -128, 127).astype(np.int8)

    return x_q_sat, scale


def quantize_asymmetric_uint8(x_fp32: np.ndarray,
                               scale: Optional[float] = None,
                               zero_point: Optional[int] = None) -> Tuple[np.ndarray, float, int]:
    """
    Asymmetric quantization to UINT8

    Mathematical formula:
        scale = (x_max - x_min) / 255
        zero_point = round(-x_min / scale)
        x_q = round(x_fp32 / scale) + zero_point
        x_q_sat = clip(x_q, 0, 255)

    Args:
        x_fp32: Input in FP32
        scale: Quantization scale (computed if None)
        zero_point: Zero point offset (computed if None)

    Returns:
        x_uint8: Quantized UINT8 value
        scale: Scale factor used
        zero_point: Zero point offset
    """
    if scale is None or zero_point is None:
        x_min = np.min(x_fp32)
        x_max = np.max(x_fp32)
        scale = (x_max - x_min) / 255.0
        scale = max(scale, 1e-8)
        zero_point = int(np.round(-x_min / scale))
        zero_point = np.clip(zero_point, 0, 255)

    # Quantize
    x_q = np.round(x_fp32 / scale) + zero_point

    # Saturate to UINT8 range [0, 255]
    x_q_sat = np.clip(x_q, 0, 255).astype(np.uint8)

    return x_q_sat, scale, zero_point


def verify_quantization(x_hw: np.ndarray,
                         x_fp32: np.ndarray,
                         scale: float,
                         zero_point: int = 0,
                         tolerance: int = 1) -> Dict[str, Any]:
    """
    Verify hardware quantization matches software golden reference

    Args:
        x_hw: Hardware quantized values
        x_fp32: Original FP32 values
        scale: Quantization scale
        zero_point: Zero point (0 for symmetric)
        tolerance: Allowed difference (default ±1 for rounding)

    Returns:
        Verification results dictionary
    """
    # Compute golden reference
    if zero_point == 0:
        x_sw, _ = quantize_symmetric_int8(x_fp32, scale)
    else:
        x_sw, _, _ = quantize_asymmetric_uint8(x_fp32, scale, zero_point)

    # Compute error
    error = np.abs(x_hw.astype(np.int32) - x_sw.astype(np.int32))
    max_error = np.max(error)
    mean_error = np.mean(error)

    # Check tolerance
    passed = max_error <= tolerance

    # Check saturation correctness
    if zero_point == 0:
        saturation_correct = np.all((x_hw >= -128) & (x_hw <= 127))
    else:
        saturation_correct = np.all((x_hw >= 0) & (x_hw <= 255))

    return {
        'passed': passed and saturation_correct,
        'max_error': max_error,
        'mean_error': mean_error,
        'saturation_correct': saturation_correct,
        'error_map': error,
        'scale': scale,
        'zero_point': zero_point
    }


# ==============================================================================
# Stage 2: Memory Load
# ==============================================================================

def verify_weight_load(weight_hw: np.ndarray,
                       weight_table: np.ndarray,
                       addr: int) -> Dict[str, Any]:
    """
    Verify hardware loaded correct weight from memory (bit-exact)

    Args:
        weight_hw: Weight value read by hardware
        weight_table: Expected weight values
        addr: Memory address

    Returns:
        Verification results
    """
    weight_expected = weight_table.flatten()[addr]

    # Bit-exact match required
    match = (weight_hw == weight_expected)

    # Check for bit errors (Hamming distance)
    if not match:
        xor_val = np.bitwise_xor(np.uint8(weight_hw), np.uint8(weight_expected))
        hamming_distance = bin(xor_val).count('1')
    else:
        hamming_distance = 0

    return {
        'passed': match,
        'weight_hw': weight_hw,
        'weight_expected': weight_expected,
        'hamming_distance': hamming_distance,
        'address': addr
    }


def verify_address_mapping(row: int, col: int, width: int, addr_hw: int) -> Dict[str, Any]:
    """
    Verify address calculation is correct

    Address formula: addr = row * width + col

    Args:
        row: Row index
        col: Column index
        width: Matrix width
        addr_hw: Hardware-computed address

    Returns:
        Verification results
    """
    addr_expected = row * width + col
    match = (addr_hw == addr_expected)

    return {
        'passed': match,
        'addr_hw': addr_hw,
        'addr_expected': addr_expected,
        'row': row,
        'col': col,
        'width': width
    }


# ==============================================================================
# Stage 3: Multiply-Accumulate (MAC)
# ==============================================================================

def mac_golden(weight: np.int8,
               activation: np.int8,
               partial_sum: np.int16 = 0,
               saturate: bool = True) -> Tuple[np.int16, bool]:
    """
    Golden reference for MAC operation: result = weight × activation + partial_sum

    Args:
        weight: INT8 weight value [-128, 127]
        activation: INT8 activation value [-128, 127]
        partial_sum: INT16 partial sum (default 0)
        saturate: Apply saturation to INT16 range

    Returns:
        mac_result: INT16 MAC result
        overflow_flag: True if overflow occurred
    """
    # Convert to wider integer type to detect overflow
    weight_32 = np.int32(weight)
    activation_32 = np.int32(activation)
    partial_sum_32 = np.int32(partial_sum)

    # Multiply
    product = weight_32 * activation_32

    # Add partial sum
    mac_result_32 = product + partial_sum_32

    # Check for overflow
    overflow = (mac_result_32 > 32767) or (mac_result_32 < -32768)

    # Saturate if requested
    if saturate:
        if mac_result_32 > 32767:
            mac_result = np.int16(32767)
        elif mac_result_32 < -32768:
            mac_result = np.int16(-32768)
        else:
            mac_result = np.int16(mac_result_32)
    else:
        mac_result = np.int16(mac_result_32)  # Will wrap on overflow

    return mac_result, overflow


def verify_mac(mac_hw: np.int16,
               weight: np.int8,
               activation: np.int8,
               partial_sum: np.int16 = 0,
               saturate: bool = True) -> Dict[str, Any]:
    """
    Verify hardware MAC matches golden reference (bit-exact for integers)

    Args:
        mac_hw: Hardware MAC result
        weight: Weight value
        activation: Activation value
        partial_sum: Partial sum input
        saturate: Whether saturation is enabled

    Returns:
        Verification results
    """
    mac_sw, overflow_expected = mac_golden(weight, activation, partial_sum, saturate)

    # Bit-exact match required for integer MAC
    match = (mac_hw == mac_sw)

    # Calculate component values for debugging
    product = np.int32(weight) * np.int32(activation)
    mac_result_32 = product + np.int32(partial_sum)

    return {
        'passed': match,
        'mac_hw': mac_hw,
        'mac_sw': mac_sw,
        'overflow': overflow_expected,
        'weight': weight,
        'activation': activation,
        'partial_sum': partial_sum,
        'product': product,
        'mac_result_32': mac_result_32
    }


# ==============================================================================
# Stage 4: Accumulation
# ==============================================================================

def accumulate_golden(mac_results: List[np.int16],
                      acc_width: int = 16,
                      saturate: bool = True) -> Tuple[np.int16, bool]:
    """
    Golden reference for sequential accumulation

    Args:
        mac_results: List of MAC results
        acc_width: Accumulator width in bits (default 16)
        saturate: Apply saturation

    Returns:
        final_accumulator: Final accumulated value
        overflow_occurred: True if overflow detected
    """
    # Use wider type to detect overflow
    acc = np.int32(0)
    overflow_occurred = False

    # Accumulate
    for mac in mac_results:
        acc += np.int32(mac)

    # Determine limits based on accumulator width
    max_val = (1 << (acc_width - 1)) - 1
    min_val = -(1 << (acc_width - 1))

    # Check for overflow
    if acc > max_val or acc < min_val:
        overflow_occurred = True

    # Saturate if requested
    if saturate:
        if acc > max_val:
            final_acc = np.int16(max_val)
        elif acc < min_val:
            final_acc = np.int16(min_val)
        else:
            final_acc = np.int16(acc)
    else:
        final_acc = np.int16(acc)

    return final_acc, overflow_occurred


def verify_accumulation(acc_hw: np.int16,
                        mac_results: List[np.int16],
                        tolerance: int = 0) -> Dict[str, Any]:
    """
    Verify hardware accumulation

    Args:
        acc_hw: Hardware accumulator value
        mac_results: List of MAC results that were accumulated
        tolerance: Allowed error (default 0 for bit-exact)

    Returns:
        Verification results
    """
    acc_sw, overflow = accumulate_golden(mac_results)

    error = abs(int(acc_hw) - int(acc_sw))
    passed = error <= tolerance

    return {
        'passed': passed,
        'acc_hw': acc_hw,
        'acc_sw': acc_sw,
        'error': error,
        'overflow': overflow,
        'num_macs': len(mac_results),
        'mac_sum_32bit': sum(np.int32(m) for m in mac_results)
    }


# ==============================================================================
# Stage 5: Activation Functions
# ==============================================================================

def relu_golden(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation: y = max(0, x)

    Bit-exact for integer inputs
    """
    return np.maximum(0, x)


def verify_relu(relu_hw: np.ndarray, input_val: np.ndarray) -> Dict[str, Any]:
    """
    Verify ReLU - must be bit-exact for integer inputs
    """
    relu_sw = relu_golden(input_val)

    # Bit-exact match
    match = np.all(relu_hw == relu_sw)

    # Check specific conditions
    positive_correct = np.all(relu_hw[input_val > 0] == input_val[input_val > 0])
    negative_correct = np.all(relu_hw[input_val <= 0] == 0)

    return {
        'passed': match and positive_correct and negative_correct,
        'relu_hw': relu_hw,
        'relu_sw': relu_sw,
        'positive_correct': positive_correct,
        'negative_correct': negative_correct,
        'max_error': np.max(np.abs(relu_hw - relu_sw))
    }


def tanh_pwl_golden(x: np.ndarray,
                    threshold: float = 2.0,
                    scale: float = 2.0) -> np.ndarray:
    """
    Piecewise linear tanh approximation

    tanh_approx(x) = {
        -1.0,       if x < -threshold
        x / scale,  if -threshold ≤ x ≤ threshold
        +1.0,       if x > threshold
    }

    Args:
        x: Input values
        threshold: Saturation threshold (default 2.0)
        scale: Linear region scale factor (default 2.0)

    Returns:
        Tanh approximation
    """
    x = np.asarray(x, dtype=np.float32)
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


def verify_tanh_pwl(tanh_hw: np.ndarray,
                    input_val: np.ndarray,
                    threshold: float = 2.0,
                    scale: float = 2.0,
                    tolerance: float = 0.01) -> Dict[str, Any]:
    """
    Verify piecewise linear tanh

    Args:
        tanh_hw: Hardware tanh output
        input_val: Input values
        threshold: Saturation threshold
        scale: Linear region scale
        tolerance: Relative error tolerance (default 0.01 = 1%)

    Returns:
        Verification results
    """
    tanh_sw = tanh_pwl_golden(input_val, threshold, scale)

    # Calculate errors
    abs_error = np.abs(tanh_hw - tanh_sw)
    rel_error = abs_error / (np.abs(tanh_sw) + 1e-8)

    max_abs_error = np.max(abs_error)
    max_rel_error = np.max(rel_error)
    mean_rel_error = np.mean(rel_error)

    # For saturated regions, should be nearly exact
    saturated_mask = np.abs(input_val) > threshold
    saturated_error = np.max(abs_error[saturated_mask]) if np.any(saturated_mask) else 0.0

    # Overall pass condition
    passed = (max_rel_error <= tolerance) and (saturated_error < 0.001)

    return {
        'passed': passed,
        'max_abs_error': max_abs_error,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'saturated_error': saturated_error,
        'tanh_hw': tanh_hw,
        'tanh_sw': tanh_sw
    }


def sigmoid_pwl_golden(x: np.ndarray,
                       neg_threshold: float = -4.0,
                       pos_threshold: float = 4.0,
                       slope: float = 0.125) -> np.ndarray:
    """
    Piecewise linear sigmoid approximation

    sigmoid_approx(x) = {
        0,             if x < neg_threshold
        0.5 + x/8,     if neg_threshold ≤ x ≤ pos_threshold
        1,             if x > pos_threshold
    }
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.zeros_like(x, dtype=np.float32)

    # Saturate low
    mask_low = x < neg_threshold
    y[mask_low] = 0.0

    # Linear region
    mask_mid = (x >= neg_threshold) & (x <= pos_threshold)
    y[mask_mid] = 0.5 + x[mask_mid] * slope

    # Saturate high
    mask_high = x > pos_threshold
    y[mask_high] = 1.0

    return y


def verify_sigmoid_pwl(sigmoid_hw: np.ndarray,
                       input_val: np.ndarray,
                       tolerance: float = 0.02) -> Dict[str, Any]:
    """
    Verify piecewise linear sigmoid (±2% tolerance)
    """
    sigmoid_sw = sigmoid_pwl_golden(input_val)

    abs_error = np.abs(sigmoid_hw - sigmoid_sw)
    rel_error = abs_error / (np.abs(sigmoid_sw) + 1e-8)

    max_rel_error = np.max(rel_error)
    passed = max_rel_error <= tolerance

    return {
        'passed': passed,
        'max_abs_error': np.max(abs_error),
        'max_rel_error': max_rel_error,
        'mean_rel_error': np.mean(rel_error)
    }


# ==============================================================================
# Complete Pipeline Verification
# ==============================================================================

class HardwareVerifier:
    """
    Main verification class for complete hardware pipeline
    """

    def __init__(self, tolerance_config: Optional[ToleranceConfig] = None):
        """
        Initialize verifier with tolerance configuration

        Args:
            tolerance_config: Tolerance settings (uses defaults if None)
        """
        self.tolerance = tolerance_config or ToleranceConfig()
        self.results = []
        self.errors = []

    def verify_complete_pipeline(self,
                                  fpga_output: np.ndarray,
                                  test_input: np.ndarray,
                                  expected_output: Optional[np.ndarray] = None,
                                  layer_outputs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Verify entire inference pipeline

        Args:
            fpga_output: Final output from FPGA
            test_input: Input test vector
            expected_output: Expected output from golden model
            layer_outputs: Intermediate layer outputs from FPGA (if available)

        Returns:
            Comprehensive verification report
        """
        results = {
            'passed': True,
            'stages': {},
            'overall_error': {},
            'classification': {}
        }

        # 1. Input validation
        results['stages']['input'] = verify_input(test_input)
        if not results['stages']['input']['passed']:
            results['passed'] = False

        # 2. Final output comparison
        if expected_output is not None:
            # Calculate error metrics
            mse = np.mean((fpga_output - expected_output) ** 2)
            mae = np.mean(np.abs(fpga_output - expected_output))
            max_error = np.max(np.abs(fpga_output - expected_output))

            # Correlation
            correlation = np.corrcoef(fpga_output.flatten(), expected_output.flatten())[0, 1]

            results['overall_error'] = {
                'mse': float(mse),
                'mae': float(mae),
                'max_error': float(max_error),
                'correlation': float(correlation),
                'rmse': float(np.sqrt(mse))
            }

            # Check if errors are within acceptable bounds
            # For classification: check if argmax matches
            if len(fpga_output.shape) == 1:  # Classification output
                fpga_pred = np.argmax(fpga_output)
                expected_pred = np.argmax(expected_output)

                results['classification'] = {
                    'fpga_prediction': int(fpga_pred),
                    'expected_prediction': int(expected_pred),
                    'match': fpga_pred == expected_pred,
                    'confidence_hw': float(fpga_output[fpga_pred]),
                    'confidence_sw': float(expected_output[expected_pred])
                }

                if fpga_pred != expected_pred:
                    results['passed'] = False

        # 3. Layer-by-layer verification (if available)
        if layer_outputs is not None:
            results['stages']['layers'] = {}
            for layer_name, layer_output_hw in layer_outputs.items():
                # Would need golden layer outputs to compare
                results['stages']['layers'][layer_name] = {
                    'shape': layer_output_hw.shape,
                    'min': float(np.min(layer_output_hw)),
                    'max': float(np.max(layer_output_hw)),
                    'mean': float(np.mean(layer_output_hw)),
                    'std': float(np.std(layer_output_hw))
                }

        return results

    def generate_report(self) -> str:
        """
        Generate human-readable verification report

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 80,
            "HARDWARE VERIFICATION REPORT",
            "=" * 80,
            "",
            f"Total Tests: {len(self.results)}",
            f"Passed: {sum(1 for r in self.results if r.get('passed', False))}",
            f"Failed: {sum(1 for r in self.results if not r.get('passed', False))}",
            "",
        ]

        # Detailed results
        for i, result in enumerate(self.results):
            report_lines.append(f"Test {i + 1}:")
            report_lines.append(f"  Status: {'✓ PASS' if result.get('passed', False) else '✗ FAIL'}")

            if 'overall_error' in result:
                report_lines.append(f"  MSE: {result['overall_error']['mse']:.6f}")
                report_lines.append(f"  MAE: {result['overall_error']['mae']:.6f}")
                report_lines.append(f"  Max Error: {result['overall_error']['max_error']:.6f}")
                report_lines.append(f"  Correlation: {result['overall_error']['correlation']:.6f}")

            if 'classification' in result:
                report_lines.append(f"  Classification:")
                report_lines.append(f"    HW Prediction: {result['classification']['fpga_prediction']}")
                report_lines.append(f"    SW Prediction: {result['classification']['expected_prediction']}")
                report_lines.append(f"    Match: {'✓' if result['classification']['match'] else '✗'}")

            report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def add_result(self, result: Dict[str, Any]):
        """Add verification result to collection"""
        self.results.append(result)

    def reset(self):
        """Reset results and errors"""
        self.results = []
        self.errors = []


# ==============================================================================
# Test Vector Generation
# ==============================================================================

class TestVectorGenerator:
    """Comprehensive test vector generation for hardware verification"""

    def __init__(self, seed: int = 42):
        """
        Initialize generator with random seed

        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

    def generate_corner_cases(self, shape: Tuple = (28, 28), dtype=np.int8) -> List[Dict]:
        """
        Generate corner case test vectors

        Returns:
            List of test vector dictionaries
        """
        test_vectors = []

        # All zeros
        test_vectors.append({
            'type': 'all_zero',
            'input': np.zeros(shape, dtype=dtype),
            'description': 'All zero input'
        })

        # All maximum
        max_val = 127 if dtype == np.int8 else 255
        test_vectors.append({
            'type': 'all_max',
            'input': np.full(shape, max_val, dtype=dtype),
            'description': f'All maximum value ({max_val})'
        })

        # All minimum
        min_val = -128 if dtype == np.int8 else 0
        test_vectors.append({
            'type': 'all_min',
            'input': np.full(shape, min_val, dtype=dtype),
            'description': f'All minimum value ({min_val})'
        })

        # Checkerboard pattern
        checkerboard = np.zeros(shape, dtype=dtype)
        checkerboard[::2, ::2] = max_val
        checkerboard[1::2, 1::2] = max_val
        test_vectors.append({
            'type': 'checkerboard',
            'input': checkerboard,
            'description': 'Checkerboard pattern'
        })

        # Single hot pixel
        single_hot = np.zeros(shape, dtype=dtype)
        center = (shape[0] // 2, shape[1] // 2)
        single_hot[center] = max_val
        test_vectors.append({
            'type': 'single_hot',
            'input': single_hot,
            'description': 'Single hot pixel at center'
        })

        return test_vectors

    def generate_random_cases(self,
                              count: int = 1000,
                              shape: Tuple = (28, 28),
                              dtype=np.int8) -> List[Dict]:
        """
        Generate random test vectors with various distributions

        Args:
            count: Number of test vectors
            shape: Shape of each vector
            dtype: Data type

        Returns:
            List of test vectors
        """
        test_vectors = []
        min_val = -128 if dtype == np.int8 else 0
        max_val = 127 if dtype == np.int8 else 255

        for i in range(count):
            # Vary distribution
            if i % 3 == 0:
                # Uniform distribution
                data = np.random.randint(min_val, max_val + 1, shape, dtype=dtype)
                dist_type = 'uniform'
            elif i % 3 == 1:
                # Gaussian distribution
                mean = (max_val + min_val) / 2
                std = (max_val - min_val) / 6
                data = np.random.normal(mean, std, shape)
                data = np.clip(data, min_val, max_val).astype(dtype)
                dist_type = 'gaussian'
            else:
                # Sparse (mostly zeros)
                data = np.zeros(shape, dtype=dtype)
                num_nonzero = int(0.1 * np.prod(shape))
                indices = np.random.choice(np.prod(shape), num_nonzero, replace=False)
                flat_data = data.flatten()
                flat_data[indices] = np.random.randint(min_val, max_val + 1, num_nonzero, dtype=dtype)
                data = flat_data.reshape(shape)
                dist_type = 'sparse'

            test_vectors.append({
                'type': dist_type,
                'input': data,
                'description': f'Random {dist_type} distribution #{i}'
            })

        return test_vectors

    def generate_adversarial_cases(self, shape: Tuple = (28, 28), dtype=np.int8) -> List[Dict]:
        """
        Generate adversarial test cases to stress the hardware

        Returns:
            List of adversarial test vectors
        """
        test_vectors = []
        max_val = 127 if dtype == np.int8 else 255

        # Alternating maximum values (stress MAC accumulation)
        alternating = np.zeros(shape, dtype=dtype)
        alternating[::2, :] = max_val
        test_vectors.append({
            'type': 'alternating_rows',
            'input': alternating,
            'description': 'Alternating rows of max values'
        })

        # High-frequency pattern
        high_freq = np.zeros(shape, dtype=dtype)
        for i in range(shape[0]):
            for j in range(shape[1]):
                high_freq[i, j] = max_val if (i + j) % 2 == 0 else 0
        test_vectors.append({
            'type': 'high_frequency',
            'input': high_freq,
            'description': 'High-frequency checkerboard'
        })

        return test_vectors


# ==============================================================================
# Usage Example
# ==============================================================================

if __name__ == "__main__":
    print("Hardware Mathematical Verification Module")
    print("=" * 80)

    # Example: Verify quantization
    print("\nExample 1: Quantization Verification")
    print("-" * 80)

    x_fp32 = np.array([1.0, -0.5, 0.0, 2.0, -1.5], dtype=np.float32)
    x_hw = np.array([127, -64, 0, 127, -127], dtype=np.int8)  # Simulated HW output
    scale = 1.0 / 127.0

    result = verify_quantization(x_hw, x_fp32, scale, tolerance=1)
    print(f"Passed: {result['passed']}")
    print(f"Max Error: {result['max_error']}")

    # Example: Verify MAC
    print("\nExample 2: MAC Verification")
    print("-" * 80)

    weight = np.int8(10)
    activation = np.int8(20)
    partial_sum = np.int16(100)
    mac_hw = np.int16(300)  # 10*20 + 100

    result = verify_mac(mac_hw, weight, activation, partial_sum)
    print(f"Passed: {result['passed']}")
    print(f"MAC HW: {result['mac_hw']}, MAC SW: {result['mac_sw']}")

    # Example: Complete pipeline verification
    print("\nExample 3: Complete Pipeline")
    print("-" * 80)

    verifier = HardwareVerifier()

    # Simulate FPGA output
    fpga_output = np.array([0.1, 0.05, 0.15, 0.7], dtype=np.float32)  # Simulated softmax output
    expected_output = np.array([0.09, 0.06, 0.14, 0.71], dtype=np.float32)
    test_input = np.random.randn(28, 28).astype(np.float32)

    result = verifier.verify_complete_pipeline(fpga_output, test_input, expected_output)
    verifier.add_result(result)

    print(verifier.generate_report())

    print("\n" + "=" * 80)
    print("Module loaded successfully. Ready for hardware verification.")
    print("=" * 80)
