// =============================================================================
// Verification Checkers for AI Accelerator
// =============================================================================
// Description: Comprehensive verification assertions and checkers for
//              mathematical correctness at each pipeline stage
// Author: AI Accelerator Verification Team
// Date: 2026-01-05
// Target: UPduino v3.1 (iCE40 UP5K)
// =============================================================================
//
// This file provides:
// 1. Assertion properties for each pipeline stage
// 2. Verification tasks for testbenches
// 3. Error checking and reporting
// 4. Coverage points
//
// Usage: Include this file in your testbench:
//   `include "verification_checkers.v"
//
// =============================================================================

`ifndef VERIFICATION_CHECKERS_V
`define VERIFICATION_CHECKERS_V

// =============================================================================
// STAGE 0: Input Validation
// =============================================================================

// Check input range for INT8
task check_input_range_int8;
    input signed [7:0] data_in;
    input string signal_name;
    begin
        if (data_in < -8'sd128 || data_in > 8'sd127) begin
            $error("[INPUT] %s out of INT8 range: %d", signal_name, data_in);
        end
    end
endtask

// Check input validity
task check_input_valid;
    input valid_signal;
    input [7:0] data_in;
    begin
        if (valid_signal) begin
            // For integer inputs, no NaN check needed
            // Just verify it's within expected range
            check_input_range_int8(data_in, "input_data");
        end
    end
endtask

// =============================================================================
// STAGE 1: Quantization Verification
// =============================================================================

// Verify quantization saturation
task verify_quantization_saturation;
    input signed [15:0] data_in;      // Input before quantization
    input signed [7:0]  data_out;     // Quantized output
    input integer       shift_amount; // Right shift amount
    reg signed [15:0]   shifted;
    begin
        shifted = data_in >>> shift_amount;

        // Check saturation to INT8 range
        if (shifted > 16'sd127) begin
            if (data_out !== 8'sd127) begin
                $error("[QUANT] Saturation failed (high): in=%d, out=%d, expected=127",
                       shifted, data_out);
            end
        end else if (shifted < -16'sd128) begin
            if (data_out !== -8'sd128) begin
                $error("[QUANT] Saturation failed (low): in=%d, out=%d, expected=-128",
                       shifted, data_out);
            end
        end else begin
            // No saturation needed
            if (data_out !== shifted[7:0]) begin
                $error("[QUANT] Non-saturated output mismatch: in=%d, out=%d, expected=%d",
                       shifted, data_out, shifted[7:0]);
            end
        end
    end
endtask

// Verify rounding mode
task verify_rounding;
    input signed [15:0] data_in;
    input signed [7:0]  data_out;
    input integer       shift_amount;
    reg signed [15:0]   rounded;
    reg signed [15:0]   rounding_bit;
    begin
        // Add rounding bit (0.5 in fixed-point)
        rounding_bit = (1 << (shift_amount - 1));
        rounded = (data_in + rounding_bit) >>> shift_amount;

        // Check if output matches rounded value (with saturation)
        if (rounded > 16'sd127) begin
            if (data_out !== 8'sd127)
                $error("[ROUND] Rounding with saturation failed (high)");
        end else if (rounded < -16'sd128) begin
            if (data_out !== -8'sd128)
                $error("[ROUND] Rounding with saturation failed (low)");
        end else begin
            if (data_out !== rounded[7:0])
                $error("[ROUND] Rounding mismatch: got=%d, expected=%d", data_out, rounded[7:0]);
        end
    end
endtask

// =============================================================================
// STAGE 2: Memory Access Verification
// =============================================================================

// Verify address calculation
task verify_address_calculation;
    input integer row_idx;
    input integer col_idx;
    input integer matrix_width;
    input integer addr_hw;
    reg integer addr_expected;
    begin
        addr_expected = row_idx * matrix_width + col_idx;

        if (addr_hw !== addr_expected) begin
            $error("[ADDR] Address mismatch: row=%d, col=%d, width=%d, hw=%d, expected=%d",
                   row_idx, col_idx, matrix_width, addr_hw, addr_expected);
        end
    end
endtask

// Verify memory read (bit-exact)
task verify_memory_read;
    input [7:0] data_hw;
    input [7:0] data_expected;
    input integer addr;
    integer hamming_dist;
    begin
        if (data_hw !== data_expected) begin
            // Calculate Hamming distance for bit error detection
            hamming_dist = $countones(data_hw ^ data_expected);
            $error("[MEM] Memory read mismatch at addr=%d: hw=%d, expected=%d, hamming_dist=%d",
                   addr, data_hw, data_expected, hamming_dist);
        end
    end
endtask

// =============================================================================
// STAGE 3: MAC (Multiply-Accumulate) Verification
// =============================================================================

// Verify multiplication (INT8 × INT8 → INT16)
task verify_multiply;
    input signed [7:0]  weight;
    input signed [7:0]  activation;
    input signed [15:0] product_hw;
    reg signed [31:0]   product_expected_32;
    reg signed [15:0]   product_expected;
    begin
        // Compute expected product with sign extension
        product_expected_32 = $signed(weight) * $signed(activation);
        product_expected = product_expected_32[15:0];

        if (product_hw !== product_expected) begin
            $error("[MAC-MUL] Multiplication mismatch: %d × %d = %d (hw) vs %d (expected)",
                   weight, activation, product_hw, product_expected);
        end

        // Verify product is within valid INT16 range for INT8×INT8
        if (product_expected_32 > 32'sd16129 || product_expected_32 < -32'sd16384) begin
            $error("[MAC-MUL] Product out of expected range for INT8×INT8: %d",
                   product_expected_32);
        end
    end
endtask

// Verify MAC operation (multiply + add)
task verify_mac;
    input signed [7:0]  weight;
    input signed [7:0]  activation;
    input signed [15:0] partial_sum_in;
    input signed [15:0] mac_result_hw;
    reg signed [31:0]   product;
    reg signed [31:0]   mac_expected_32;
    reg signed [15:0]   mac_expected;
    reg                 overflow;
    begin
        // Calculate expected MAC result
        product = $signed(weight) * $signed(activation);
        mac_expected_32 = product + $signed(partial_sum_in);

        // Check for overflow
        overflow = (mac_expected_32 > 32'sd32767) || (mac_expected_32 < -32'sd32768);

        // Saturate if overflow
        if (mac_expected_32 > 32'sd32767)
            mac_expected = 16'sd32767;
        else if (mac_expected_32 < -32'sd32768)
            mac_expected = -16'sd32768;
        else
            mac_expected = mac_expected_32[15:0];

        if (mac_result_hw !== mac_expected) begin
            $error("[MAC] MAC mismatch: (%d × %d) + %d = %d (hw) vs %d (expected), overflow=%b",
                   weight, activation, partial_sum_in, mac_result_hw, mac_expected, overflow);
        end

        // Warning if overflow occurred
        if (overflow) begin
            $warning("[MAC] Overflow occurred: (%d × %d) + %d = %d",
                     weight, activation, partial_sum_in, mac_expected_32);
        end
    end
endtask

// Verify sign extension in MAC
task verify_sign_extension;
    input signed [15:0] product;
    input signed [15:0] mac_result;
    begin
        // Check that sign bit is properly extended
        // This is implicit in Verilog signed arithmetic, but we verify it
        if (product[15] !== mac_result[15] &&
            !(mac_result == 16'sd32767 || mac_result == -16'sd32768)) begin
            // Sign changed without saturation - potential error
            $warning("[MAC] Sign bit changed unexpectedly: product=%d, mac=%d",
                     product, mac_result);
        end
    end
endtask

// =============================================================================
// STAGE 4: Accumulation Verification
// =============================================================================

// Verify accumulation step
task verify_accumulation_step;
    input signed [15:0] acc_prev;
    input signed [15:0] mac_in;
    input signed [15:0] acc_current;
    reg signed [31:0]   acc_expected_32;
    reg signed [15:0]   acc_expected;
    reg                 overflow;
    begin
        acc_expected_32 = $signed(acc_prev) + $signed(mac_in);
        overflow = (acc_expected_32 > 32'sd32767) || (acc_expected_32 < -32'sd32768);

        // Saturate if overflow
        if (acc_expected_32 > 32'sd32767)
            acc_expected = 16'sd32767;
        else if (acc_expected_32 < -32'sd32768)
            acc_expected = -16'sd32768;
        else
            acc_expected = acc_expected_32[15:0];

        if (acc_current !== acc_expected) begin
            $error("[ACC] Accumulation mismatch: %d + %d = %d (hw) vs %d (expected)",
                   acc_prev, mac_in, acc_current, acc_expected);
        end

        if (overflow) begin
            $warning("[ACC] Accumulator overflow: %d + %d = %d",
                     acc_prev, mac_in, acc_expected_32);
        end
    end
endtask

// Verify final accumulation value
task verify_final_accumulation;
    input signed [15:0] acc_hw;
    input integer       num_macs;
    input signed [15:0] mac_history [0:127];  // Array of MAC results
    reg signed [31:0]   acc_sw;
    integer             i;
    reg                 overflow;
    begin
        acc_sw = 0;
        overflow = 0;

        // Accumulate all MACs
        for (i = 0; i < num_macs; i = i + 1) begin
            acc_sw = acc_sw + mac_history[i];
        end

        // Check for overflow
        if (acc_sw > 32'sd32767 || acc_sw < -32'sd32768) begin
            overflow = 1;
            if (acc_sw > 32'sd32767)
                acc_sw = 32'sd32767;
            else
                acc_sw = -32'sd32768;
        end

        if (acc_hw !== acc_sw[15:0]) begin
            $error("[ACC-FINAL] Final accumulation mismatch: hw=%d, sw=%d, num_macs=%d, overflow=%b",
                   acc_hw, acc_sw[15:0], num_macs, overflow);
        end else begin
            $display("[ACC-FINAL] ✓ Accumulation verified: %d MACs, result=%d", num_macs, acc_hw);
        end
    end
endtask

// =============================================================================
// STAGE 5: Activation Functions Verification
// =============================================================================

// Verify ReLU (bit-exact)
task verify_relu;
    input signed [15:0] data_in;
    input signed [15:0] data_out;
    begin
        if (data_in >= 0) begin
            // Positive values pass through unchanged
            if (data_out !== data_in) begin
                $error("[RELU] Positive input mismatch: in=%d, out=%d", data_in, data_out);
            end
        end else begin
            // Negative values become zero
            if (data_out !== 16'd0) begin
                $error("[RELU] Negative input not zeroed: in=%d, out=%d", data_in, data_out);
            end
        end
    end
endtask

// Verify Tanh piecewise linear approximation
task verify_tanh_pwl;
    input signed [15:0] data_in;
    input signed [15:0] data_out;
    input signed [15:0] threshold;  // Saturation threshold (e.g., 2.0 in fixed-point)
    input signed [15:0] max_val;    // Maximum output value (1.0 in fixed-point)
    input signed [15:0] min_val;    // Minimum output value (-1.0 in fixed-point)
    begin
        if (data_in > threshold) begin
            // Should saturate to max
            if (data_out !== max_val) begin
                $error("[TANH] High saturation failed: in=%d, out=%d, expected=%d",
                       data_in, data_out, max_val);
            end
        end else if (data_in < -threshold) begin
            // Should saturate to min
            if (data_out !== min_val) begin
                $error("[TANH] Low saturation failed: in=%d, out=%d, expected=%d",
                       data_in, data_out, min_val);
            end
        end else begin
            // Linear region: out ≈ in / scale
            // This check is approximate due to quantization
            // We just verify it's in a reasonable range
            if (data_out > max_val || data_out < min_val) begin
                $error("[TANH] Linear region output out of range: in=%d, out=%d",
                       data_in, data_out);
            end
        end
    end
endtask

// =============================================================================
// STAGE 6 & 7: Output Verification
// =============================================================================

// Verify output quantization
task verify_output_quantization;
    input signed [15:0] data_in;
    input signed [7:0]  data_out;
    input integer       shift_amount;
    begin
        verify_quantization_saturation(data_in, data_out, shift_amount);
    end
endtask

// Verify final output range
task verify_output_range;
    input signed [7:0] output_data;
    input signed [7:0] min_expected;
    input signed [7:0] max_expected;
    begin
        if (output_data < min_expected || output_data > max_expected) begin
            $error("[OUTPUT] Output out of expected range: %d not in [%d, %d]",
                   output_data, min_expected, max_expected);
        end
    end
endtask

// =============================================================================
// SystemVerilog Assertions (SVA)
// =============================================================================

`ifdef SYSTEMVERILOG

// Assertion: MAC product within valid range for INT8×INT8
property mac_product_range;
    @(posedge clk)
    logic signed [15:0] product_local;
    (weight_load, product_local = $signed(weight_in) * $signed(activation_in))
    |-> (product_local >= -16'sd16384 && product_local <= 16'sd16129);
endproperty

assert property (mac_product_range)
    else $error("[ASSERT] MAC product out of valid INT8×INT8 range");

// Assertion: ReLU correctness
property relu_positive_passthrough;
    @(posedge clk)
    (activation_type == 2'b01 && data_in >= 0) |-> (data_out == data_in);
endproperty

assert property (relu_positive_passthrough)
    else $error("[ASSERT] ReLU failed to pass through positive value");

property relu_negative_zero;
    @(posedge clk)
    (activation_type == 2'b01 && data_in < 0) |-> (data_out == 0);
endproperty

assert property (relu_negative_zero)
    else $error("[ASSERT] ReLU failed to zero negative value");

// Assertion: Quantization saturation
property quantization_saturate_high;
    @(posedge clk)
    logic signed [15:0] shifted_local;
    (valid_in, shifted_local = data_in >>> (DATA_WIDTH - OUT_WIDTH))
    |-> (shifted_local > ((1 << (OUT_WIDTH-1)) - 1)) |->
        (data_out == ((1 << (OUT_WIDTH-1)) - 1));
endproperty

assert property (quantization_saturate_high)
    else $error("[ASSERT] Quantization failed to saturate high");

property quantization_saturate_low;
    @(posedge clk)
    logic signed [15:0] shifted_local;
    (valid_in, shifted_local = data_in >>> (DATA_WIDTH - OUT_WIDTH))
    |-> (shifted_local < -(1 << (OUT_WIDTH-1))) |->
        (data_out == -(1 << (OUT_WIDTH-1)));
endproperty

assert property (quantization_saturate_low)
    else $error("[ASSERT] Quantization failed to saturate low");

// Assertion: Memory read stability
property memory_read_stable;
    @(posedge clk)
    logic [7:0] addr_prev;
    (weight_load, addr_prev = weight_addr) ##1 (weight_addr == addr_prev)
    |-> (weight_out == $past(weight_out));
endproperty

assert property (memory_read_stable)
    else $error("[ASSERT] Memory read unstable for same address");

`endif // SYSTEMVERILOG

// =============================================================================
// Coverage Points
// =============================================================================

`ifdef FUNCTIONAL_COVERAGE

covergroup cg_mac_operations @(posedge clk);
    // Cover different ranges of weights
    cp_weight: coverpoint weight_in {
        bins zero = {0};
        bins positive_small = {[1:32]};
        bins positive_medium = {[33:96]};
        bins positive_large = {[97:127]};
        bins negative_small = {[-32:-1]};
        bins negative_medium = {[-96:-33]};
        bins negative_large = {[-128:-97]};
    }

    // Cover different ranges of activations
    cp_activation: coverpoint activation_in {
        bins zero = {0};
        bins positive_small = {[1:32]};
        bins positive_medium = {[33:96]};
        bins positive_large = {[97:127]};
        bins negative_small = {[-32:-1]};
        bins negative_medium = {[-96:-33]};
        bins negative_large = {[-128:-97]};
    }

    // Cross coverage: weight × activation combinations
    cx_weight_activation: cross cp_weight, cp_activation;

    // Cover overflow conditions
    cp_overflow: coverpoint overflow_flag {
        bins no_overflow = {0};
        bins overflow = {1};
    }
endgroup

covergroup cg_activation_functions @(posedge clk);
    cp_activation_type: coverpoint activation_type {
        bins linear = {2'b00};
        bins relu = {2'b01};
        bins tanh = {2'b10};
    }

    // Cover input ranges for activation functions
    cp_activation_input: coverpoint data_in {
        bins zero = {0};
        bins positive_low = {[1:127]};
        bins positive_high = {[128:32767]};
        bins negative_low = {[-127:-1]};
        bins negative_high = {[-32768:-128]};
    }

    cx_activation: cross cp_activation_type, cp_activation_input;
endgroup

covergroup cg_quantization @(posedge clk);
    // Cover quantization scenarios
    cp_saturation: coverpoint saturation_occurred {
        bins no_sat = {0};
        bins saturated_high = {1};
        bins saturated_low = {2};
    }

    cp_rounding: coverpoint rounding_occurred {
        bins no_round = {0};
        bins round_up = {1};
        bins round_down = {2};
    }
endgroup

// Instantiate coverage groups
`ifdef ENABLE_COVERAGE
cg_mac_operations cg_mac_inst = new();
cg_activation_functions cg_act_inst = new();
cg_quantization cg_quant_inst = new();
`endif

`endif // FUNCTIONAL_COVERAGE

// =============================================================================
// Utility Functions
// =============================================================================

// Convert fixed-point to real for display
function real fixed_to_real;
    input signed [15:0] fixed_val;
    input integer frac_bits;
    begin
        fixed_to_real = $itor(fixed_val) / $itor(1 << frac_bits);
    end
endfunction

// Display formatted error message
task display_error;
    input string stage;
    input string message;
    input integer error_code;
    begin
        $display("================================================================================");
        $display("ERROR [%s]: %s (code: %0d)", stage, message, error_code);
        $display("Time: %0t", $time);
        $display("================================================================================");
    end
endtask

// Display formatted warning message
task display_warning;
    input string stage;
    input string message;
    begin
        $display("WARNING [%s]: %s (time: %0t)", stage, message, $time);
    end
endtask

// Display pass message
task display_pass;
    input string stage;
    input string message;
    begin
        $display("✓ PASS [%s]: %s", stage, message);
    end
endtask

// =============================================================================
// End of verification checkers
// =============================================================================

`endif // VERIFICATION_CHECKERS_V

// =============================================================================
// Usage Example in Testbench
// =============================================================================
//
// module ai_accelerator_tb;
//     `include "verification_checkers.v"
//
//     // ... testbench signals ...
//
//     always @(posedge clk) begin
//         if (mac_valid) begin
//             verify_mac(weight_val, activation_val, partial_sum, mac_result);
//         end
//
//         if (accumulate_done) begin
//             verify_final_accumulation(accumulator, mac_count, mac_history);
//         end
//
//         if (activation_valid) begin
//             case (activation_type)
//                 2'b01: verify_relu(activation_in, activation_out);
//                 2'b10: verify_tanh_pwl(activation_in, activation_out, threshold, max_val, min_val);
//             endcase
//         end
//     end
// endmodule
//
// =============================================================================
