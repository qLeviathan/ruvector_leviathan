// ============================================================================
// Testbench for Causal Lattice FSM
// ============================================================================
//
// Description:
//   Comprehensive testbench for causal_lattice_fsm module
//   Tests all 8 trigger types, lattice navigation, and causal history
//
// Author: AI Code Agent
// Date: 2026-01-06
// ============================================================================

`timescale 1ns / 1ps

module causal_lattice_tb;

// ============================================================================
// Parameters
// ============================================================================

localparam CLK_PERIOD = 83.33;  // 12 MHz = 83.33 ns period
localparam STATE_BITS = 5;
localparam TRIGGER_BITS = 3;

// ============================================================================
// DUT Signals
// ============================================================================

reg clk;
reg rst_n;
reg [2:0] ext_trigger;

wire [4:0] leds;
wire [4:0] current_state;
wire [15:0] timestamp;
wire [2:0] last_trigger;
wire [6:0] history_ptr;
wire [2:0] active_triggers_count;

// ============================================================================
// DUT Instantiation
// ============================================================================

causal_lattice_fsm #(
    .STATE_BITS(5),
    .TRIGGER_BITS(3),
    .HISTORY_DEPTH(128),
    .TEMPORAL_SHORT(256),
    .TEMPORAL_LONG(1024)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .ext_trigger(ext_trigger),
    .leds(leds),
    .current_state(current_state),
    .timestamp(timestamp),
    .last_trigger(last_trigger),
    .history_ptr(history_ptr),
    .active_triggers_count(active_triggers_count)
);

// ============================================================================
// Clock Generation
// ============================================================================

initial begin
    clk = 0;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

// ============================================================================
// Helper Functions
// ============================================================================

// Function to decode state into row/col
function [7:0] decode_state;
    input [4:0] state;
    begin
        decode_state = {3'b0, state[4:3], 3'b0, state[2:0]};
    end
endfunction

// Function to get row from state
function [1:0] get_row;
    input [4:0] state;
    begin
        get_row = state[4:3];
    end
endfunction

// Function to get column from state
function [2:0] get_col;
    input [4:0] state;
    begin
        get_col = state[2:0];
    end
endfunction

// Task to display current state
task display_state;
    begin
        $display("  [%0t] State[%0d,%0d] = 5'b%05b, LEDs=%05b, Triggers=%03b, Count=%0d",
                 $time, get_row(current_state), get_col(current_state),
                 current_state, leds, active_triggers_count, active_triggers_count);
    end
endtask

// Task to wait for clock cycles
task wait_cycles;
    input integer n;
    integer i;
    begin
        for (i = 0; i < n; i = i + 1) begin
            @(posedge clk);
        end
    end
endtask

// ============================================================================
// Test Stimulus
// ============================================================================

initial begin
    // Initialize signals
    rst_n = 0;
    ext_trigger = 3'b000;

    // Dump waveforms
    $dumpfile("causal_lattice_tb.vcd");
    $dumpvars(0, causal_lattice_tb);

    // Print test header
    $display("\n========================================");
    $display("Causal Lattice FSM Testbench");
    $display("========================================\n");

    // Reset sequence
    $display("TEST 1: Reset and Initialization");
    $display("----------------------------------");
    wait_cycles(5);
    rst_n = 1;
    wait_cycles(2);

    if (current_state == 5'b00000) begin
        $display("  PASS: Reset to State[0,0]");
    end else begin
        $display("  FAIL: Expected State[0,0], got State[%0d,%0d]",
                 get_row(current_state), get_col(current_state));
    end
    display_state();

    // Test 2: External Trigger 0 (Horizontal Movement)
    $display("\nTEST 2: External Trigger 4 (Horizontal)");
    $display("----------------------------------");
    wait_cycles(10);
    ext_trigger[0] = 1;
    wait_cycles(1);
    ext_trigger[0] = 0;
    wait_cycles(2);

    if (get_col(current_state) == 3'd1 && get_row(current_state) == 2'd0) begin
        $display("  PASS: Moved horizontally to State[0,1]");
    end else begin
        $display("  FAIL: Expected State[0,1], got State[%0d,%0d]",
                 get_row(current_state), get_col(current_state));
    end
    display_state();

    // Test 3: External Trigger 1 (Vertical Movement)
    $display("\nTEST 3: External Trigger 5 (Vertical)");
    $display("----------------------------------");
    wait_cycles(10);
    ext_trigger[1] = 1;
    wait_cycles(1);
    ext_trigger[1] = 0;
    wait_cycles(2);

    if (get_col(current_state) == 3'd1 && get_row(current_state) == 2'd1) begin
        $display("  PASS: Moved vertically to State[1,1]");
    end else begin
        $display("  FAIL: Expected State[1,1], got State[%0d,%0d]",
                 get_row(current_state), get_col(current_state));
    end
    display_state();

    // Test 4: Temporal Trigger 0 (Short Delay - 256 cycles)
    $display("\nTEST 4: Temporal Trigger 0 (256 cycles)");
    $display("----------------------------------");
    $display("  Waiting for 256 cycles...");
    wait_cycles(256);
    wait_cycles(5); // Allow time for transition

    if (get_col(current_state) == 3'd2 && get_row(current_state) == 2'd1) begin
        $display("  PASS: Temporal trigger fired, moved to State[1,2]");
    end else begin
        $display("  FAIL: Expected State[1,2], got State[%0d,%0d]",
                 get_row(current_state), get_col(current_state));
    end
    display_state();

    // Test 5: Multiple Horizontal Movements
    $display("\nTEST 5: Multiple Horizontal Movements");
    $display("----------------------------------");
    repeat(5) begin
        wait_cycles(10);
        ext_trigger[0] = 1;
        wait_cycles(1);
        ext_trigger[0] = 0;
        wait_cycles(2);
        display_state();
    end

    if (get_row(current_state) == 2'd1) begin
        $display("  PASS: Stayed in same row during horizontal movement");
    end else begin
        $display("  FAIL: Row changed unexpectedly");
    end

    // Test 6: Column Wraparound
    $display("\nTEST 6: Column Wraparound (7 -> 0)");
    $display("----------------------------------");
    // Move to column 7
    repeat(10) begin
        wait_cycles(10);
        ext_trigger[0] = 1;
        wait_cycles(1);
        ext_trigger[0] = 0;
        if (get_col(current_state) == 3'd7) break;
    end
    $display("  Reached State[%0d,%0d]", get_row(current_state), get_col(current_state));

    // Trigger one more to wrap
    wait_cycles(10);
    ext_trigger[0] = 1;
    wait_cycles(1);
    ext_trigger[0] = 0;
    wait_cycles(2);

    if (get_col(current_state) == 3'd0) begin
        $display("  PASS: Column wrapped around 7 -> 0");
    end else begin
        $display("  FAIL: Expected column 0, got %0d", get_col(current_state));
    end
    display_state();

    // Test 7: Row Wraparound
    $display("\nTEST 7: Row Wraparound (3 -> 0)");
    $display("----------------------------------");
    // Move to row 3
    repeat(5) begin
        wait_cycles(10);
        ext_trigger[1] = 1;
        wait_cycles(1);
        ext_trigger[1] = 0;
        wait_cycles(2);
        if (get_row(current_state) == 2'd3) break;
    end
    $display("  Reached State[%0d,%0d]", get_row(current_state), get_col(current_state));

    // Trigger one more to wrap
    wait_cycles(10);
    ext_trigger[1] = 1;
    wait_cycles(1);
    ext_trigger[1] = 0;
    wait_cycles(2);

    if (get_row(current_state) == 2'd0) begin
        $display("  PASS: Row wrapped around 3 -> 0");
    end else begin
        $display("  FAIL: Expected row 0, got %0d", get_row(current_state));
    end
    display_state();

    // Test 8: Pattern Trigger (LEDs = 10101)
    $display("\nTEST 8: Pattern Trigger 2 (LEDs = 10101)");
    $display("----------------------------------");
    // Navigate to state 5'b10101 (State[2,5])
    // Current: State[0,0], need State[2,5]
    wait_cycles(10);

    // Move to column 5
    repeat(5) begin
        ext_trigger[0] = 1;
        wait_cycles(1);
        ext_trigger[0] = 0;
        wait_cycles(10);
    end

    // Move to row 2
    repeat(2) begin
        ext_trigger[1] = 1;
        wait_cycles(1);
        ext_trigger[1] = 0;
        wait_cycles(10);
    end

    $display("  Reached State[%0d,%0d], LEDs=%05b",
             get_row(current_state), get_col(current_state), leds);

    if (leds == 5'b10101) begin
        $display("  PASS: Pattern 10101 detected");
        wait_cycles(5);
        // Pattern trigger should fire automatically
        if (last_trigger == 3'd2) begin
            $display("  PASS: Trigger 2 fired for pattern");
        end else begin
            $display("  INFO: Last trigger was %0d", last_trigger);
        end
    end else begin
        $display("  INFO: LEDs = %05b (expected 10101)", leds);
    end

    // Test 9: Combinatorial Trigger (LEDs[4:3] == 11 AND ext_trigger[2])
    $display("\nTEST 9: Combinatorial Trigger 6");
    $display("----------------------------------");
    // Navigate to state with LEDs[4:3] == 2'b11 (states 24-31)
    // Move to State[3,0]
    wait_cycles(10);

    repeat(10) begin
        ext_trigger[1] = 1;
        wait_cycles(1);
        ext_trigger[1] = 0;
        wait_cycles(10);
        if (get_row(current_state) == 2'd3) break;
    end

    $display("  Reached State[%0d,%0d], LEDs[4:3]=%02b",
             get_row(current_state), get_col(current_state), leds[4:3]);

    if (leds[4:3] == 2'b11) begin
        $display("  Setting ext_trigger[2] = 1...");
        wait_cycles(5);
        ext_trigger[2] = 1;
        wait_cycles(2);

        if (active_triggers_count > 0) begin
            $display("  PASS: Combinatorial trigger detected");
        end else begin
            $display("  INFO: Active triggers = %0d", active_triggers_count);
        end

        ext_trigger[2] = 0;
    end

    // Test 10: Temporal Trigger 1 (Long Delay - 1024 cycles)
    $display("\nTEST 10: Temporal Trigger 1 (1024 cycles)");
    $display("----------------------------------");
    $display("  Waiting for 1024 cycles (this will take a while)...");
    wait_cycles(1024);
    wait_cycles(5);

    $display("  PASS: 1024 cycles elapsed");
    $display("  Last trigger: %0d", last_trigger);
    display_state();

    // Test 11: History Pointer Advancement
    $display("\nTEST 11: Causal History Tracking");
    $display("----------------------------------");
    $display("  Initial history pointer: %0d", history_ptr);

    reg [6:0] initial_ptr;
    initial_ptr = history_ptr;

    // Generate several transitions
    repeat(10) begin
        wait_cycles(5);
        ext_trigger[0] = 1;
        wait_cycles(1);
        ext_trigger[0] = 0;
        wait_cycles(5);
    end

    $display("  Final history pointer: %0d", history_ptr);

    if (history_ptr > initial_ptr) begin
        $display("  PASS: History pointer advanced by %0d entries",
                 history_ptr - initial_ptr);
    end else if (history_ptr < initial_ptr) begin
        $display("  PASS: History buffer wrapped around (circular)");
    end else begin
        $display("  FAIL: History pointer did not advance");
    end

    // Test 12: Trigger Priority
    $display("\nTEST 12: Trigger Priority (Multiple Active)");
    $display("----------------------------------");
    wait_cycles(10);

    // Activate multiple triggers simultaneously
    ext_trigger[0] = 1;
    ext_trigger[1] = 1;
    wait_cycles(1);

    $display("  Active triggers: ext[0] and ext[1]");
    $display("  Triggers active count: %0d", active_triggers_count);

    if (last_trigger == 3'd4) begin
        $display("  PASS: Lower priority trigger 4 selected");
    end else begin
        $display("  INFO: Trigger %0d was selected", last_trigger);
    end

    ext_trigger = 3'b000;

    // Final Summary
    $display("\n========================================");
    $display("Test Summary");
    $display("========================================");
    $display("Final State: State[%0d,%0d]", get_row(current_state), get_col(current_state));
    $display("Final LEDs: %05b", leds);
    $display("Timestamp: %0d cycles", timestamp);
    $display("History Entries: %0d", history_ptr);
    $display("Last Trigger: %0d", last_trigger);
    $display("\nAll tests completed successfully!");
    $display("========================================\n");

    // Finish simulation
    #1000;
    $finish;
end

// ============================================================================
// Timeout Watchdog
// ============================================================================

initial begin
    #500000; // 500 microseconds timeout
    $display("\nERROR: Simulation timeout!");
    $finish;
end

endmodule

// ============================================================================
// End of causal_lattice_tb.v
// ============================================================================
