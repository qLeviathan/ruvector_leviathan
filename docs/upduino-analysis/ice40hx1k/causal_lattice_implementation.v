// ============================================================================
// Causal Lattice Finite State Machine for iCE40HX1K
// ============================================================================
//
// Description:
//   Non-linear state machine with 2D lattice structure (4x8 grid = 32 states)
//   Features multiple trigger types and causal history tracking in BRAM.
//
// Target Device: iCE40HX1K (UPduino v3.0)
// Clock: 12 MHz
// Resources: ~70 LUTs, ~30 DFFs, 4096-bit BRAM
//
// Author: AI Code Agent
// Date: 2026-01-06
// ============================================================================

module causal_lattice_fsm #(
    parameter STATE_BITS = 5,           // 5 bits = 32 states
    parameter TRIGGER_BITS = 3,         // 8 trigger types
    parameter HISTORY_DEPTH = 128,      // 128 causal history entries
    parameter HISTORY_ADDR_BITS = 7,    // log2(128) = 7
    parameter COUNTER_BITS = 16,        // 16-bit timestamp counter
    parameter TEMPORAL_SHORT = 256,     // Short delay threshold
    parameter TEMPORAL_LONG = 1024      // Long delay threshold
) (
    // Clock and Reset
    input wire clk,                     // 12 MHz clock
    input wire rst_n,                   // Active-low async reset

    // External Trigger Inputs
    input wire [2:0] ext_trigger,       // 3 external trigger inputs

    // State Outputs
    output reg [4:0] leds,              // 5 LEDs showing current state
    output wire [4:0] current_state,    // Current state value

    // Debug/Status Outputs
    output wire [COUNTER_BITS-1:0] timestamp,        // Current timestamp
    output wire [TRIGGER_BITS-1:0] last_trigger,     // Last active trigger
    output wire [HISTORY_ADDR_BITS-1:0] history_ptr, // History write pointer
    output wire [2:0] active_triggers_count          // Number of active triggers
);

// ============================================================================
// State Encoding and Lattice Structure
// ============================================================================

// State format: {row[1:0], col[2:0]}
// Row: 0-3 (y-axis), Column: 0-7 (x-axis)
reg [STATE_BITS-1:0] state;             // Current state
reg [STATE_BITS-1:0] next_state;        // Next state (combinatorial)
wire [1:0] row;                         // Current row
wire [2:0] col;                         // Current column
wire [1:0] next_row;                    // Next row
wire [2:0] next_col;                    // Next column

assign row = state[4:3];
assign col = state[2:0];
assign current_state = state;

// ============================================================================
// Timing and Counters
// ============================================================================

reg [COUNTER_BITS-1:0] time_counter;    // Global timestamp counter
reg [COUNTER_BITS-1:0] trigger_counter; // Counter for temporal triggers

assign timestamp = time_counter;

// ============================================================================
// Trigger Detection Logic
// ============================================================================

wire [7:0] trigger_active;              // One-hot (can be multiple)
reg [TRIGGER_BITS-1:0] trigger_selected; // Priority-encoded trigger
reg [TRIGGER_BITS-1:0] trigger_prev;    // Previous trigger (for history)

// Trigger 0: Temporal short (256 cycles)
assign trigger_active[0] = (trigger_counter >= TEMPORAL_SHORT);

// Trigger 1: Temporal long (1024 cycles)
assign trigger_active[1] = (trigger_counter >= TEMPORAL_LONG);

// Trigger 2: Pattern match 10101
assign trigger_active[2] = (leds == 5'b10101);

// Trigger 3: Pattern match 01010
assign trigger_active[3] = (leds == 5'b01010);

// Trigger 4: External trigger 0
assign trigger_active[4] = ext_trigger[0];

// Trigger 5: External trigger 1
assign trigger_active[5] = ext_trigger[1];

// Trigger 6: Combinatorial (LEDs[4:3] == 11 AND ext_trigger[2])
assign trigger_active[6] = (leds[4:3] == 2'b11) && ext_trigger[2];

// Trigger 7: History-based (last transition was same-row)
reg history_same_row;
assign trigger_active[7] = history_same_row;

// Count active triggers
assign active_triggers_count =
    trigger_active[0] + trigger_active[1] + trigger_active[2] + trigger_active[3] +
    trigger_active[4] + trigger_active[5] + trigger_active[6] + trigger_active[7];

// Priority encoder: select lowest-numbered active trigger
always @(*) begin
    casez (trigger_active)
        8'bzzzzzzz1: trigger_selected = 3'd0;
        8'bzzzzzz10: trigger_selected = 3'd1;
        8'bzzzzz100: trigger_selected = 3'd2;
        8'bzzzz1000: trigger_selected = 3'd3;
        8'bzzz10000: trigger_selected = 3'd4;
        8'bzz100000: trigger_selected = 3'd5;
        8'bz1000000: trigger_selected = 3'd6;
        8'b10000000: trigger_selected = 3'd7;
        default:     trigger_selected = 3'd0; // No trigger active
    endcase
end

assign last_trigger = trigger_prev;

// ============================================================================
// State Transition Logic (Lattice Navigation)
// ============================================================================

// Compute next row and column based on trigger type
assign next_row = (trigger_selected == 3'd1 || trigger_selected == 3'd3 ||
                   trigger_selected == 3'd5) ? (row + 2'd1) :       // Vertical
                  (trigger_selected == 3'd6) ? (row + 2'd1) :       // Diagonal
                  (trigger_selected == 3'd7) ? col[1:0] :           // Non-linear
                  row;                                               // Horizontal (stay)

assign next_col = (trigger_selected == 3'd0 || trigger_selected == 3'd2 ||
                   trigger_selected == 3'd4) ? (col + 3'd1) :       // Horizontal
                  (trigger_selected == 3'd6) ? (col + 3'd1) :       // Diagonal
                  (trigger_selected == 3'd7) ? {row[1:0], col[2]} : // Non-linear
                  col;                                               // Vertical (stay)

// Combine into next state
always @(*) begin
    if (|trigger_active) begin
        next_state = {next_row, next_col};
    end else begin
        next_state = state; // Hold current state if no trigger
    end
end

// ============================================================================
// Causal History BRAM
// ============================================================================

// History entry format (32 bits):
// [31:16] - timestamp (16 bits)
// [15:11] - previous_state (5 bits)
// [10:8]  - trigger_id (3 bits)
// [7:3]   - next_state (5 bits)
// [2:0]   - active_triggers_count (3 bits)

reg [31:0] history_mem [0:HISTORY_DEPTH-1]; // 128 x 32-bit BRAM
reg [HISTORY_ADDR_BITS-1:0] history_write_ptr; // Circular buffer pointer

assign history_ptr = history_write_ptr;

// History write data
wire [31:0] history_entry;
assign history_entry = {
    time_counter,               // [31:16] timestamp
    state,                      // [15:11] previous state
    trigger_selected,           // [10:8]  trigger ID
    next_state,                 // [7:3]   next state
    active_triggers_count       // [2:0]   trigger count
};

// ============================================================================
// Sequential Logic
// ============================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Reset to state [0,0]
        state <= 5'b00000;
        leds <= 5'b00000;
        time_counter <= 16'd0;
        trigger_counter <= 16'd0;
        history_write_ptr <= 7'd0;
        trigger_prev <= 3'd0;
        history_same_row <= 1'b0;
    end else begin
        // Update timestamp
        time_counter <= time_counter + 16'd1;

        // Update trigger counter
        trigger_counter <= trigger_counter + 16'd1;

        // State transition logic
        if (|trigger_active) begin
            // Trigger active - transition to next state
            state <= next_state;
            leds <= next_state; // LEDs directly show state

            // Record in causal history
            history_mem[history_write_ptr] <= history_entry;
            history_write_ptr <= history_write_ptr + 7'd1; // Circular buffer

            // Reset trigger counter on transition
            trigger_counter <= 16'd0;

            // Save trigger for next cycle
            trigger_prev <= trigger_selected;

            // Update history flag: did we stay in same row?
            history_same_row <= (next_row == row);
        end else begin
            // No trigger - hold state
            state <= state;
            leds <= state;
        end
    end
end

// ============================================================================
// Synthesis Directives (for iCE40)
// ============================================================================

// Force inference of Block RAM for history memory
// synthesis attribute history_mem ram_style "block"

// ============================================================================
// Assertions and Debug (for simulation only)
// ============================================================================

`ifdef SIMULATION
    // Check state stays within valid range
    always @(posedge clk) begin
        if (state > 5'd31) begin
            $error("State out of range: %d", state);
        end
    end

    // Monitor transitions
    always @(posedge clk) begin
        if (|trigger_active && (state != next_state)) begin
            $display("T=%0t: State[%d,%d] --trigger%0d--> State[%d,%d]",
                     $time, row, col, trigger_selected, next_row, next_col);
        end
    end
`endif

endmodule

// ============================================================================
// End of causal_lattice_fsm.v
// ============================================================================
