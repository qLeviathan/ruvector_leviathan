// =============================================================================
// Processing Element (PE) for AI Accelerator
// =============================================================================
// Description: MAC unit with weight stationary dataflow for systolic array
// Author: AI Accelerator Design Team
// Target: UPduino v3.1 (iCE40 UP5K)
// =============================================================================

module processing_element #(
    parameter DATA_WIDTH = 8,      // Activation/Weight bit width
    parameter ACC_WIDTH = 16       // Accumulator bit width
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Data inputs
    input  wire [DATA_WIDTH-1:0]    weight_in,      // Weight input (stationary)
    input  wire [DATA_WIDTH-1:0]    activation_in,  // Activation input (moving)
    input  wire [ACC_WIDTH-1:0]     partial_sum_in, // Partial sum from previous PE

    // Control signals
    input  wire                     weight_load,    // Load new weight
    input  wire                     accumulate,     // Enable accumulation
    input  wire                     clear_acc,      // Clear accumulator

    // Data outputs
    output reg  [DATA_WIDTH-1:0]    activation_out, // Pass activation to next PE
    output reg  [ACC_WIDTH-1:0]     partial_sum_out // MAC result to next PE
);

    // Internal registers
    reg signed [DATA_WIDTH-1:0]     weight_reg;
    reg signed [ACC_WIDTH-1:0]      accumulator;

    // MAC computation wires
    wire signed [DATA_WIDTH-1:0]    act_signed;
    wire signed [DATA_WIDTH-1:0]    weight_signed;
    wire signed [2*DATA_WIDTH-1:0]  product;
    wire signed [ACC_WIDTH-1:0]     mac_result;

    // Sign-extend inputs for signed multiplication
    assign act_signed = activation_in;
    assign weight_signed = weight_reg;

    // Multiply-Accumulate operation
    assign product = act_signed * weight_signed;
    assign mac_result = partial_sum_in + {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};

    // Weight register (stationary)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= 0;
        end else if (weight_load) begin
            weight_reg <= weight_in;
        end
    end

    // Accumulator register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
        end else if (clear_acc) begin
            accumulator <= 0;
        end else if (accumulate) begin
            accumulator <= mac_result;
        end
    end

    // Pipeline activation (pass-through with 1 cycle delay)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            activation_out <= 0;
        end else begin
            activation_out <= activation_in;
        end
    end

    // Output partial sum
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum_out <= 0;
        end else if (accumulate) begin
            partial_sum_out <= mac_result;
        end else begin
            partial_sum_out <= partial_sum_in;
        end
    end

endmodule
