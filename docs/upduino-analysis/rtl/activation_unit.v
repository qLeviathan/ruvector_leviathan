// =============================================================================
// Activation Function Unit
// =============================================================================
// Description: ReLU and tanh approximation for neural network layers
// Author: AI Accelerator Design Team
// Target: UPduino v3.1 (iCE40 UP5K)
// =============================================================================

module activation_unit #(
    parameter DATA_WIDTH = 16,     // Input data width
    parameter OUT_WIDTH = 8        // Output data width (quantized)
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Input data
    input  wire [DATA_WIDTH-1:0]    data_in,
    input  wire                     valid_in,

    // Control
    input  wire [1:0]               activation_type,  // 00: Linear, 01: ReLU, 10: Tanh, 11: Reserved

    // Output data
    output reg  [OUT_WIDTH-1:0]     data_out,
    output reg                      valid_out
);

    // Activation types
    localparam ACT_LINEAR = 2'b00;
    localparam ACT_RELU   = 2'b01;
    localparam ACT_TANH   = 2'b10;

    // Internal signals
    wire signed [DATA_WIDTH-1:0] data_signed;
    reg  signed [DATA_WIDTH-1:0] result;

    assign data_signed = data_in;

    // ReLU implementation
    wire signed [DATA_WIDTH-1:0] relu_out;
    assign relu_out = (data_signed[DATA_WIDTH-1]) ? {DATA_WIDTH{1'b0}} : data_signed;

    // Tanh approximation using piecewise linear function
    // tanh(x) ≈ x for -1 < x < 1
    //         ≈ sign(x) for |x| >= 1
    wire signed [DATA_WIDTH-1:0] tanh_out;
    wire [DATA_WIDTH-2:0] abs_data;
    wire is_saturated;

    assign abs_data = data_signed[DATA_WIDTH-1] ? -data_signed[DATA_WIDTH-2:0] : data_signed[DATA_WIDTH-2:0];
    assign is_saturated = |abs_data[DATA_WIDTH-2:DATA_WIDTH-4];  // Check if |x| >= threshold

    assign tanh_out = is_saturated ?
                      (data_signed[DATA_WIDTH-1] ? {{(DATA_WIDTH-8){1'b1}}, 8'h80} : {{(DATA_WIDTH-8){1'b0}}, 8'h7F}) :
                      data_signed;

    // Activation function selection
    always @(*) begin
        case (activation_type)
            ACT_LINEAR: result = data_signed;
            ACT_RELU:   result = relu_out;
            ACT_TANH:   result = tanh_out;
            default:    result = data_signed;
        endcase
    end

    // Quantization: Scale down from DATA_WIDTH to OUT_WIDTH
    wire [OUT_WIDTH-1:0] quantized;
    wire signed [DATA_WIDTH-1:0] scaled;

    // Simple right shift for quantization (divide by 2^(DATA_WIDTH-OUT_WIDTH))
    assign scaled = result >>> (DATA_WIDTH - OUT_WIDTH);

    // Saturation logic
    assign quantized = scaled[DATA_WIDTH-1] ?
                       (|scaled[DATA_WIDTH-1:OUT_WIDTH-1] ? {1'b1, {(OUT_WIDTH-1){1'b0}}} : scaled[OUT_WIDTH-1:0]) :
                       (|scaled[DATA_WIDTH-1:OUT_WIDTH] ? {1'b0, {(OUT_WIDTH-1){1'b1}}} : scaled[OUT_WIDTH-1:0]);

    // Pipeline registers
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 0;
            valid_out <= 0;
        end else begin
            data_out <= quantized;
            valid_out <= valid_in;
        end
    end

endmodule
