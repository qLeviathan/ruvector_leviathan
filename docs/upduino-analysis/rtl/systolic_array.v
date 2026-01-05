// =============================================================================
// Systolic Array for Matrix Multiplication
// =============================================================================
// Description: 4x4 PE grid with weight stationary dataflow
// Author: AI Accelerator Design Team
// Target: UPduino v3.1 (iCE40 UP5K)
// =============================================================================

module systolic_array #(
    parameter ARRAY_SIZE = 4,      // 4x4 systolic array
    parameter DATA_WIDTH = 8,      // 8-bit quantized data
    parameter ACC_WIDTH = 16       // 16-bit accumulator
) (
    input  wire                                 clk,
    input  wire                                 rst_n,

    // Weight loading interface (broadcast to columns)
    input  wire [ARRAY_SIZE*DATA_WIDTH-1:0]     weight_in,
    input  wire [ARRAY_SIZE-1:0]                weight_load,

    // Activation inputs (one per row)
    input  wire [ARRAY_SIZE*DATA_WIDTH-1:0]     activation_in,

    // Control signals
    input  wire                                 compute_enable,
    input  wire                                 clear_acc,

    // Partial sum outputs (one per column)
    output wire [ARRAY_SIZE*ACC_WIDTH-1:0]      partial_sum_out,

    // Status
    output reg                                  busy
);

    // Internal activation connections (horizontal)
    wire [DATA_WIDTH-1:0] act_h [0:ARRAY_SIZE-1][0:ARRAY_SIZE];

    // Internal partial sum connections (vertical)
    wire [ACC_WIDTH-1:0]  psum_v [0:ARRAY_SIZE][0:ARRAY_SIZE-1];

    // Input activation assignments
    genvar i, j;
    generate
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : gen_act_input
            assign act_h[i][0] = activation_in[i*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // Initialize partial sums to zero at top of columns
    generate
        for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : gen_psum_init
            assign psum_v[0][j] = {ACC_WIDTH{1'b0}};
        end
    endgenerate

    // Instantiate 4x4 PE array
    generate
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : gen_rows
            for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : gen_cols
                processing_element #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),

                    // Data
                    .weight_in(weight_in[j*DATA_WIDTH +: DATA_WIDTH]),
                    .activation_in(act_h[i][j]),
                    .partial_sum_in(psum_v[i][j]),

                    // Control
                    .weight_load(weight_load[j]),
                    .accumulate(compute_enable),
                    .clear_acc(clear_acc),

                    // Outputs
                    .activation_out(act_h[i][j+1]),
                    .partial_sum_out(psum_v[i+1][j])
                );
            end
        end
    endgenerate

    // Output partial sums from bottom of columns
    generate
        for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : gen_psum_output
            assign partial_sum_out[j*ACC_WIDTH +: ACC_WIDTH] = psum_v[ARRAY_SIZE][j];
        end
    endgenerate

    // Busy status generation
    reg [3:0] compute_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_counter <= 0;
            busy <= 0;
        end else if (clear_acc) begin
            compute_counter <= 0;
            busy <= 0;
        end else if (compute_enable) begin
            if (compute_counter < (ARRAY_SIZE + 2)) begin
                compute_counter <= compute_counter + 1;
                busy <= 1;
            end else begin
                busy <= 0;
            end
        end
    end

endmodule
