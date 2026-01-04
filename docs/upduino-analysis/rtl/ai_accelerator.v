// =============================================================================
// AI Inference Accelerator - Top Level Module
// =============================================================================
// Description: Complete AI accelerator with systolic array, memory, and control
// Author: AI Accelerator Design Team
// Target: UPduino v3.1 (iCE40 UP5K)
// Features:
//   - 4x4 Systolic array for matrix multiplication
//   - INT8 quantization support
//   - Weight stationary dataflow
//   - SPRAM-based weight storage
//   - Configurable activation functions
// =============================================================================

module ai_accelerator #(
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 16,
    parameter MEM_ADDR_WIDTH = 14
) (
    // Clock and reset
    input  wire                             clk,
    input  wire                             rst_n,

    // Control interface
    input  wire                             start,
    input  wire [1:0]                       activation_type,
    input  wire [7:0]                       layer_size,      // Number of computations
    output wire                             done,
    output wire                             busy,

    // Memory interface for weight loading
    input  wire [MEM_ADDR_WIDTH-1:0]        weight_addr,
    input  wire [15:0]                      weight_data,
    input  wire                             weight_we,

    // Input feature map interface
    input  wire [ARRAY_SIZE*DATA_WIDTH-1:0] ifmap_data,
    input  wire                             ifmap_valid,
    output wire                             ifmap_ready,

    // Output feature map interface
    output wire [ARRAY_SIZE*DATA_WIDTH-1:0] ofmap_data,
    output wire                             ofmap_valid,
    input  wire                             ofmap_ready
);

    // FSM states
    localparam IDLE = 3'b000;
    localparam LOAD_WEIGHTS = 3'b001;
    localparam COMPUTE = 3'b010;
    localparam ACTIVATE = 3'b011;
    localparam OUTPUT = 3'b100;
    localparam DONE = 3'b101;

    reg [2:0] state, next_state;
    reg [7:0] compute_count;
    reg [7:0] output_count;

    // Internal signals
    wire [ARRAY_SIZE*DATA_WIDTH-1:0]    weight_bus;
    wire [ARRAY_SIZE-1:0]               weight_load_en;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0]     psum_array;
    wire                                array_busy;
    reg                                 compute_enable;
    reg                                 clear_accumulator;

    // Memory controller signals
    wire [MEM_ADDR_WIDTH-1:0]   mem_array_addr;
    wire                        mem_array_re;
    wire [15:0]                 mem_array_rdata;
    wire                        mem_array_valid;

    // Weight loading control
    reg [MEM_ADDR_WIDTH-1:0]    weight_load_addr;
    reg [1:0]                   weight_col_select;

    // Activation unit signals
    wire [ACC_WIDTH-1:0]        act_data_in [0:ARRAY_SIZE-1];
    wire [DATA_WIDTH-1:0]       act_data_out [0:ARRAY_SIZE-1];
    wire [ARRAY_SIZE-1:0]       act_valid_in;
    wire [ARRAY_SIZE-1:0]       act_valid_out;

    // Output buffer
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] output_buffer;
    reg output_valid_reg;

    // =========================================================================
    // Control FSM
    // =========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start)
                    next_state = LOAD_WEIGHTS;
            end

            LOAD_WEIGHTS: begin
                if (weight_col_select == ARRAY_SIZE-1)
                    next_state = COMPUTE;
            end

            COMPUTE: begin
                if (compute_count >= layer_size && !array_busy)
                    next_state = ACTIVATE;
            end

            ACTIVATE: begin
                if (output_count >= ARRAY_SIZE)
                    next_state = OUTPUT;
            end

            OUTPUT: begin
                if (ofmap_ready)
                    next_state = DONE;
            end

            DONE: begin
                next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase
    end

    // Compute counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_count <= 0;
        end else if (state == IDLE) begin
            compute_count <= 0;
        end else if (state == COMPUTE && ifmap_valid) begin
            compute_count <= compute_count + 1;
        end
    end

    // Output counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_count <= 0;
        end else if (state == ACTIVATE) begin
            if (|act_valid_out)
                output_count <= output_count + 1;
        end else begin
            output_count <= 0;
        end
    end

    // Weight column select
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_col_select <= 0;
            weight_load_addr <= 0;
        end else if (state == LOAD_WEIGHTS) begin
            weight_col_select <= weight_col_select + 1;
            weight_load_addr <= weight_load_addr + 1;
        end else begin
            weight_col_select <= 0;
            weight_load_addr <= 0;
        end
    end

    // Control signal generation
    always @(*) begin
        compute_enable = (state == COMPUTE) && ifmap_valid;
        clear_accumulator = (state == IDLE) || (state == LOAD_WEIGHTS);
    end

    // Weight loading
    assign weight_load_en = (state == LOAD_WEIGHTS) ? (1 << weight_col_select) : 4'b0000;
    assign mem_array_addr = weight_load_addr;
    assign mem_array_re = (state == LOAD_WEIGHTS);

    // Pack weights for array (replicate for all rows)
    genvar g;
    generate
        for (g = 0; g < ARRAY_SIZE; g = g + 1) begin : gen_weight_bus
            assign weight_bus[g*DATA_WIDTH +: DATA_WIDTH] =
                mem_array_valid ? mem_array_rdata[DATA_WIDTH-1:0] : 8'h00;
        end
    endgenerate

    // =========================================================================
    // Memory Controller Instance
    // =========================================================================

    memory_controller #(
        .ADDR_WIDTH(MEM_ADDR_WIDTH),
        .DATA_WIDTH(16)
    ) mem_ctrl_inst (
        .clk(clk),
        .rst_n(rst_n),

        // Host interface
        .host_addr(weight_addr),
        .host_wdata(weight_data),
        .host_we(weight_we),
        .host_re(1'b0),
        .host_rdata(),
        .host_ready(),

        // Array interface
        .array_addr(mem_array_addr),
        .array_re(mem_array_re),
        .array_rdata(mem_array_rdata),
        .array_valid(mem_array_valid)
    );

    // =========================================================================
    // Systolic Array Instance
    // =========================================================================

    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) systolic_inst (
        .clk(clk),
        .rst_n(rst_n),

        // Weights
        .weight_in(weight_bus),
        .weight_load(weight_load_en),

        // Activations
        .activation_in(ifmap_data),

        // Control
        .compute_enable(compute_enable),
        .clear_acc(clear_accumulator),

        // Outputs
        .partial_sum_out(psum_array),
        .busy(array_busy)
    );

    // =========================================================================
    // Activation Function Units (one per column)
    // =========================================================================

    generate
        for (g = 0; g < ARRAY_SIZE; g = g + 1) begin : gen_activation
            assign act_data_in[g] = psum_array[g*ACC_WIDTH +: ACC_WIDTH];
            assign act_valid_in[g] = (state == ACTIVATE);

            activation_unit #(
                .DATA_WIDTH(ACC_WIDTH),
                .OUT_WIDTH(DATA_WIDTH)
            ) act_inst (
                .clk(clk),
                .rst_n(rst_n),
                .data_in(act_data_in[g]),
                .valid_in(act_valid_in[g]),
                .activation_type(activation_type),
                .data_out(act_data_out[g]),
                .valid_out(act_valid_out[g])
            );
        end
    endgenerate

    // =========================================================================
    // Output Buffer
    // =========================================================================

    generate
        for (g = 0; g < ARRAY_SIZE; g = g + 1) begin : gen_output
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    output_buffer[g*DATA_WIDTH +: DATA_WIDTH] <= 0;
                end else if (act_valid_out[g]) begin
                    output_buffer[g*DATA_WIDTH +: DATA_WIDTH] <= act_data_out[g];
                end
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_valid_reg <= 0;
        end else begin
            output_valid_reg <= (state == OUTPUT);
        end
    end

    // =========================================================================
    // Output Assignments
    // =========================================================================

    assign done = (state == DONE);
    assign busy = (state != IDLE) && (state != DONE);
    assign ifmap_ready = (state == COMPUTE);
    assign ofmap_data = output_buffer;
    assign ofmap_valid = output_valid_reg;

endmodule
