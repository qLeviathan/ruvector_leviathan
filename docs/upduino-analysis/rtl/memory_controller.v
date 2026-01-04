// =============================================================================
// SPRAM Memory Controller for Weight Storage
// =============================================================================
// Description: Controller for iCE40 UP5K SPRAM blocks (128KB total)
// Author: AI Accelerator Design Team
// Target: UPduino v3.1 (iCE40 UP5K - 4 x 32K SPRAM blocks)
// =============================================================================

module memory_controller #(
    parameter ADDR_WIDTH = 14,     // 16K words x 16-bit
    parameter DATA_WIDTH = 16      // 16-bit data width
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Host interface for weight loading
    input  wire [ADDR_WIDTH-1:0]    host_addr,
    input  wire [DATA_WIDTH-1:0]    host_wdata,
    input  wire                     host_we,
    input  wire                     host_re,
    output reg  [DATA_WIDTH-1:0]    host_rdata,
    output reg                      host_ready,

    // Array interface for weight fetching
    input  wire [ADDR_WIDTH-1:0]    array_addr,
    input  wire                     array_re,
    output reg  [DATA_WIDTH-1:0]    array_rdata,
    output reg                      array_valid
);

    // SPRAM signals
    wire [13:0] spram_addr;
    wire [15:0] spram_wdata;
    wire [15:0] spram_rdata;
    wire [3:0]  spram_maskwren;
    wire        spram_wren;
    wire        spram_chipselect;
    wire        spram_standby;
    wire        spram_sleep;
    wire        spram_poweroff;

    // Arbitration state machine
    localparam IDLE = 2'b00;
    localparam HOST_ACCESS = 2'b01;
    localparam ARRAY_ACCESS = 2'b10;

    reg [1:0] state, next_state;
    reg [ADDR_WIDTH-1:0] access_addr;
    reg access_we;
    reg access_re;
    reg [DATA_WIDTH-1:0] write_data;

    // State machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end

    // Next state logic (priority to host for weight loading)
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (host_we || host_re)
                    next_state = HOST_ACCESS;
                else if (array_re)
                    next_state = ARRAY_ACCESS;
            end
            HOST_ACCESS: begin
                next_state = IDLE;
            end
            ARRAY_ACCESS: begin
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    // Access control
    always @(*) begin
        access_addr = {ADDR_WIDTH{1'b0}};
        access_we = 1'b0;
        access_re = 1'b0;
        write_data = {DATA_WIDTH{1'b0}};

        case (state)
            HOST_ACCESS: begin
                access_addr = host_addr;
                access_we = host_we;
                access_re = host_re;
                write_data = host_wdata;
            end
            ARRAY_ACCESS: begin
                access_addr = array_addr;
                access_re = array_re;
            end
        endcase
    end

    // SPRAM control signals
    assign spram_addr = access_addr;
    assign spram_wdata = write_data;
    assign spram_wren = access_we;
    assign spram_chipselect = access_we || access_re;
    assign spram_maskwren = 4'b1111;  // Write all bytes
    assign spram_standby = 1'b0;
    assign spram_sleep = 1'b0;
    assign spram_poweroff = 1'b1;

    // Instantiate iCE40 SPRAM primitive
    SB_SPRAM256KA spram_inst (
        .DATAIN(spram_wdata),
        .ADDRESS(spram_addr),
        .MASKWREN(spram_maskwren),
        .WREN(spram_wren),
        .CHIPSELECT(spram_chipselect),
        .CLOCK(clk),
        .STANDBY(spram_standby),
        .SLEEP(spram_sleep),
        .POWEROFF(spram_poweroff),
        .DATAOUT(spram_rdata)
    );

    // Output registers
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            host_rdata <= 0;
            host_ready <= 0;
            array_rdata <= 0;
            array_valid <= 0;
        end else begin
            // Host interface
            if (state == HOST_ACCESS) begin
                host_rdata <= spram_rdata;
                host_ready <= 1'b1;
            end else begin
                host_ready <= 1'b0;
            end

            // Array interface
            if (state == ARRAY_ACCESS) begin
                array_rdata <= spram_rdata;
                array_valid <= 1'b1;
            end else begin
                array_valid <= 1'b0;
            end
        end
    end

endmodule
