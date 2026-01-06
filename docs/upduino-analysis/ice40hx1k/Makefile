# ============================================================================
# Makefile for Causal Lattice FSM
# ============================================================================
#
# Targets:
#   make sim        - Run testbench simulation with Icarus Verilog
#   make synth      - Synthesize for iCE40HX1K with Yosys
#   make pnr        - Place and route with nextpnr
#   make bitstream  - Generate bitstream for UPduino v3.0
#   make program    - Program UPduino board via iceprog
#   make visualize  - Generate state diagrams (requires Python/matplotlib)
#   make clean      - Remove generated files
#   make all        - Build everything (sim + synth + pnr + bitstream)
#
# Requirements:
#   - Icarus Verilog (iverilog) for simulation
#   - Yosys for synthesis
#   - nextpnr-ice40 for place-and-route
#   - icepack for bitstream generation
#   - iceprog for programming (optional)
#   - Python 3 with matplotlib for visualization (optional)
#
# Author: AI Code Agent
# Date: 2026-01-06
# ============================================================================

# Target device
DEVICE = hx1k
PACKAGE = vq100
PCF = upduino.pcf

# Source files
RTL_SRC = causal_lattice_implementation.v
TB_SRC = causal_lattice_tb.v
TOP_MODULE = causal_lattice_fsm

# Build directories
BUILD_DIR = build
SIM_DIR = $(BUILD_DIR)/sim
SYNTH_DIR = $(BUILD_DIR)/synth

# Tool configuration
IVERILOG = iverilog
VVP = vvp
YOSYS = yosys
NEXTPNR = nextpnr-ice40
ICEPACK = icepack
ICEPROG = iceprog
PYTHON = python3

# Flags
IVERILOG_FLAGS = -g2012 -DSIMULATION
YOSYS_FLAGS = -q
NEXTPNR_FLAGS = --freq 12

# Output files
SIM_OUT = $(SIM_DIR)/$(TOP_MODULE)_tb.vvp
VCD_OUT = $(SIM_DIR)/causal_lattice_tb.vcd
JSON_OUT = $(SYNTH_DIR)/$(TOP_MODULE).json
ASC_OUT = $(SYNTH_DIR)/$(TOP_MODULE).asc
BIN_OUT = $(SYNTH_DIR)/$(TOP_MODULE).bin

# ============================================================================
# Main Targets
# ============================================================================

.PHONY: all sim synth pnr bitstream program visualize clean help

all: sim synth pnr bitstream

help:
	@echo "Causal Lattice FSM Build System"
	@echo "================================"
	@echo ""
	@echo "Targets:"
	@echo "  make sim        - Run testbench simulation"
	@echo "  make synth      - Synthesize design"
	@echo "  make pnr        - Place and route"
	@echo "  make bitstream  - Generate bitstream"
	@echo "  make program    - Program UPduino board"
	@echo "  make visualize  - Generate visualizations"
	@echo "  make clean      - Clean build files"
	@echo "  make all        - Build everything"
	@echo ""

# ============================================================================
# Simulation
# ============================================================================

sim: $(VCD_OUT)
	@echo "=== Simulation Complete ==="
	@echo "Waveform: $(VCD_OUT)"
	@echo "View with: gtkwave $(VCD_OUT)"

$(VCD_OUT): $(SIM_OUT)
	@mkdir -p $(SIM_DIR)
	@echo "Running testbench..."
	cd $(SIM_DIR) && $(VVP) $(notdir $(SIM_OUT))

$(SIM_OUT): $(RTL_SRC) $(TB_SRC)
	@mkdir -p $(SIM_DIR)
	@echo "Compiling testbench..."
	$(IVERILOG) $(IVERILOG_FLAGS) -o $@ -s causal_lattice_tb $(RTL_SRC) $(TB_SRC)

# ============================================================================
# Synthesis
# ============================================================================

synth: $(JSON_OUT)

$(JSON_OUT): $(RTL_SRC)
	@mkdir -p $(SYNTH_DIR)
	@echo "Synthesizing for iCE40HX1K..."
	$(YOSYS) $(YOSYS_FLAGS) -p "read_verilog -sv $(RTL_SRC); \
	                            synth_ice40 -top $(TOP_MODULE) -json $@"
	@echo "=== Synthesis Complete ==="
	@echo "JSON netlist: $@"

# ============================================================================
# Place and Route
# ============================================================================

pnr: $(ASC_OUT)

$(ASC_OUT): $(JSON_OUT)
	@echo "Place and route..."
	@if [ -f $(PCF) ]; then \
		$(NEXTPNR) --$(DEVICE) --package $(PACKAGE) $(NEXTPNR_FLAGS) \
		           --json $< --pcf $(PCF) --asc $@; \
	else \
		echo "Warning: $(PCF) not found, running without pin constraints"; \
		$(NEXTPNR) --$(DEVICE) --package $(PACKAGE) $(NEXTPNR_FLAGS) \
		           --json $< --asc $@; \
	fi
	@echo "=== Place and Route Complete ==="
	@echo "ASCII netlist: $@"

# ============================================================================
# Bitstream Generation
# ============================================================================

bitstream: $(BIN_OUT)

$(BIN_OUT): $(ASC_OUT)
	@echo "Generating bitstream..."
	$(ICEPACK) $< $@
	@echo "=== Bitstream Complete ==="
	@echo "Binary file: $@"
	@echo "Size: $$(stat -c%s $@) bytes"

# ============================================================================
# Programming
# ============================================================================

program: $(BIN_OUT)
	@echo "Programming UPduino v3.0..."
	$(ICEPROG) $<
	@echo "=== Programming Complete ==="

# ============================================================================
# Visualization
# ============================================================================

visualize:
	@echo "Generating visualizations..."
	$(PYTHON) causal_lattice_visualization.py
	@echo "=== Visualization Complete ==="

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "Cleaning build files..."
	rm -rf $(BUILD_DIR)
	rm -f *.vcd *.png
	@echo "=== Clean Complete ==="

# ============================================================================
# Utility Targets
# ============================================================================

.PHONY: info stats timing view-sim view-synth

info:
	@echo "Project: Causal Lattice FSM"
	@echo "Target: iCE40HX1K-VQ100"
	@echo "Top Module: $(TOP_MODULE)"
	@echo "RTL Source: $(RTL_SRC)"
	@echo "Testbench: $(TB_SRC)"

stats: $(JSON_OUT)
	@echo "Resource Utilization:"
	@$(YOSYS) -p "read_json $<; stat"

timing: $(ASC_OUT)
	@echo "Timing Analysis:"
	@$(NEXTPNR) --$(DEVICE) --package $(PACKAGE) --json $(JSON_OUT) \
	            --asc $< --report $(SYNTH_DIR)/timing.rpt
	@cat $(SYNTH_DIR)/timing.rpt

view-sim: $(VCD_OUT)
	@echo "Opening waveform viewer..."
	gtkwave $< &

view-synth: $(ASC_OUT)
	@echo "Synthesis complete. Use 'icebox_vlog' to view netlist."
	@echo "Example: icebox_vlog $< > $(SYNTH_DIR)/netlist.v"

# ============================================================================
# Dependencies
# ============================================================================

$(SIM_OUT): $(RTL_SRC) $(TB_SRC)
$(JSON_OUT): $(RTL_SRC)
$(ASC_OUT): $(JSON_OUT)
$(BIN_OUT): $(ASC_OUT)
