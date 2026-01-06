# Causal Lattice FSM for iCE40HX1K

A sophisticated non-linear state machine implementation featuring a 2D lattice structure with multiple trigger types and causal history tracking, designed for the Lattice iCE40HX1K FPGA.

## Overview

This design implements a 32-state finite state machine organized as a 4×8 lattice grid. Unlike traditional linear state machines, transitions follow multiple pathways based on 8 different trigger conditions, creating complex causal sequences that can be tracked and analyzed.

### Key Features

- **32-State Lattice**: 4×8 2D grid with wraparound boundaries
- **8 Trigger Types**: Temporal, pattern-based, external, combinatorial, and history-based
- **Causal History**: BRAM-based circular buffer recording 128 transitions
- **LED Display**: 5 LEDs directly display current state in binary
- **Resource Efficient**: ~70 LUTs, ~30 DFFs, 4KB BRAM

## Files

| File | Description |
|------|-------------|
| `causal_lattice_specification.md` | Complete system specification and architecture |
| `causal_lattice_implementation.v` | Verilog RTL implementation |
| `causal_lattice_tb.v` | Comprehensive testbench with 12 test cases |
| `causal_lattice_visualization.py` | Python visualization tool for state diagrams |
| `Makefile` | Build system for simulation, synthesis, and programming |
| `README.md` | This file |

## Quick Start

### Prerequisites

**For Simulation:**
- Icarus Verilog (`iverilog`, `vvp`)
- GTKWave (for waveform viewing, optional)

**For Synthesis:**
- Yosys (open-source synthesis)
- nextpnr-ice40 (place and route)
- icepack (bitstream generation)
- iceprog (programming tool)

**For Visualization:**
```bash
pip install matplotlib numpy
```

### Running Simulation

```bash
# Run testbench
make sim

# View waveforms
make view-sim
# or manually:
gtkwave build/sim/causal_lattice_tb.vcd
```

### Building for Hardware

```bash
# Synthesize design
make synth

# Place and route
make pnr

# Generate bitstream
make bitstream

# Program UPduino v3.0 board
make program
```

### Generating Visualizations

```bash
# Generate state diagrams
make visualize

# Output files:
# - causal_lattice_grid.png (basic lattice)
# - causal_lattice_trigger[0-7].png (each trigger pattern)
# - causal_lattice_all_triggers.png (overview)
# - causal_lattice_trajectory.png (example path)
```

## Architecture

### State Encoding

States are encoded as 5-bit values: `{row[1:0], col[2:0]}`

```
Row 3: [24][25][26][27][28][29][30][31]
Row 2: [16][17][18][19][20][21][22][23]
Row 1: [ 8][ 9][10][11][12][13][14][15]
Row 0: [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]
       Col: 0   1   2   3   4   5   6   7
```

### Trigger System

| ID | Type | Condition | Transition |
|----|------|-----------|------------|
| 0 | Temporal | 256 cycles (21.3μs) | Horizontal right |
| 1 | Temporal | 1024 cycles (85.3μs) | Vertical up |
| 2 | Pattern | LEDs == 10101 | Horizontal right |
| 3 | Pattern | LEDs == 01010 | Vertical up |
| 4 | External | ext_trigger[0] | Horizontal right |
| 5 | External | ext_trigger[1] | Vertical up |
| 6 | Combinatorial | LEDs[4:3]==11 && ext[2] | Diagonal right-up |
| 7 | History | Same-row transition | Non-linear jump |

### Resource Usage

**Logic:**
- LUTs: ~70 / 1,280 (5.5%)
- DFFs: ~30 / 1,280 (2.3%)

**Memory:**
- BRAM: 4,096 / 65,536 bits (6.25%)
- History entries: 128 × 32 bits

**I/O:**
- Inputs: 5 (clk, rst_n, ext_trigger[2:0])
- Outputs: 5 (leds[4:0])

## Testbench Coverage

The testbench (`causal_lattice_tb.v`) includes 12 comprehensive tests:

1. ✅ Reset and initialization
2. ✅ External trigger 0 (horizontal movement)
3. ✅ External trigger 1 (vertical movement)
4. ✅ Temporal trigger 0 (256 cycles)
5. ✅ Multiple horizontal movements
6. ✅ Column wraparound (7 → 0)
7. ✅ Row wraparound (3 → 0)
8. ✅ Pattern trigger (LEDs = 10101)
9. ✅ Combinatorial trigger
10. ✅ Temporal trigger 1 (1024 cycles)
11. ✅ Causal history tracking
12. ✅ Trigger priority (multiple active)

Expected simulation time: ~450μs real-time (54,000 clock cycles @ 12MHz)

## Example Usage

### Example 1: Horizontal Scan

```verilog
// Start at State[0,0], repeatedly trigger external[0]
// Result: State[0,0] → [0,1] → [0,2] → ... → [0,7] → [0,0]
```

### Example 2: Spiral Pattern

```verilog
// Alternate triggers 4 and 5
// State[0,0] → [0,1] (T4) → [1,1] (T5) → [1,2] (T4) → [2,2] (T5) ...
```

### Example 3: Non-Linear Jump

```verilog
// Navigate to State[2,7] (LEDs=10111)
// Trigger 7 (history) fires
// Jump to: {col[1:0], row, col[2]} = {11, 10, 1} = State[3,5]
```

## Advanced Features

### Causal History Analysis

The BRAM stores complete transition history:

```verilog
Entry format (32 bits):
[31:16] - Timestamp (wraps every 5.46ms @ 12MHz)
[15:11] - Previous state
[10:8]  - Trigger ID that fired
[7:3]   - Next state
[2:0]   - Number of active triggers
```

This enables:
- Replay of exact causal sequences
- Pattern recognition in state traversals
- Debugging complex trigger interactions
- Learning optimal trigger strategies

### Visualization Examples

Run `make visualize` to generate:

1. **Lattice Grid**: Basic 4×8 state space with labels
2. **Trigger Patterns**: Individual diagrams for each of 8 triggers
3. **All Triggers**: Overview showing all transition types
4. **Example Trajectory**: Annotated path through state space

## Hardware Deployment (UPduino v3.0)

### Pin Mapping

Create `upduino.pcf` for actual hardware deployment:

```
# Clock (12 MHz oscillator)
set_io clk 35

# LEDs
set_io leds[0] 39  # Red LED
set_io leds[1] 40  # Green LED
set_io leds[2] 41  # Blue LED
set_io leds[3] 42  # GPIO LED 1
set_io leds[4] 43  # GPIO LED 2

# External triggers (GPIO)
set_io ext_trigger[0] 23
set_io ext_trigger[1] 25
set_io ext_trigger[2] 26

# Reset (active-low button)
set_io rst_n 28
```

### Observing Behavior

With 5 LEDs showing the state:
- **LEDs[4:3]**: Row indicator (0-3)
- **LEDs[2:0]**: Column indicator (0-7)

Example: State[2,5] displays as `🔴⚫🔴⚫🔴` (binary 10101)

## Performance Characteristics

### Timing
- Clock frequency: 12 MHz (83.33ns period)
- Trigger detection: 1 cycle (83.33ns)
- State transition: 1 cycle
- History write: 1 cycle
- Total latency: 3 cycles (250ns)

### Trigger Periods
- Short temporal: 21.33μs (256 cycles)
- Long temporal: 85.33μs (1024 cycles)

## Future Enhancements

1. **3D Lattice**: Expand to 4×4×4 cube (64 states)
2. **Adaptive Triggers**: Learn thresholds from history
3. **UART Interface**: Export causal history over serial
4. **Multi-Agent**: Multiple coupled FSMs
5. **Neural Triggers**: On-FPGA neural pattern recognition
6. **Quantum-Inspired**: Probabilistic state superposition

## Troubleshooting

### Simulation Issues

**Problem**: Testbench timeout
- **Solution**: Increase timeout in testbench (default 500μs)

**Problem**: No waveforms generated
- **Solution**: Check that `$dumpfile` and `$dumpvars` are present

### Synthesis Issues

**Problem**: BRAM not inferred
- **Solution**: Check synthesis log, ensure memory size is power of 2

**Problem**: Timing violations
- **Solution**: Reduce clock frequency in constraints

### Hardware Issues

**Problem**: LEDs not changing
- **Solution**: Verify clock is running (12 MHz oscillator)
- Check reset is released (rst_n pulled high)

**Problem**: Erratic behavior
- **Solution**: Check power supply stability
- Verify external trigger inputs are properly grounded when unused

## References

- [iCE40 LP/HX Family Data Sheet](https://www.latticesemi.com/view_document?document_id=49312)
- [UPduino v3.0 Hardware Guide](https://github.com/tinyvision-ai-inc/UPduino-v3.0)
- [Project IceStorm Tools](http://www.clifford.at/icestorm/)
- [Yosys Synthesis Suite](https://yosyshq.net/yosys/)

## License

This implementation is provided as-is for educational and research purposes.

## Author

AI Code Agent
Date: 2026-01-06

---

**For questions or issues, please refer to the specification document (`causal_lattice_specification.md`) for detailed technical information.**
