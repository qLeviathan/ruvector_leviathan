# Causal Lattice FSM Specification for iCE40HX1K

## Overview

A non-linear state machine implementing a 2D lattice structure with multiple trigger types and causal history tracking. Designed for the iCE40HX1K FPGA on UPduino v3.0 board.

## System Architecture

### State Space
- **Total States**: 32 (5-bit encoding)
- **Lattice Structure**: 4×8 2D grid
- **Encoding**: `state[4:0] = {row[1:0], col[2:0]}`
  - Row: 0-3 (y-axis)
  - Column: 0-7 (x-axis)
  - Example: State[2,5] = 5'b10101

### Visual Representation
```
Row 3: [24][25][26][27][28][29][30][31]
         ↕   ↕   ↕   ↕   ↕   ↕   ↕   ↕
Row 2: [16][17][18][19][20][21][22][23]
         ↕   ↕   ↕   ↕   ↕   ↕   ↕   ↕
Row 1: [ 8][ 9][10][11][12][13][14][15]
         ↕   ↕   ↕   ↕   ↕   ↕   ↕   ↕
Row 0: [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]
        ↔   ↔   ↔   ↔   ↔   ↔   ↔   ↔
       Col: 0   1   2   3   4   5   6   7
```

## Trigger System

### 8 Trigger Types (3-bit encoding)

| ID | Type | Condition | Description |
|----|------|-----------|-------------|
| 0 | Temporal | Counter == 256 | Short delay (21.3μs @ 12MHz) |
| 1 | Temporal | Counter == 1024 | Medium delay (85.3μs @ 12MHz) |
| 2 | Pattern | LEDs == 5'b10101 | Alternating pattern match |
| 3 | Pattern | LEDs == 5'b01010 | Inverse alternating pattern |
| 4 | External | ext_trigger[0] | External input 0 |
| 5 | External | ext_trigger[1] | External input 1 |
| 6 | Combinatorial | LEDs[4:3] == 2'b11 && ext_trigger[2] | Complex condition |
| 7 | History | last_row == current_row | Same-row transition detected |

### Trigger Evaluation

Triggers are evaluated every clock cycle (12 MHz = 83.33ns period). Multiple triggers can be active simultaneously; priority is given to the lowest-numbered active trigger.

## Transition Rules

### Lattice Navigation

Transitions follow a structured pattern based on trigger types:

1. **Horizontal Movement** (Triggers 0, 2, 4):
   - Move right: `next_col = (current_col + 1) % 8`
   - Stay in same row

2. **Vertical Movement** (Triggers 1, 3, 5):
   - Move up: `next_row = (current_row + 1) % 4`
   - Stay in same column

3. **Diagonal Movement** (Trigger 6):
   - Move right-up: `{next_row, next_col} = {(row+1)%4, (col+1)%8}`

4. **Non-linear Jump** (Trigger 7):
   - Jump to: `{next_row, next_col} = {col[1:0], row[1:0], col[2]}`
   - Creates causal loops and non-sequential behavior

### Transition Table Structure

Stored as combinatorial logic (FPGA LUTs) due to regular pattern:

```verilog
case(trigger_active)
    3'b000: next_state = {row, col + 1};           // Horizontal
    3'b001: next_state = {row + 1, col};           // Vertical
    3'b010: next_state = {row, col + 1};           // Horizontal
    3'b011: next_state = {row + 1, col};           // Vertical
    3'b100: next_state = {row, col + 1};           // Horizontal
    3'b101: next_state = {row + 1, col};           // Vertical
    3'b110: next_state = {row + 1, col + 1};       // Diagonal
    3'b111: next_state = {col[1:0], row, col[2]};  // Non-linear jump
endcase
```

## Causal History System

### BRAM Structure

**Capacity**: 128 entries (circular buffer)
**Entry Format** (32 bits):
```
[31:16] - timestamp (16-bit counter, wraps every 5.46ms @ 12MHz)
[15:11] - previous_state (5 bits)
[10:8]  - trigger_id (3 bits)
[7:3]   - next_state (5 bits)
[2:0]   - trigger_count (3 bits, how many triggers were active)
```

### History Operations

1. **Write on Transition**: Every state change writes to circular buffer
2. **Read on Request**: History can be read back via debug interface
3. **Overflow Handling**: Oldest entries automatically overwritten
4. **Pointer Management**: 7-bit write pointer tracks current position

### Causality Tracking

The system maintains causality by:
- Recording exact temporal ordering (timestamp)
- Preserving trigger conditions that caused transitions
- Enabling replay/analysis of causal chains
- Supporting pattern recognition in transition sequences

## LED Mapping

5 LEDs directly display current state:
```
LED[4:0] = current_state[4:0]
```

Visual interpretation:
- **LEDs[4:3]**: Row indicator (binary 0-3)
- **LEDs[2:0]**: Column indicator (binary 0-7)

Example: State[2,5] → LEDs = 5'b10101 → 🔴⚫🔴⚫🔴

## Timing Characteristics

### Clock Domain
- **Primary Clock**: 12 MHz (external crystal oscillator)
- **Period**: 83.33 ns
- **All operations**: Single clock domain (no CDC required)

### Trigger Latency
- **Detection**: 1 clock cycle (83.33 ns)
- **State Transition**: 1 clock cycle (83.33 ns)
- **History Write**: 1 clock cycle (83.33 ns)
- **Total Latency**: 3 clock cycles (250 ns) from trigger to history commit

### Temporal Trigger Periods
- **Trigger 0** (256 cycles): 21.33 μs
- **Trigger 1** (1024 cycles): 85.33 μs

## Resource Utilization Estimate

### Logic Resources
- **State Register**: 5 flip-flops
- **Counter**: 16 flip-flops
- **History Pointer**: 7 flip-flops
- **Trigger Logic**: ~50 LUTs (combinatorial)
- **Transition Logic**: ~20 LUTs
- **Total LUTs**: ~70 / 1,280 (5.5%)
- **Total DFFs**: ~30 / 1,280 (2.3%)

### Memory Resources
- **Causal History BRAM**: 4096 bits (128 entries × 32 bits)
- **Total BRAM**: 4096 / 65,536 bits (6.25%)

### I/O
- **Inputs**: 5 (clk, rst_n, ext_trigger[2:0])
- **Outputs**: 5 (leds[4:0])
- **Total I/O**: 10 / 41 available (24.4%)

## Operating Modes

### 1. Normal Operation Mode
- Continuous trigger evaluation
- Automatic state transitions
- Real-time history recording

### 2. Debug Mode (Future Extension)
- Freeze state machine
- Read causal history
- Manual state injection

### 3. Pattern Learning Mode (Future Extension)
- Record frequent transition patterns
- Optimize trigger conditions
- Adaptive lattice reconfiguration

## Non-Linear Behavior Examples

### Example 1: Causal Loop
```
State[0,0] --trigger0--> State[0,1] --trigger0--> State[0,2]
                                                      |
                                                  trigger7
                                                      ↓
State[0,0] <--trigger0-- State[0,5] <--trigger0-- State[0,4]
```

Creates a deterministic loop with 5 states, period depends on trigger timing.

### Example 2: Emergent Pattern
```
Start: State[0,0]
t=0μs:    trigger4 → State[0,1]
t=21μs:   trigger0 → State[0,2]  (temporal fired)
t=42μs:   trigger0 → State[0,3]
t=63μs:   trigger0 → State[0,4]  (LEDs=5'b00100)
t=84μs:   trigger0 → State[0,5]  (LEDs=5'b00101, matches trigger2)
t=84μs:   trigger2 → State[0,6]  (pattern trigger fires immediately)
```

Pattern trigger creates acceleration in traversal.

### Example 3: Non-Deterministic Jump
```
State[2,7] (LEDs=5'b10111)
  - trigger7 active (history-based)
  - Next state: {col[1:0], row, col[2]} = {2'b11, 2'b10, 1'b1} = State[3,5]
  - Jumped from row 2 to row 3, col 7 to col 5
```

Creates non-sequential causal dependencies.

## Verification Strategy

### Testbench Coverage
1. **Reset Behavior**: Verify state returns to [0,0]
2. **All 8 Triggers**: Test each trigger type individually
3. **Trigger Priority**: Verify lowest-numbered trigger wins
4. **Lattice Boundaries**: Test wraparound (col 7→0, row 3→0)
5. **History Recording**: Verify all transitions logged correctly
6. **Timing**: Verify temporal triggers at correct intervals

### Expected Patterns
- Horizontal scan: triggers 0,2,4 should traverse columns
- Vertical scan: triggers 1,3,5 should traverse rows
- Complex paths: trigger 6,7 should create non-linear sequences

## Future Enhancements

1. **Adaptive Triggers**: Learn optimal trigger thresholds from history
2. **3D Lattice**: Expand to 4×4×4 cube (64 states with 6-bit encoding)
3. **Multi-Agent**: Multiple FSMs with causal coupling
4. **UART Interface**: Export causal history for offline analysis
5. **Neural Triggers**: Train neural network on FPGA for pattern triggers
6. **Quantum-Inspired**: Superposition states with probabilistic triggers

## References

- iCE40 LP/HX Family Data Sheet (Lattice Semiconductor)
- UPduino v3.0 Hardware User Guide
- Verilog HDL Synthesis (Clifford E. Cummings)
- Digital Design and Computer Architecture (Harris & Harris)

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-06 | AI Code Agent | Initial specification |
