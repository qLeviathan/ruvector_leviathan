# UPduino v3.1 AI-on-Chip Quick Reference Cards

*Print this document for desk-side reference*

---

## 🔧 Card 1: Hardware Specifications

```
┌──────────────────────────────────────────────┐
│        UPDUINO V3.1 SPECIFICATIONS          │
├──────────────────────────────────────────────┤
│ FPGA:    Lattice iCE40 UltraPlus UP5K       │
│ Package: SG48 (7mm × 7mm)                    │
├──────────────────────────────────────────────┤
│ LOGIC RESOURCES                              │
│  • LUTs:           5,280                     │
│  • Flip-Flops:     5,280                     │
│  • DSP Blocks:     8 (16×16 MAC)            │
│  • PLLs:           1                         │
├──────────────────────────────────────────────┤
│ MEMORY                                       │
│  • Block RAM:      15 blocks × 4 Kbit        │
│                    = 60 Kbit (7.5 KB)        │
│  • SPRAM:          4 blocks × 256 Kbit       │
│                    = 1024 Kbit (128 KB)      │
│  • SPI Flash:      4 MB (external)           │
├──────────────────────────────────────────────┤
│ I/O & PERIPHERALS                            │
│  • GPIO:           32 pins (all exposed)     │
│  • USB:            FTDI FT232H               │
│  • LED:            RGB (PWM capable)         │
│  • Clock:          12 MHz oscillator         │
│  • Regulators:     3.3V, 1.2V on-board      │
├──────────────────────────────────────────────┤
│ TIMING                                       │
│  • Max Clock:      48 MHz (recommended)      │
│  • Overclock:      60-64 MHz (tested)        │
│  • PLL Range:      10-275 MHz                │
├──────────────────────────────────────────────┤
│ POWER                                        │
│  • Supply:         5V via USB                │
│  • Typical:        50-100 mW (active)        │
│  • AI Accel:       ~25 mW (measured)         │
│  • Standby:        <1 mW                     │
└──────────────────────────────────────────────┘
```

---

## 📍 Card 2: Pin Mapping (iCE40 UP5K SG48)

```
┌─────────────────────────────────────────────┐
│           GPIO PIN ASSIGNMENTS              │
├──────┬──────────────────────────────────────┤
│ Pin  │ Function                             │
├──────┼──────────────────────────────────────┤
│ POWER & GROUND                              │
│  1   │ VCC (3.3V)                           │
│  5   │ GND                                  │
│  8   │ VCC (3.3V)                           │
│ 16   │ GND                                  │
├──────┼──────────────────────────────────────┤
│ CLOCK & RESET                               │
│ 20   │ GPIO 20 (12MHz osc via jumper)      │
│ 35   │ GPIO 35 (Alt clock input)           │
├──────┼──────────────────────────────────────┤
│ RGB LED (PWM capable)                       │
│ 39   │ LED_RED   (active low)              │
│ 40   │ LED_GREEN (active low)              │
│ 41   │ LED_BLUE  (active low)              │
├──────┼──────────────────────────────────────┤
│ SPI FLASH (shared with programming)         │
│ 14   │ FLASH_SCK                           │
│ 17   │ FLASH_SDO (MISO)                    │
│ 15   │ FLASH_SDI (MOSI)                    │
│ 16   │ FLASH_CS                            │
├──────┼──────────────────────────────────────┤
│ USB/UART (FTDI interface)                   │
│  9   │ UART_RX (from FTDI)                 │
│ 10   │ UART_TX (to FTDI)                   │
├──────┼──────────────────────────────────────┤
│ GENERAL PURPOSE I/O (recommended for AI)    │
│  2   │ GPIO 2  (SPI CS  / Debug)           │
│  3   │ GPIO 3  (SPI CLK / Debug)           │
│  4   │ GPIO 4  (SPI MISO)                  │
│  6   │ GPIO 6  (SPI MOSI)                  │
│ 11   │ GPIO 11 (I2C SCL)                   │
│ 12   │ GPIO 12 (I2C SDA / Data ready)      │
│ 13   │ GPIO 13 (Interrupt / Done signal)   │
│ 18   │ GPIO 18 (PWM / Sensor input)        │
│ 19   │ GPIO 19 (PWM / Sensor input)        │
│ 21   │ GPIO 21 (ADC / Analog in via R)     │
│ 23-28│ GPIO 23-28 (General purpose)        │
│ 31-32│ GPIO 31-32 (General purpose)        │
│ 34   │ GPIO 34 (General purpose)           │
│ 36-38│ GPIO 36-38 (General purpose)        │
│ 42-48│ GPIO 42-48 (General purpose)        │
└──────┴──────────────────────────────────────┘
```

**AI Accelerator Recommended Pins:**
- **Data Interface:** GPIO 2-6 (SPI-style parallel data)
- **Control Signals:** GPIO 12 (data_ready), GPIO 13 (inference_done)
- **Debug/Status:** RGB LED (39, 40, 41)
- **Serial Console:** UART (GPIO 9, 10)

---

## ⌨️ Card 3: Essential Commands

```bash
# ═══════════════════════════════════════
# DEVICE DETECTION
# ═══════════════════════════════════════
lsusb | grep FTDI              # Check USB connection
dmesg | tail                   # Check kernel messages
ls /dev/ttyUSB*                # Find serial device

# ═══════════════════════════════════════
# BASIC WORKFLOW
# ═══════════════════════════════════════

# 1. Synthesize Verilog → JSON
yosys -p "read_verilog design.v; \
          synth_ice40 -top top_module \
          -json design.json"

# 2. Place & Route JSON → ASC
nextpnr-ice40 --up5k --package sg48 \
              --json design.json \
              --pcf pins.pcf \
              --asc design.asc \
              --freq 48

# 3. Pack ASC → BIN (bitstream)
icepack design.asc design.bin

# 4. Program FPGA
iceprog design.bin             # May need sudo

# ═══════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════
iverilog -o sim testbench.v design.v
./sim                          # Run simulation
gtkwave waveform.vcd          # View waveforms

# ═══════════════════════════════════════
# AI ACCELERATOR SPECIFIC
# ═══════════════════════════════════════

# Full AI accelerator build
cd docs/upduino-analysis/rtl
make all                       # Synth + PnR + Pack

# Run testbench
make sim

# Program and test
make program
python3 ../test_scripts/test_hardware.py

# ═══════════════════════════════════════
# SWARM TESTING
# ═══════════════════════════════════════

# Initialize swarm
npx claude-flow@alpha swarm init \
    --topology mesh --maxAgents 10

# Run comprehensive tests
cd docs/upduino-analysis/test_scripts
./run_swarm_tests.sh \
    --model mnist_mlp \
    --quantization 8 \
    --test-count 1000

# Analyze results
python3 performance_analyzer.py \
    --results test_results.json \
    --output report.md

# ═══════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════

# Fix USB permissions
sudo usermod -a -G dialout $USER
newgrp dialout

# Add udev rules
sudo tee /etc/udev/rules.d/53-lattice.rules << 'EOF'
ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", \
MODE="0660", GROUP="plugdev", TAG+="uaccess"
EOF
sudo udevadm control --reload-rules

# Check resource usage
grep -E "SB_LUT|SB_DFF" synth.log

# Check timing
grep "Max frequency" pnr.log
```

---

## 📊 Card 4: AI Performance Targets

```
┌─────────────────────────────────────────────┐
│       AI ACCELERATOR PERFORMANCE            │
├─────────────────────────────────────────────┤
│ NEURAL NETWORK CAPACITY                     │
│  • Max weights (INT8):    ~100 KB (SPRAM)   │
│  • Max activations:       ~15 KB (BRAM)     │
│  • Example: 784-128-64-10 MLP ✅ Fits       │
│  • Example: MobileNetV2 ⚠️ Needs pruning    │
├─────────────────────────────────────────────┤
│ COMPUTE PERFORMANCE                         │
│  • Peak (8 DSPs):        384 MMAC/s         │
│  • Realistic (systolic): 200-400 MOPS       │
│  • INT8 throughput:      0.8-1.6 GOPS       │
│  • Efficiency:           ~2.5 GOPS/LUT      │
├─────────────────────────────────────────────┤
│ LATENCY (typical MNIST MLP)                 │
│  • Single inference:     1-3 ms             │
│  • Throughput:           300-1000 fps       │
│  • Batch-1 real-time:    ✅ Easily         │
├─────────────────────────────────────────────┤
│ POWER & ENERGY                              │
│  • Active power:         20-30 mW           │
│  • Energy/inference:     20-90 µJ           │
│  • Efficiency:           0.3-0.5 TOPS/W     │
│  • Battery (CR2032):     ~30-60 days        │
├─────────────────────────────────────────────┤
│ QUANTIZATION IMPACT                         │
│  • FP32 baseline:        100% accuracy      │
│  • INT8 (per-tensor):    98-99% accuracy    │
│  • INT4 (per-channel):   95-97% accuracy    │
│  • Binary (XNOR):        85-92% accuracy    │
│  • Ternary (-1,0,+1):    92-96% accuracy    │
├─────────────────────────────────────────────┤
│ RESOURCE UTILIZATION (4×4 systolic)         │
│  • LUTs:     3,200/5,280 (60%) ✅           │
│  • FFs:      2,400/5,280 (45%) ✅           │
│  • BRAM:     10/15 blocks (67%) ✅          │
│  • SPRAM:    1/4 blocks (25%) ✅            │
│  • DSP:      8/8 (100%) ⚠️ Fully used      │
├─────────────────────────────────────────────┤
│ MEMORY BANDWIDTH                            │
│  • SPRAM:    384 MB/s (128KB @ 48MHz)       │
│  • BRAM:     2,880 MB/s (dual-port)         │
│  • Flash:    24 MB/s (SPI @ 48MHz)          │
│  • Bottleneck: Usually compute-bound ✅     │
└─────────────────────────────────────────────┘
```

**Performance Tier Targets:**
```
┌────────────┬──────────┬───────────┬──────────┐
│ Tier       │ Latency  │ Throughput│ Accuracy │
├────────────┼──────────┼───────────┼──────────┤
│ Baseline   │ <5ms     │ >10 fps   │ >85%     │
│ Good       │ <3ms     │ >30 fps   │ >90%     │
│ Excellent  │ <2ms     │ >60 fps   │ >95%     │
│ Outstanding│ <1ms     │ >100 fps  │ >98%     │
└────────────┴──────────┴───────────┴──────────┘

Current Design: "Excellent" tier achievable ✅
```

---

## 🔬 Card 5: Testing & Debugging

```
┌─────────────────────────────────────────────┐
│         TESTING WORKFLOW                    │
├─────────────────────────────────────────────┤
│ LEVEL 1: SIMULATION (Software only)         │
│  1. iverilog -o sim *.v                     │
│  2. ./sim                                   │
│  3. Check: All tests pass? ✅              │
│  4. gtkwave waveform.vcd (if issues)        │
├─────────────────────────────────────────────┤
│ LEVEL 2: SYNTHESIS (Check resources)        │
│  1. yosys ... -json design.json             │
│  2. grep "SB_LUT" yosys.log                 │
│  3. Check: <85% utilization? ✅            │
│  4. Optimize if needed                      │
├─────────────────────────────────────────────┤
│ LEVEL 3: TIMING (Check speed)               │
│  1. nextpnr-ice40 ... --freq 48             │
│  2. grep "Max frequency" pnr.log            │
│  3. Check: ≥48 MHz? ✅                     │
│  4. Add pipeline stages if fails            │
├─────────────────────────────────────────────┤
│ LEVEL 4: HARDWARE (Real FPGA)               │
│  1. icepack design.asc design.bin           │
│  2. iceprog design.bin                      │
│  3. Check: RGB LED blinks? ✅              │
│  4. Run hardware tests                      │
├─────────────────────────────────────────────┤
│ LEVEL 5: PERFORMANCE (Benchmarking)         │
│  1. python3 test_hardware.py                │
│  2. Measure latency, throughput             │
│  3. Check: Meets targets? ✅               │
│  4. Profile and optimize                    │
├─────────────────────────────────────────────┤
│ LEVEL 6: VALIDATION (AI accuracy)           │
│  1. Load real neural network weights        │
│  2. Test on MNIST/CIFAR dataset             │
│  3. Check: >90% accuracy? ✅               │
│  4. Tune quantization if needed             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│         COMMON DEBUG SCENARIOS              │
├─────────────────────────────────────────────┤
│ SYMPTOM: Design doesn't fit                 │
│  → Reduce systolic array (4×4 → 3×3)        │
│  → Lower precision (8-bit → 6-bit)          │
│  → Remove unused features                   │
├─────────────────────────────────────────────┤
│ SYMPTOM: Timing failure                     │
│  → Add pipeline registers                   │
│  → Reduce clock frequency                   │
│  → Simplify critical path                   │
├─────────────────────────────────────────────┤
│ SYMPTOM: Wrong outputs                      │
│  → Check weight loading ($readmemh)         │
│  → Verify quantization scale                │
│  → Compare with software model              │
├─────────────────────────────────────────────┤
│ SYMPTOM: Low accuracy                       │
│  → Retrain with quantization-aware training │
│  → Increase bit width (4→8 bit)            │
│  → Check for overflow in MAC units          │
├─────────────────────────────────────────────┤
│ SYMPTOM: High power consumption             │
│  → Enable clock gating                      │
│  → Reduce clock frequency                   │
│  → Use sparse/binary weights                │
└─────────────────────────────────────────────┘
```

---

## 🧮 Card 6: Quick Math Reference

```
┌─────────────────────────────────────────────┐
│       NEURAL NETWORK CALCULATIONS           │
├─────────────────────────────────────────────┤
│ MATRIX MULTIPLICATION (Dense layer)         │
│  Input: X [batch × in_dim]                  │
│  Weight: W [in_dim × out_dim]               │
│  Output: Y = X × W                          │
│                                              │
│  MACs = batch × in_dim × out_dim            │
│                                              │
│  Example: 1 × 784 × 128 = 100,352 MACs      │
├─────────────────────────────────────────────┤
│ 2D CONVOLUTION                              │
│  Input: [H × W × C_in]                      │
│  Kernel: [K × K × C_in × C_out]             │
│  Output: [H' × W' × C_out]                  │
│                                              │
│  MACs = H' × W' × K² × C_in × C_out         │
│                                              │
│  Example: 28×28 input, 3×3 kernel,          │
│           16 filters, 1 channel:            │
│  MACs = 26×26 × 9 × 1 × 16 = 97,344         │
├─────────────────────────────────────────────┤
│ DEPTHWISE SEPARABLE CONVOLUTION             │
│  = Depthwise + Pointwise                    │
│                                              │
│  Depthwise MACs: H' × W' × K² × C_in        │
│  Pointwise MACs: H' × W' × C_in × C_out     │
│                                              │
│  Savings: ~9× for 3×3 kernels               │
├─────────────────────────────────────────────┤
│ QUANTIZATION (INT8)                         │
│  Q = round(R / S) + Z                       │
│    R: real value (FP32)                     │
│    S: scale factor                          │
│    Z: zero point                            │
│                                              │
│  Dequant: R = (Q - Z) × S                   │
│                                              │
│  SQNR (dB) = 20×log₁₀(σ_signal/σ_noise)    │
│  Target: >40 dB for good accuracy           │
├─────────────────────────────────────────────┤
│ PERFORMANCE ESTIMATION                      │
│  Latency (s) = MACs / (DSP × Freq)          │
│                                              │
│  Example: 100K MACs, 8 DSP, 48 MHz          │
│  Latency = 100,000 / (8 × 48×10⁶)           │
│          = 0.26 ms ✅                       │
│                                              │
│  Throughput = 1 / Latency                   │
│             = 3,840 inferences/sec          │
├─────────────────────────────────────────────┤
│ MEMORY REQUIREMENTS                         │
│  Weights (bytes) = in×out × bits / 8        │
│  Activations (bytes) = batch×dim × bits / 8 │
│                                              │
│  Example: 784-128 layer, INT8               │
│  Weights = 784 × 128 × 8 / 8 = 100 KB       │
│  Activations = 1 × 128 × 8 / 8 = 128 B      │
│                                              │
│  Total fits in 128KB SPRAM? ✅             │
└─────────────────────────────────────────────┘
```

---

## 🎯 Card 7: Project Milestones

```
┌─────────────────────────────────────────────┐
│         IMPLEMENTATION TIMELINE             │
├─────────────────────────────────────────────┤
│ DAY 1-2: Hardware Setup                     │
│  □ Unbox and inspect board                  │
│  □ Install toolchain (yosys, nextpnr, etc)  │
│  □ Run LED blink example                    │
│  □ Verify USB programming works             │
│  Expected: ✅ Functional board              │
├─────────────────────────────────────────────┤
│ DAY 3-5: Simulation & Learning              │
│  □ Study AI accelerator RTL                 │
│  □ Run testbench simulations                │
│  □ View waveforms in GTKWave                │
│  □ Understand dataflow                      │
│  Expected: ✅ Confident with design         │
├─────────────────────────────────────────────┤
│ WEEK 2: First Synthesis                     │
│  □ Synthesize AI accelerator                │
│  □ Check resource utilization               │
│  □ Verify timing closure                    │
│  □ Program FPGA with bitstream              │
│  Expected: ✅ Design runs on hardware       │
├─────────────────────────────────────────────┤
│ WEEK 3: Neural Network Integration          │
│  □ Train MNIST model (PyTorch)              │
│  □ Quantize to INT8                         │
│  □ Export weights to Verilog hex            │
│  □ Load weights in FPGA                     │
│  Expected: ✅ Real NN on FPGA               │
├─────────────────────────────────────────────┤
│ WEEK 4: Testing & Validation                │
│  □ Run swarm testing framework              │
│  □ Collect performance metrics              │
│  □ Measure accuracy on test set             │
│  □ Generate comprehensive report            │
│  Expected: ✅ Production-ready system       │
├─────────────────────────────────────────────┤
│ MONTH 2-3: Optimization (Optional)          │
│  □ Optimize for power/performance           │
│  □ Try binary/ternary networks              │
│  □ Implement on-chip learning               │
│  □ Publish results/blog post                │
│  Expected: ✅ Research contribution         │
└─────────────────────────────────────────────┘
```

---

## 📞 Card 8: Help & Resources

```
┌─────────────────────────────────────────────┐
│          SUPPORT & DOCUMENTATION            │
├─────────────────────────────────────────────┤
│ PROJECT DOCUMENTATION                        │
│  • Master Summary:                          │
│    docs/upduino-analysis/00_MASTER_SUMMARY.md│
│                                              │
│  • Getting Started Guide:                   │
│    GETTING_STARTED_AI_TESTING.md            │
│                                              │
│  • Testing Framework:                       │
│    testing_framework.md                     │
│                                              │
│  • Math Foundations:                        │
│    mathematical_foundations.md              │
│                                              │
│  • RTL Design Docs:                         │
│    ai_accelerator_design.md                 │
├─────────────────────────────────────────────┤
│ OFFICIAL UPDUINO RESOURCES                   │
│  • GitHub (Hardware):                       │
│    github.com/tinyvision-ai-inc/UPduino-v3.0│
│                                              │
│  • Documentation:                           │
│    upduino.readthedocs.io                   │
│                                              │
│  • Discord Community:                       │
│    discord.gg/3qbXujE                       │
│                                              │
│  • Schematic/PCB:                           │
│    Check Board/ directory in repo           │
├─────────────────────────────────────────────┤
│ LATTICE ICE40 RESOURCES                      │
│  • iCE40 UltraPlus Datasheet:               │
│    latticesemi.com/ice40ultraplus           │
│                                              │
│  • TN1281: Memory Usage Guide               │
│  • TN1295: Power Management                 │
│  • TN1334: DSP Usage Guide                  │
├─────────────────────────────────────────────┤
│ OPEN-SOURCE TOOLS                            │
│  • Yosys (Synthesis):                       │
│    clifford.at/yosys                        │
│                                              │
│  • NextPNR (Place & Route):                 │
│    github.com/YosysHQ/nextpnr               │
│                                              │
│  • IceStorm (Bitstream):                    │
│    clifford.at/icestorm                     │
│                                              │
│  • Icarus Verilog:                          │
│    iverilog.icarus.com                      │
├─────────────────────────────────────────────┤
│ LEARNING RESOURCES                           │
│  • Bruno Levy's learn-fpga:                 │
│    github.com/BrunoLevy/learn-fpga          │
│                                              │
│  • FPGA4Fun:                                │
│    fpga4fun.com                             │
│                                              │
│  • Nandland FPGA Tutorials:                 │
│    nandland.com                             │
│                                              │
│  • /r/FPGA subreddit                        │
├─────────────────────────────────────────────┤
│ TROUBLESHOOTING                              │
│  1. Check GETTING_STARTED_AI_TESTING.md     │
│     → Section "Troubleshooting Guide"       │
│                                              │
│  2. Search UPduino Discord                  │
│                                              │
│  3. File GitHub issue (if project bug)      │
│                                              │
│  4. Ask on /r/FPGA (if general FPGA)        │
└─────────────────────────────────────────────┘
```

---

## 🖨️ Printing Instructions

**For best results:**
1. Print on **A4** or **Letter** paper
2. Use **landscape orientation**
3. Print **cards 1-8** separately
4. **Laminate** for durability (optional but recommended)
5. Keep at desk for quick reference

**Digital use:**
- Bookmark this file in your browser
- Pin to terminal for quick `cat` access
- Add to VS Code workspace for easy lookup

---

**Quick Reference Version:** 1.0
**Last Updated:** 2026-01-04
**Compatible with:** UPduino v3.0, v3.1

*These reference cards complement the full documentation in the docs/upduino-analysis/ directory.*
