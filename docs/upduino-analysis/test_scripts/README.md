# FPGA AI Hardware Testing Scripts

This directory contains comprehensive testing scripts for AI hardware implementations on FPGA platforms (UPDuino v3.1 with Lattice iCE40 UP5K).

## Overview

The testing framework uses **Claude-Flow swarm orchestration** to coordinate multiple specialized agents for parallel test execution, verification, and analysis.

## Scripts

### Main Testing Script

#### `run_swarm_tests.sh`
Comprehensive end-to-end FPGA testing pipeline with swarm coordination.

**Usage:**
```bash
# Full hardware test with 8-bit quantization
./run_swarm_tests.sh --model mnist_cnn --quantization 8 --test-count 1000

# Simulation-only test (no hardware required)
./run_swarm_tests.sh --simulation-only --test-count 100

# Skip synthesis (use existing bitstream)
./run_swarm_tests.sh --skip-synthesis --skip-programming

# Custom topology
./run_swarm_tests.sh --topology hierarchical --quantization 4
```

**Options:**
- `--model MODEL` - Model name (default: mnist_cnn)
- `--quantization BITS` - Quantization level: 4, 8, 16 (default: 8)
- `--test-count COUNT` - Number of test vectors (default: 1000)
- `--topology TOPOLOGY` - Swarm topology: mesh, hierarchical (default: mesh)
- `--skip-synthesis` - Skip RTL synthesis step
- `--skip-programming` - Skip FPGA programming step
- `--simulation-only` - Run simulation tests only (no hardware)
- `--help` - Show help message

**Pipeline Stages:**
1. Initialize Claude-Flow swarm (10 specialized agents)
2. Generate test vectors (random, edge cases, known datasets)
3. RTL synthesis (Yosys)
4. Place & Route (NextPNR)
5. Bitstream generation (IcePack)
6. FPGA programming (iceprog)
7. Test execution (hardware or simulation)
8. Results analysis and reporting
9. Neural pattern training

### Test Generation Scripts

#### `generate_adversarial_tests.py`
Generate adversarial test inputs for robustness testing.

**Usage:**
```bash
python3 generate_adversarial_tests.py \
    --count 100 \
    --bit-width 8 \
    --output adversarial_tests.json
```

**Adversarial Techniques:**
- FGSM (Fast Gradient Sign Method) attacks
- Random Gaussian noise
- Salt-and-pepper noise
- Occlusion attacks
- Rotation perturbations
- Quantization error amplification

### Analysis Scripts

#### `performance_analyzer.py`
Comprehensive performance analysis and report generation.

**Usage:**
```bash
python3 performance_analyzer.py \
    --results test_results.json \
    --output performance_report.md \
    --json-output performance_summary.json
```

**Metrics Analyzed:**
- **Latency**: Mean, median, P95, P99, jitter
- **Throughput**: FPS, inferences/minute
- **Accuracy**: Pass rate, error rate, target compliance
- **Power**: Average power, energy/inference, battery life estimation
- **Health Score**: Overall system health (0-100) with letter grade

## Directory Structure

```
test_scripts/
├── README.md                           # This file
├── run_swarm_tests.sh                  # Main testing pipeline
├── generate_adversarial_tests.py       # Adversarial test generation
└── performance_analyzer.py             # Performance analysis
```

## Dependencies

### Required Tools

**FPGA Toolchain:**
- [Yosys](https://github.com/YosysHQ/yosys) - RTL synthesis
- [NextPNR](https://github.com/YosysHQ/nextpnr) - Place & route
- [IceStorm](https://github.com/YosysHQ/icestorm) - Bitstream tools
- [iceprog](https://github.com/YosysHQ/icestorm) - FPGA programming

**Software:**
- Python 3.8+
- Node.js (for Claude-Flow)
- Claude-Flow: `npm install -g claude-flow@alpha`

**Python Packages:**
```bash
pip install numpy
```

**Optional:**
- pyserial (for UART communication)
- matplotlib (for visualization)

### Installation

#### Ubuntu/Debian
```bash
# Install FPGA tools
sudo apt-get update
sudo apt-get install -y \
    yosys \
    nextpnr-ice40 \
    fpga-icestorm \
    python3 \
    python3-pip \
    nodejs \
    npm

# Install Claude-Flow
npm install -g claude-flow@alpha

# Install Python dependencies
pip3 install numpy pyserial matplotlib
```

#### macOS
```bash
# Install FPGA tools via Homebrew
brew install yosys icestorm nextpnr-ice40

# Install Node.js and npm
brew install node

# Install Claude-Flow
npm install -g claude-flow@alpha

# Install Python dependencies
pip3 install numpy pyserial matplotlib
```

## Hardware Setup

### UPDuino v3.1 Connection

1. **USB Connection**: Connect UPDuino to computer via micro-USB
2. **Verify Detection**:
   ```bash
   lsusb | grep 0403:6014
   ```
   Should show: `Future Technology Devices International`

3. **Set Permissions** (Linux):
   ```bash
   sudo chmod 666 /dev/ttyUSB0
   # Or add user to dialout group (permanent)
   sudo usermod -a -G dialout $USER
   ```

4. **Test Programming**:
   ```bash
   iceprog -t
   ```

## Usage Examples

### Example 1: Full Test Suite with Hardware

```bash
# Run complete test suite on FPGA hardware
./run_swarm_tests.sh \
    --model mnist_cnn \
    --quantization 8 \
    --test-count 1000 \
    --topology mesh
```

**Expected Output:**
```
╔═══════════════════════════════════════════════════════════════╗
║     FPGA AI Hardware Testing - Swarm Orchestration            ║
║     Powered by Claude-Flow Multi-Agent Coordination           ║
╚═══════════════════════════════════════════════════════════════╝

Configuration:
  Model:        mnist_cnn
  Quantization: 8bit
  Test Vectors: 1000
  Topology:     mesh
  Session ID:   fpga-test-1704403200

[2024-01-04 12:00:00] Checking dependencies...
[2024-01-04 12:00:01] All dependencies satisfied
[2024-01-04 12:00:02] Initializing Claude-Flow swarm (topology: mesh)...
[2024-01-04 12:00:05] Swarm initialized with 10 agents
[2024-01-04 12:00:06] Generating test vectors (count: 1000)...
[2024-01-04 12:00:15] Test vectors generated successfully
...
```

### Example 2: Adversarial Testing

```bash
# Generate adversarial tests
python3 generate_adversarial_tests.py \
    --count 200 \
    --bit-width 8 \
    --output adversarial_tests.json

# Run adversarial test suite
./run_swarm_tests.sh \
    --test-vectors adversarial_tests.json \
    --model mnist_cnn
```

### Example 3: Quantization Comparison

```bash
# Test multiple quantization levels
for bits in 4 8 16; do
    ./run_swarm_tests.sh \
        --model mnist_cnn \
        --quantization $bits \
        --test-count 500 \
        --output "results_${bits}bit.json"
done

# Compare results
python3 performance_analyzer.py --results results_4bit.json --output report_4bit.md
python3 performance_analyzer.py --results results_8bit.json --output report_8bit.md
python3 performance_analyzer.py --results results_16bit.json --output report_16bit.md
```

### Example 4: Simulation-Only Testing

```bash
# Run tests in simulation mode (no hardware required)
./run_swarm_tests.sh \
    --simulation-only \
    --test-count 100 \
    --model mnist_cnn
```

## Output Files

After running tests, the following files are generated:

```
docs/upduino-analysis/
├── build/
│   ├── mnist_cnn.json              # Synthesized netlist
│   ├── mnist_cnn.asc               # Placed & routed design
│   ├── mnist_cnn.bin               # FPGA bitstream
│   ├── timing_report.json          # Timing analysis
│   └── test_vectors.json           # Generated test vectors
├── test_results/
│   ├── test_results.json           # Raw test results
│   └── final_report.md             # Human-readable report
└── logs/
    ├── session_<timestamp>.log     # Session log
    ├── synthesis.log               # Synthesis output
    └── pnr.log                     # Place & route output
```

## Performance Targets (UPDuino iCE40 UP5K)

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Inference Latency (P95)** | < 100ms | Real-time responsiveness |
| **Throughput** | > 10 fps | Video processing capability |
| **Power Consumption** | < 50 mW | Battery-powered edge devices |
| **Accuracy (8-bit)** | > 90% | Acceptable quantization loss |
| **LUT Utilization** | 60-80% | Efficient without overcrowding |
| **Resource Efficiency** | > 1 GOPS/LUT | Competitive performance |

## Troubleshooting

### Issue: FPGA not detected

```bash
# Check USB connection
lsusb | grep 0403:6014

# Check permissions
ls -l /dev/ttyUSB0

# Fix permissions
sudo chmod 666 /dev/ttyUSB0
```

### Issue: Synthesis fails with resource overflow

```bash
# Check resource usage
grep "Number of cells" logs/synthesis.log

# Reduce model complexity or increase quantization
./run_swarm_tests.sh --quantization 4  # More aggressive quantization
```

### Issue: Timing violations

```bash
# Check timing report
cat build/timing_report.json

# Reduce clock frequency
# Edit run_swarm_tests.sh and change CLOCK_FREQ_MHZ=10
```

### Issue: Low accuracy

```bash
# Verify golden model
python3 verify_golden_model.py

# Test quantization accuracy
python3 test_quantization.py --bit-width 8

# Enable debug logging
./run_swarm_tests.sh --verbose --debug
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: FPGA Testing CI

on:
  push:
    branches: [ main ]

jobs:
  simulation-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install dependencies
        run: |
          sudo apt-get install -y yosys nextpnr-ice40 icestorm
          npm install -g claude-flow@alpha

      - name: Run simulation tests
        run: |
          cd docs/upduino-analysis/test_scripts
          ./run_swarm_tests.sh --simulation-only --test-count 100

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: docs/upduino-analysis/test_results/
```

## Advanced Features

### Memory Coordination

The testing framework uses Claude-Flow memory for cross-agent coordination:

```bash
# Store test configuration
npx claude-flow@alpha memory store \
    --key "swarm/fpga/test-config" \
    --namespace "coordination" \
    --value '{"model": "mnist_cnn", "quantization": "8bit"}'

# Retrieve results from other agents
npx claude-flow@alpha memory retrieve \
    --key "swarm/synthesis/results" \
    --namespace "coordination"
```

### Neural Pattern Training

Test results are used to train neural patterns for optimization:

```bash
# Train patterns from successful runs
npx claude-flow@alpha neural train \
    --pattern fpga_optimization \
    --data test_results.json

# Query learned patterns
npx claude-flow@alpha neural patterns \
    --query "optimal clock frequency for mnist inference"
```

## Contributing

When adding new test scripts:

1. Follow existing naming conventions
2. Add comprehensive help text
3. Update this README
4. Include usage examples
5. Test with both hardware and simulation modes

## License

See project root LICENSE file.

## Support

For issues and questions:
- GitHub Issues: [ruvector_leviathan/issues](https://github.com/ruvnet/ruvector/issues)
- Documentation: [docs/upduino-analysis/testing_framework.md](../testing_framework.md)
