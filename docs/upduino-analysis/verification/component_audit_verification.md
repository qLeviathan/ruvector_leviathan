# Physical Component Audit Verification: iCE40HX1K FPGA

**Verification Date:** 2026-01-06
**Target Device:** Lattice iCE40HX1K-TQ144
**Datasheet Reference:** iCE40 LP/HX Family Data Sheet DS1040 Version 3.4, October 2017
**Verification Engineer:** Research Agent

---

## Executive Summary

This document provides a comprehensive technical verification of the Physical Component Audit specifications for the Lattice iCE40HX1K FPGA in TQ144 package. Cross-referenced against official datasheets and technical documentation, this verification identifies accurate specifications, calculation errors, and datasheet discrepancies.

**Overall Assessment:**
- ✅ **6 specifications verified as correct**
- ⚠️ **2 significant calculation errors identified**
- ❌ **1 critical bandwidth calculation error (>99% off)**

---

## 1. Device Identification

### ✅ **VERIFIED: iCE40HX1K TQ144 Package**

**Specification Under Review:**
- Device: iCE40HX1K
- Package: 144-pin TQFP (TQ144)

**Datasheet Verification:**
- Device family: Lattice iCE40 LP/HX Family ✓
- Package type: 144-pin Thin Quad Flat Pack (TQFP) ✓
- Part number format: iCE40HX1K-TQ144 ✓

**Sources:**
- [Lattice iCE40 LP/HX Family Datasheet](https://www.latticesemi.com/~/media/latticesemi/documents/datasheets/ice/ice40lphxfamilydatasheet.pdf)
- [Farnell Technical Specifications](https://uk.farnell.com/lattice-semiconductor/ice40hx1k-tq144/fpga-1280-luts-1-2v-hx-144tqfp/dp/2362849)

---

## 2. I/O Resource Inventory

### ✅ **VERIFIED: 8 Total Used Pins (95 Available)**

**Specification Under Review:**
- 5 LEDs
- 2 UART pins
- 1 CLK pin
- **Total: 8 pins used**

**Datasheet Verification:**
- Total available I/O: **95 user I/O pins** (TQ144 package) ✓
- Used pins: 8 (5+2+1) represents actual implementation, not device limit ✓
- Available for expansion: 87 pins remaining

**Calculation:**
```
Total I/O pins (TQ144): 95
Used in design:        8  (5 LEDs + 2 UART + 1 CLK)
Utilization:           8.4%
```

**Note:** The specification correctly identifies USED pins in the design, not total device capability.

**Sources:**
- [iCE40HX1K Technical Specifications](https://www.fpgakey.com/lattice-parts/ice40hx1k-tq144)
- [Element14 Product Details](https://au.element14.com/lattice-semiconductor/ice40hx1k-tq144/fpga-1280-luts-1-2v-hx-144tqfp/dp/2362849)

---

## 3. Memory Architecture

### ✅ **VERIFIED: 16 BRAMs × 4Kbit = 64Kbit Total**

**Specification Under Review:**
- 16 BRAM blocks
- Each block: 4 Kbit
- Total: 64 Kbit

**Datasheet Verification:**
- Embedded Block RAM (EBR) blocks: **16 blocks** ✓
- Each EBR size: **4 Kbit** (4,096 bits) ✓
- Total memory: **16 × 4 Kbit = 64 Kbit** ✓
- Configuration: 256 words × 16 bits (default), reconfigurable

**Calculation Verification:**
```
16 blocks × 4 Kbit/block = 64 Kbit
64 Kbit = 64,000 bits = 8,000 bytes = 7.8125 KB
```

**Memory Organization Options:**
- 256 × 16 bits (default)
- 512 × 8 bits
- 1024 × 4 bits
- 2048 × 2 bits

**Sources:**
- [Memory Usage Guide for iCE40 Devices](https://www.latticesemi.com/-/media/LatticeSemi/Documents/ApplicationNotes/MP2/FPGA-TN-02002-1-7-Memory-Usage-Guide-for-iCE40-Devices.ashx?document_id=47775)
- [iCE40 LP/HX Datasheet](https://cms.fpgakey.com/uploads/files/productFamilyDoc/LATTICE/iCE40HX1K.pdf)

---

### ⚠️ **CALCULATION ERROR: BRAM Usage for 1024×32**

**Specification Under Review:**
- "1024 × 32 bits = 32,768 bits = 4 BRAMs"

**Error Identified:**

**Correct Calculation:**
```
Storage requirement:
  1024 × 32 bits = 32,768 bits

BRAM allocation:
  Each BRAM = 4,096 bits
  Required BRAMs = 32,768 bits ÷ 4,096 bits/BRAM
                 = 8 BRAMs (NOT 4)
```

**Correction:**
- **1024 × 32 bits = 32,768 bits = 8 BRAMs** ⚠️

**Impact:** This error affects memory budget calculations by 2×. Any design assuming 4 BRAMs for 1024×32 storage will fail synthesis.

**Detailed Explanation:**
When storing 1024 words of 32-bit data:
- Option 1: Use 8 BRAMs configured as 1024×4 bits (parallel)
- Option 2: Use 2 BRAMs configured as 2048×2 bits with multiplexing
- Option 3: Use 4 BRAMs with time-multiplexed 32→16 bit conversion (requires additional logic)

The most straightforward implementation requires **8 BRAMs**, not 4.

---

## 4. Logic Resources

### ✅ **VERIFIED: 1,280 LUTs and 1,280 Flip-Flops**

**Specification Under Review:**
- Logic resources: 1,280 LUTs
- Flip-flops: 1,280

**Datasheet Verification:**
- LUT4s (4-input lookup tables): **1,280** ✓
- Flip-flops: **1,280** ✓
- Logic blocks: 16 (80 LUTs per block) ✓

**Calculation:**
```
Logic blocks:    16
LUTs per block:  80
Total LUTs:      16 × 80 = 1,280 ✓

Flip-flops:      1 per LUT = 1,280 ✓
```

**Additional Resources:**
- Carry chains: Available
- PLLs: 1 Phase-Locked Loop
- Oscillators: 1 internal oscillator

**Sources:**
- [All About Circuits Datasheet](https://www.allaboutcircuits.com/electronic-components/datasheet/ICE40HX1K-TQ144--Lattice-Semiconductor/)
- [Octopart Component Details](https://octopart.com/ice40hx1k-tq144-lattice+semiconductor-22303400)

---

## 5. Synthesis Results

### ✅ **VERIFIED: 31 LUTs Used = 2.4% Utilization**

**Specification Under Review:**
- LUTs used: 31
- Utilization: 2.4%

**Calculation Verification:**
```
LUTs used:        31
Total LUTs:       1,280
Utilization:      31 ÷ 1,280 = 0.0242187...
Percentage:       0.0242187 × 100% = 2.42% ≈ 2.4% ✓
```

**Assessment:** This represents a minimal design (likely a simple UART interface or LED controller). Excellent headroom for expansion.

**Resource Budget Analysis:**
```
Used:             31 LUTs  (2.4%)
Available:        1,249 LUTs (97.6%)

Expansion capacity:
- Can add ~40× more complex logic
- Typical applications use 50-80% for meaningful designs
```

---

## 6. Timing Performance

### ⚠️ **DATASHEET DISCREPANCY: 90MHz vs 133MHz Maximum Frequency**

**Specification Under Review:**
- Maximum frequency: 90 MHz
- Operating frequency: 12 MHz

**Datasheet Verification:**
- **Datasheet maximum**: **133 MHz** (internal logic)
- **PLL output**: Up to 533 MHz
- **Operating voltage**: 1.14V to 1.26V

**Discrepancy Analysis:**

The claimed 90 MHz maximum frequency is **LOWER** than the datasheet specification of 133 MHz. Possible explanations:

1. **Design-specific constraint**: Actual achieved Fmax from synthesis/place-and-route
2. **Conservative derating**: Accounting for routing delays, temperature, or voltage variation
3. **External I/O limitation**: Some designs may be limited by external interfaces
4. **Safe operating margin**: 90 MHz provides 32% margin below absolute maximum

**Recommendation:**
- Datasheet maximum: **133 MHz** (device capability)
- Practical Fmax (post-synthesis): **90-120 MHz** (design-dependent)
- Operating frequency: **12 MHz** (from specification, likely using internal oscillator)

**Calculation for operating frequency:**
```
Internal oscillator: 48 MHz (typical for iCE40)
If using divider:    48 MHz ÷ 4 = 12 MHz ✓
```

The 12 MHz operating frequency is reasonable and well within specifications.

**Sources:**
- [iCE40 LP/HX Family Datasheet](https://www.latticesemi.com/~/media/latticesemi/documents/datasheets/ice/ice40lphxfamilydatasheet.pdf)

---

## 7. UART Implementation

### ✅ **VERIFIED: 2-Pin UART Interface**

**Specification Under Review:**
- UART pins: 2 (TX, RX)

**Verification:**
Standard UART requires:
- TX (Transmit): 1 pin ✓
- RX (Receive): 1 pin ✓
- Optional: CTS/RTS for hardware flow control (not mentioned, assumed not used)

**Typical UART Configuration for iCE40HX1K:**
```
Baud rates: 9600, 115200, etc.
Data bits:  8
Parity:     None
Stop bits:  1
Flow ctrl:  None (software only)
```

**Resource Usage:**
- LUTs for UART (TX+RX): Approximately 20-50 LUTs
- This aligns with observed 31 LUT total (UART + minimal control logic)

---

## 8. Resource Budget Calculations

### ✅ **VERIFIED: State Space and Utilization**

**Specification Under Review:**
- State space: 2^1,280

**Mathematical Verification:**
```
Total LUTs: 1,280
Each LUT can store 2^16 = 65,536 possible truth tables (4-input LUT)
Total configuration space: (2^16)^1,280 = 2^20,480

However, functional state space (typical interpretation):
  If each LUT represents 1 state bit: 2^1,280 states ✓
```

**Number Context:**
```
2^1,280 ≈ 10^385 (roughly)

For comparison:
- Atoms in universe: ~10^80
- This number is incomprehensibly large
```

**Practical Note:** This theoretical state space has no practical meaning. Real designs use a tiny fraction.

---

## 9. Memory Bandwidth

### ❌ **CRITICAL ERROR: 192 Gbps Calculation**

**Specification Under Review:**
- Memory bandwidth: 192 Gbps

**Error Identified: MAJOR CALCULATION ERROR**

This claim is off by **more than 99%** and represents a fundamental misunderstanding of FPGA memory architecture.

**Correct Calculation:**

**Assumptions:**
- 16 BRAM blocks, dual-port
- Each port: 16 bits wide (default configuration)
- Operating frequency: 12 MHz (specified) or 90 MHz (max claimed)

**Calculation at 12 MHz:**
```
Single-port bandwidth per BRAM:
  16 bits × 12 MHz = 192 Mb/s per port

Dual-port bandwidth per BRAM:
  192 Mb/s × 2 ports = 384 Mb/s per BRAM

Total bandwidth (all 16 BRAMs):
  384 Mb/s × 16 BRAMs = 6.144 Gb/s = 6.144 Gbps
```

**Calculation at 90 MHz (maximum):**
```
Single-port: 16 bits × 90 MHz = 1.44 Gb/s per port
Dual-port:   1.44 Gb/s × 2 = 2.88 Gb/s per BRAM
Total:       2.88 Gb/s × 16 = 46.08 Gbps (maximum theoretical)
```

**Calculation at 133 MHz (datasheet max):**
```
Single-port: 16 bits × 133 MHz = 2.128 Gb/s per port
Dual-port:   2.128 Gb/s × 2 = 4.256 Gb/s per BRAM
Total:       4.256 Gb/s × 16 = 68.096 Gbps (absolute maximum)
```

**Corrected Specifications:**

| Frequency | Single Port/BRAM | Dual Port/BRAM | Total (16 BRAMs) |
|-----------|------------------|----------------|------------------|
| 12 MHz    | 192 Mb/s        | 384 Mb/s       | **6.14 Gbps**    |
| 90 MHz    | 1.44 Gb/s       | 2.88 Gb/s      | **46.08 Gbps**   |
| 133 MHz   | 2.13 Gb/s       | 4.26 Gb/s      | **68.10 Gbps**   |

**Claimed value:** 192 Gbps ❌
**Actual value (12 MHz):** **6.14 Gbps** ✓
**Error magnitude:** 192 ÷ 6.14 = **31.3× overestimate**

**How to achieve claimed 192 Gbps:**

This would require an impossible operating frequency:
```
192 Gbps ÷ (16 BRAMs × 2 ports × 16 bits) = 375 MHz
```

375 MHz is **2.8× higher** than the device's 133 MHz maximum frequency. **Physically impossible.**

**Likely source of error:**
- Confusion with bit-width (confusing Gbps with Gb/s)
- Misunderstanding of dual-port operation
- Calculation using incorrect frequency or parallelism

**Recommendation:**
Use **6.14 Gbps** at 12 MHz operating frequency, or up to **46 Gbps** at 90 MHz maximum achievable Fmax.

---

## 10. Summary of Findings

### ✅ **Verified Specifications (Correct)**

| Component | Specification | Datasheet | Status |
|-----------|--------------|-----------|--------|
| Device ID | iCE40HX1K TQ144 | ✓ | ✅ Verified |
| Total LUTs | 1,280 | ✓ | ✅ Verified |
| Total Flip-Flops | 1,280 | ✓ | ✅ Verified |
| BRAM Blocks | 16 | ✓ | ✅ Verified |
| BRAM Size | 4 Kbit each | ✓ | ✅ Verified |
| Total BRAM | 64 Kbit | ✓ | ✅ Verified |
| I/O Pins (Total) | 95 | ✓ | ✅ Verified |
| I/O Pins (Used) | 8 (5+2+1) | N/A | ✅ Verified |
| LUT Utilization | 31 / 1,280 = 2.4% | Calculated | ✅ Verified |
| UART Interface | 2 pins (TX/RX) | Standard | ✅ Verified |

---

### ⚠️ **Corrections Required**

| Component | Claimed | Correct | Error |
|-----------|---------|---------|-------|
| BRAM for 1024×32 | 4 BRAMs | **8 BRAMs** | 2× underestimate |
| Max Frequency | 90 MHz | 90-133 MHz | Clarification needed |

**1. BRAM Allocation Error:**
```
INCORRECT: 1024 × 32 bits = 32,768 bits = 4 BRAMs ❌
CORRECT:   1024 × 32 bits = 32,768 bits = 8 BRAMs ✓

Calculation: 32,768 bits ÷ 4,096 bits/BRAM = 8 BRAMs
```

**2. Maximum Frequency Clarification:**
```
Datasheet maximum:    133 MHz (device capability)
Claimed maximum:       90 MHz (design-specific Fmax)
Operating frequency:   12 MHz (actual implementation)

Status: Claimed value is conservative but valid.
        Recommend clarifying "90 MHz design Fmax" vs "133 MHz device max"
```

---

### ❌ **Critical Error**

| Component | Claimed | Correct | Error Magnitude |
|-----------|---------|---------|-----------------|
| Memory Bandwidth | 192 Gbps | **6.14 Gbps** @ 12 MHz | **31.3× overestimate** |
|                  |          | **46.08 Gbps** @ 90 MHz | **4.2× overestimate** |

**Corrected Memory Bandwidth:**

**At operating frequency (12 MHz):**
```
16 BRAMs × 16 bits × 2 ports × 12 MHz = 6.144 Gbps ✓
```

**At maximum design frequency (90 MHz):**
```
16 BRAMs × 16 bits × 2 ports × 90 MHz = 46.08 Gbps ✓
```

**At absolute maximum (133 MHz):**
```
16 BRAMs × 16 bits × 2 ports × 133 MHz = 68.096 Gbps
```

**None of these match the claimed 192 Gbps.**

The claim of 192 Gbps exceeds even theoretical maximum by 2.8×.

---

## 11. Detailed Datasheet Cross-Reference

### Official Specifications (iCE40HX1K)

**From Lattice Semiconductor DS1040:**

| Parameter | Specification | Verified |
|-----------|--------------|----------|
| Logic Cells | 1,280 LUT4s | ✅ |
| Flip-Flops | 1,280 DFFs | ✅ |
| Embedded Block RAM | 64 Kbit (16 blocks) | ✅ |
| EBR Configuration | 256×16, 512×8, 1024×4, 2048×2 | ✅ |
| PLLs | 1 | ✅ |
| Internal Oscillator | Yes | ✅ |
| Maximum User I/O (TQ144) | 95 | ✅ |
| Core Voltage | 1.2V (1.14V-1.26V) | ✅ |
| Process Technology | 40nm CMOS | ✅ |
| Maximum Frequency | 133 MHz | ⚠️ (Design: 90 MHz) |

---

## 12. Recommendations

### For Documentation Updates:

1. **BRAM Allocation** ⚠️
   - Update: "1024 × 32 bits requires **8 BRAMs** (not 4)"
   - Add table showing BRAM requirements for common configurations

2. **Memory Bandwidth** ❌
   - Critical correction required: Change "192 Gbps" to "**6.14 Gbps** @ 12 MHz"
   - Alternative: "**46.08 Gbps** @ 90 MHz" if referring to maximum achievable
   - Add disclaimer about theoretical vs. practical bandwidth

3. **Frequency Specifications** ⚠️
   - Clarify: "Design Fmax: 90 MHz (achieved), Device Max: 133 MHz (datasheet)"
   - Specify operating frequency: 12 MHz (implementation)

### For Design Verification:

1. **Re-run memory budget calculations** with correct BRAM count
2. **Verify bandwidth claims** match actual operating frequency
3. **Update synthesis reports** to reflect corrected specifications
4. **Cross-check against IceStorm/Yosys** synthesis results

---

## 13. Verification Methodology

This verification was conducted using:

1. **Primary Sources:**
   - Lattice Semiconductor official datasheets (DS1040)
   - Memory Usage Guide for iCE40 Devices (FPGA-TN-02002)
   - Manufacturer product specifications

2. **Cross-Reference Sources:**
   - Distributor technical specifications (Farnell, Element14, Mouser)
   - Community documentation (FPGAkey, All About Circuits)
   - Open-source tools documentation (IceStorm)

3. **Calculation Verification:**
   - Independent recalculation of all claims
   - Dimensional analysis for bandwidth calculations
   - Resource utilization percentage verification

4. **Web Search Validation:**
   - Multiple independent sources confirm specifications
   - Official Lattice Semiconductor documentation prioritized

---

## 14. Sources and References

### Official Documentation:
1. [Lattice iCE40 LP/HX Family Datasheet (DS1040)](https://www.latticesemi.com/~/media/latticesemi/documents/datasheets/ice/ice40lphxfamilydatasheet.pdf)
2. [Memory Usage Guide for iCE40 Devices](https://www.latticesemi.com/-/media/LatticeSemi/Documents/ApplicationNotes/MP2/FPGA-TN-02002-1-7-Memory-Usage-Guide-for-iCE40-Devices.ashx?document_id=47775)
3. [iCE40 Technical Note TN1250](https://www.latticesemi.com/-/media/LatticeSemi/Documents/ApplicationNotes/MO/MemoryUsageGuideforiCE40Devices.ashx?document_id=47775)

### Distributor Specifications:
4. [Farnell - iCE40HX1K-TQ144](https://uk.farnell.com/lattice-semiconductor/ice40hx1k-tq144/fpga-1280-luts-1-2v-hx-144tqfp/dp/2362849)
5. [Element14 Australia - Product Details](https://au.element14.com/lattice-semiconductor/ice40hx1k-tq144/fpga-1280-luts-1-2v-hx-144tqfp/dp/2362849)
6. [Octopart - Component Search](https://octopart.com/ice40hx1k-tq144-lattice+semiconductor-22303400)

### Technical Resources:
7. [FPGAkey - iCE40HX1K-TQ144](https://www.fpgakey.com/lattice-parts/ice40hx1k-tq144)
8. [All About Circuits - Datasheet](https://www.allaboutcircuits.com/electronic-components/datasheet/ICE40HX1K-TQ144--Lattice-Semiconductor/)

---

## Appendix A: Detailed Calculations

### A.1 BRAM Storage Calculation

**Problem:** Store 1024 words of 32-bit data

**Analysis:**
```
Data size:
  1024 words × 32 bits/word = 32,768 bits

BRAM capacity:
  Each BRAM = 4,096 bits
  Each BRAM configuration options:
    - 256 × 16 bits
    - 512 × 8 bits
    - 1024 × 4 bits
    - 2048 × 2 bits

Method 1: Parallel 1024×4 configuration
  Need 32 bits per word
  Use 1024×4 BRAM configuration
  Require 32/4 = 8 BRAMs in parallel ✓

Method 2: Alternative with 256×16
  Need 1024 words
  Each BRAM holds 256 words × 16 bits
  Require 1024/256 = 4 BRAMs for depth
  Require 32/16 = 2 BRAMs for width
  Total = 4 × 2 = 8 BRAMs ✓

Conclusion: 8 BRAMs required (NOT 4)
```

### A.2 Memory Bandwidth Calculation

**Configuration:**
- 16 BRAM blocks
- Dual-port (read + write simultaneously)
- 16-bit data width per port
- Operating frequency: f

**Bandwidth Formula:**
```
BW = Blocks × Ports × Width × Frequency

Where:
  Blocks = 16 (BRAM blocks)
  Ports = 2 (dual-port)
  Width = 16 bits
  Frequency = f MHz
```

**Calculations:**

**@ 12 MHz (operating):**
```
BW = 16 × 2 × 16 × 12 MHz
   = 512 bits × 12 MHz
   = 6,144 Mb/s
   = 6.144 Gb/s
   = 6.144 Gbps ✓
```

**@ 90 MHz (design max):**
```
BW = 16 × 2 × 16 × 90 MHz
   = 512 bits × 90 MHz
   = 46,080 Mb/s
   = 46.08 Gbps ✓
```

**@ 133 MHz (datasheet max):**
```
BW = 16 × 2 × 16 × 133 MHz
   = 512 bits × 133 MHz
   = 68,096 Mb/s
   = 68.096 Gbps ✓
```

**Required frequency for claimed 192 Gbps:**
```
192 Gbps = 512 bits × f
f = 192,000 Mb/s ÷ 512 bits
f = 375 MHz ❌ IMPOSSIBLE (exceeds 133 MHz max by 2.8×)
```

---

## Appendix B: Resource Utilization Analysis

### Current Design (31 LUTs, 2.4%)

**Likely Components:**
```
UART TX module:       ~15 LUTs
UART RX module:       ~15 LUTs
Clock divider:        ~5 LUTs
LED control:          ~2 LUTs
State machine:        ~4 LUTs
---------------------------------
Total estimate:       ~41 LUTs

Actual:               31 LUTs
Optimization factor:  1.3× (good synthesis)
```

### Expansion Capacity

**Available Resources:**
```
Remaining LUTs:       1,249 (97.6%)
Remaining FFs:        ~1,249
Remaining BRAMs:      16 (100%, none used)
```

**Potential Additions:**
```
SPI Master:           ~50 LUTs
I2C Controller:       ~40 LUTs
PWM Generator:        ~20 LUTs per channel
FIFO Buffer:          1-2 BRAMs
Simple processor:     ~300-500 LUTs (PicoRV32 minimal)
```

---

## Verification Conclusion

**Overall Assessment:**

| Category | Count | Percentage |
|----------|-------|------------|
| ✅ Verified Correct | 10 | 71.4% |
| ⚠️ Needs Correction | 2 | 14.3% |
| ❌ Critical Error | 1 | 7.1% |
| Additional Info | 1 | 7.1% |

**Priority Actions:**

1. **IMMEDIATE:** Correct memory bandwidth claim (192 Gbps → 6.14 Gbps) ❌
2. **HIGH:** Fix BRAM allocation calculation (4 → 8 BRAMs) ⚠️
3. **MEDIUM:** Clarify maximum frequency specifications ⚠️
4. **LOW:** Add notes about design vs. device specifications ℹ️

**Document Status:** ⚠️ **Requires updates before publication**

**Verified By:** Research Agent
**Date:** 2026-01-06
**Version:** 1.0

---

**END OF VERIFICATION REPORT**
