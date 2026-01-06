#!/usr/bin/env python3
"""
Causal Lattice FSM Visualization Tool
======================================

Generates visual diagrams of:
1. 2D lattice state space (4x8 grid)
2. Transition graphs for each trigger type
3. Causal history timeline from simulation
4. State traversal patterns

Author: AI Code Agent
Date: 2026-01-06

Requirements: matplotlib, numpy
Install: pip install matplotlib numpy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Tuple, Dict

# ============================================================================
# Configuration
# ============================================================================

ROWS = 4
COLS = 8
STATES = ROWS * COLS  # 32 total states

TRIGGER_NAMES = [
    "T0: Temporal (256)",
    "T1: Temporal (1024)",
    "T2: Pattern (10101)",
    "T3: Pattern (01010)",
    "T4: External[0]",
    "T5: External[1]",
    "T6: Combinatorial",
    "T7: History-based"
]

TRIGGER_COLORS = [
    '#FF6B6B',  # Red - Temporal short
    '#4ECDC4',  # Teal - Temporal long
    '#45B7D1',  # Blue - Pattern 1
    '#96CEB4',  # Green - Pattern 2
    '#FFEAA7',  # Yellow - External 0
    '#DFE6E9',  # Gray - External 1
    '#A29BFE',  # Purple - Combinatorial
    '#FD79A8'   # Pink - History
]

# ============================================================================
# Helper Functions
# ============================================================================

def state_to_coords(state: int) -> Tuple[int, int]:
    """Convert 5-bit state to (row, col) coordinates."""
    row = (state >> 3) & 0x3  # bits [4:3]
    col = state & 0x7          # bits [2:0]
    return (row, col)

def coords_to_state(row: int, col: int) -> int:
    """Convert (row, col) to 5-bit state."""
    return (row << 3) | col

def compute_next_state(current_state: int, trigger: int) -> int:
    """Compute next state based on current state and trigger."""
    row, col = state_to_coords(current_state)

    if trigger == 0 or trigger == 2 or trigger == 4:
        # Horizontal movement (right)
        next_row = row
        next_col = (col + 1) % COLS
    elif trigger == 1 or trigger == 3 or trigger == 5:
        # Vertical movement (up)
        next_row = (row + 1) % ROWS
        next_col = col
    elif trigger == 6:
        # Diagonal movement (right-up)
        next_row = (row + 1) % ROWS
        next_col = (col + 1) % COLS
    elif trigger == 7:
        # Non-linear jump: {col[1:0], row[1:0], col[2]}
        next_row = col & 0x3
        next_col = ((row & 0x3) << 1) | ((col >> 2) & 0x1)
    else:
        next_row = row
        next_col = col

    return coords_to_state(next_row, next_col)

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_lattice_grid():
    """Plot the basic 4x8 lattice grid with state labels."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw grid
    for row in range(ROWS):
        for col in range(COLS):
            state = coords_to_state(row, col)

            # Draw cell
            rect = mpatches.Rectangle((col, row), 1, 1,
                                     linewidth=2,
                                     edgecolor='black',
                                     facecolor='lightblue',
                                     alpha=0.3)
            ax.add_patch(rect)

            # Add state number
            ax.text(col + 0.5, row + 0.5, f'{state}',
                   ha='center', va='center',
                   fontsize=12, fontweight='bold')

            # Add binary representation
            binary = f'{state:05b}'
            ax.text(col + 0.5, row + 0.2, binary,
                   ha='center', va='center',
                   fontsize=7, color='gray')

    # Formatting
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect('equal')
    ax.set_xlabel('Column (bits [2:0])', fontsize=12)
    ax.set_ylabel('Row (bits [4:3])', fontsize=12)
    ax.set_title('Causal Lattice FSM - State Space (4×8 Grid = 32 States)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add column labels
    ax.set_xticks(np.arange(0.5, COLS, 1))
    ax.set_xticklabels(range(COLS))

    # Add row labels
    ax.set_yticks(np.arange(0.5, ROWS, 1))
    ax.set_yticklabels(range(ROWS))

    plt.tight_layout()
    return fig

def plot_trigger_transitions(trigger_id: int):
    """Plot state transitions for a specific trigger type."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw grid background
    for row in range(ROWS):
        for col in range(COLS):
            state = coords_to_state(row, col)
            rect = mpatches.Rectangle((col, row), 1, 1,
                                     linewidth=1,
                                     edgecolor='gray',
                                     facecolor='white',
                                     alpha=0.5)
            ax.add_patch(rect)

            # Add state number
            ax.text(col + 0.5, row + 0.5, f'{state}',
                   ha='center', va='center',
                   fontsize=10, color='lightgray')

    # Draw transitions
    for row in range(ROWS):
        for col in range(COLS):
            current_state = coords_to_state(row, col)
            next_state = compute_next_state(current_state, trigger_id)
            next_row, next_col = state_to_coords(next_state)

            # Calculate arrow positions
            start_x = col + 0.5
            start_y = row + 0.5
            end_x = next_col + 0.5
            end_y = next_row + 0.5

            # Handle wraparound visualization
            dx = end_x - start_x
            dy = end_y - start_y

            # Draw arrow
            if abs(dx) < COLS/2 and abs(dy) < ROWS/2:
                # Normal arrow
                ax.arrow(start_x, start_y, dx*0.7, dy*0.7,
                        head_width=0.15, head_length=0.1,
                        fc=TRIGGER_COLORS[trigger_id],
                        ec=TRIGGER_COLORS[trigger_id],
                        alpha=0.7, linewidth=2)
            else:
                # Wraparound - draw dashed line
                ax.plot([start_x, end_x], [start_y, end_y],
                       linestyle='--', color=TRIGGER_COLORS[trigger_id],
                       alpha=0.5, linewidth=1.5)

    # Formatting
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect('equal')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(f'{TRIGGER_NAMES[trigger_id]} - Transition Pattern',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_all_triggers():
    """Create a multi-panel plot showing all trigger types."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for trigger_id in range(8):
        ax = axes[trigger_id]

        # Draw grid
        for row in range(ROWS):
            for col in range(COLS):
                state = coords_to_state(row, col)
                rect = mpatches.Rectangle((col, row), 1, 1,
                                         linewidth=0.5,
                                         edgecolor='gray',
                                         facecolor='white')
                ax.add_patch(rect)

        # Draw transitions
        for row in range(ROWS):
            for col in range(COLS):
                current_state = coords_to_state(row, col)
                next_state = compute_next_state(current_state, trigger_id)
                next_row, next_col = state_to_coords(next_state)

                start_x = col + 0.5
                start_y = row + 0.5
                end_x = next_col + 0.5
                end_y = next_row + 0.5

                dx = end_x - start_x
                dy = end_y - start_y

                if abs(dx) < COLS/2 and abs(dy) < ROWS/2:
                    ax.arrow(start_x, start_y, dx*0.6, dy*0.6,
                            head_width=0.12, head_length=0.08,
                            fc=TRIGGER_COLORS[trigger_id],
                            ec=TRIGGER_COLORS[trigger_id],
                            alpha=0.6, linewidth=1.5)

        ax.set_xlim(0, COLS)
        ax.set_ylim(0, ROWS)
        ax.set_aspect('equal')
        ax.set_title(TRIGGER_NAMES[trigger_id], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('All Trigger Types - Transition Patterns',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_example_trajectory():
    """Plot an example causal trajectory through the state space."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Example trajectory: Start at [0,0], apply various triggers
    trajectory = [
        (0, 4),   # State[0,0] + Trigger 4 (ext[0]) → State[0,1]
        (1, 5),   # State[0,1] + Trigger 5 (ext[1]) → State[1,1]
        (2, 4),   # State[1,1] + Trigger 4 → State[1,2]
        (3, 4),   # State[1,2] + Trigger 4 → State[1,3]
        (4, 1),   # State[1,3] + Trigger 1 (temporal) → State[2,3]
        (5, 6),   # State[2,3] + Trigger 6 (diagonal) → State[3,4]
        (6, 7),   # State[3,4] + Trigger 7 (non-linear) → ???
    ]

    # Draw grid
    for row in range(ROWS):
        for col in range(COLS):
            state = coords_to_state(row, col)
            rect = mpatches.Rectangle((col, row), 1, 1,
                                     linewidth=1,
                                     edgecolor='gray',
                                     facecolor='white',
                                     alpha=0.3)
            ax.add_patch(rect)
            ax.text(col + 0.5, row + 0.5, f'{state}',
                   ha='center', va='center',
                   fontsize=9, color='lightgray')

    # Trace trajectory
    current_state = 0  # Start at State[0,0]
    path_x = [0.5]
    path_y = [0.5]

    for step, (idx, trigger) in enumerate(trajectory):
        next_state = compute_next_state(current_state, trigger)
        next_row, next_col = state_to_coords(next_state)

        # Highlight current state
        curr_row, curr_col = state_to_coords(current_state)
        rect = mpatches.Rectangle((curr_col, curr_row), 1, 1,
                                 linewidth=2,
                                 edgecolor='red',
                                 facecolor='yellow',
                                 alpha=0.5)
        ax.add_patch(rect)

        # Draw arrow
        start_x = curr_col + 0.5
        start_y = curr_row + 0.5
        end_x = next_col + 0.5
        end_y = next_row + 0.5

        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2.5,
                                  color=TRIGGER_COLORS[trigger],
                                  alpha=0.8))

        # Add step number
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        ax.text(mid_x, mid_y, f'{step+1}',
               bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'),
               ha='center', va='center', fontsize=10, fontweight='bold')

        path_x.append(end_x)
        path_y.append(end_y)
        current_state = next_state

    # Highlight final state
    final_row, final_col = state_to_coords(current_state)
    rect = mpatches.Rectangle((final_col, final_row), 1, 1,
                             linewidth=3,
                             edgecolor='green',
                             facecolor='lightgreen',
                             alpha=0.6)
    ax.add_patch(rect)

    # Formatting
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect('equal')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title('Example Causal Trajectory Through Lattice',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [
        mpatches.Patch(color='yellow', label='Visited State'),
        mpatches.Patch(color='lightgreen', label='Final State'),
    ]
    for i, (idx, trigger) in enumerate(trajectory):
        legend_elements.append(
            mpatches.Patch(color=TRIGGER_COLORS[trigger],
                          label=f'Step {i+1}: {TRIGGER_NAMES[trigger]}')
        )
    ax.legend(handles=legend_elements, loc='center left',
             bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.tight_layout()
    return fig

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Generate all visualizations."""
    print("Generating Causal Lattice FSM Visualizations...")

    # 1. Basic lattice grid
    print("  1. Lattice grid...")
    fig1 = plot_lattice_grid()
    fig1.savefig('causal_lattice_grid.png', dpi=300, bbox_inches='tight')
    print("     Saved: causal_lattice_grid.png")

    # 2. Individual trigger patterns
    print("  2. Individual trigger patterns...")
    for trigger_id in range(8):
        fig = plot_trigger_transitions(trigger_id)
        filename = f'causal_lattice_trigger{trigger_id}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"     Saved: {filename}")
        plt.close(fig)

    # 3. All triggers in one view
    print("  3. All triggers overview...")
    fig3 = plot_all_triggers()
    fig3.savefig('causal_lattice_all_triggers.png', dpi=300, bbox_inches='tight')
    print("     Saved: causal_lattice_all_triggers.png")

    # 4. Example trajectory
    print("  4. Example trajectory...")
    fig4 = plot_example_trajectory()
    fig4.savefig('causal_lattice_trajectory.png', dpi=300, bbox_inches='tight')
    print("     Saved: causal_lattice_trajectory.png")

    print("\nAll visualizations generated successfully!")
    print("\nFiles created:")
    print("  - causal_lattice_grid.png")
    print("  - causal_lattice_trigger[0-7].png (8 files)")
    print("  - causal_lattice_all_triggers.png")
    print("  - causal_lattice_trajectory.png")

    # Show all plots
    plt.show()

if __name__ == '__main__':
    main()
