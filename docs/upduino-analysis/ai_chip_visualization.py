"""
Advanced Manim Visualization for AI-on-Chip Architecture on UPduino
=====================================================================

A professional-grade visualization suite for technical presentations on FPGA-based AI acceleration.

Usage:
    manim -pql ai_chip_visualization.py AIChipOverview
    manim -pqh ai_chip_visualization.py FPGAResourceMapping
    manim -pqh ai_chip_visualization.py DataflowAnimation
    manim -pqh ai_chip_visualization.py MemoryAsInference
    manim -pqh ai_chip_visualization.py PerformanceComparison
    manim -pqh ai_chip_visualization.py TestingWorkflow
    manim -pqh ai_chip_visualization.py FullPresentation

Requirements:
    pip install manim numpy
"""

from manim import *
import numpy as np

# Color scheme for technical visualization
AI_COLORS = {
    "input": BLUE,
    "weight": ORANGE,
    "compute": GREEN,
    "output": PURPLE,
    "memory": YELLOW,
    "control": RED,
    "data_flow": TEAL,
    "highlight": GOLD,
}


class AIChipOverview(Scene):
    """Scene 1: AI Hardware Overview and Paradigm Shift"""

    def construct(self):
        # Title
        title = Text("AI-on-Chip Architecture Evolution", font_size=48, gradient=(BLUE, PURPLE))
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Section 1: Traditional von Neumann Architecture
        self.show_von_neumann_bottleneck()
        self.wait(2)
        self.clear()

        # Section 2: Memory-Centric Computing
        self.show_memory_centric_paradigm()
        self.wait(2)
        self.clear()

        # Section 3: UPduino as Prototyping Platform
        self.show_upduino_platform()
        self.wait(2)

    def show_von_neumann_bottleneck(self):
        subtitle = Text("Traditional von Neumann Bottleneck", font_size=36)
        subtitle.to_edge(UP, buff=0.8)
        self.play(Write(subtitle))

        # CPU and Memory boxes
        cpu = Rectangle(width=3, height=2, color=BLUE, fill_opacity=0.3)
        cpu_label = Text("CPU\n(Processing)", font_size=24).move_to(cpu)
        cpu_group = VGroup(cpu, cpu_label).shift(LEFT * 4)

        memory = Rectangle(width=3, height=2, color=ORANGE, fill_opacity=0.3)
        memory_label = Text("Memory\n(Storage)", font_size=24).move_to(memory)
        memory_group = VGroup(memory, memory_label).shift(RIGHT * 4)

        # Bottleneck arrows
        arrow_right = Arrow(cpu.get_right(), memory.get_left(), color=RED, stroke_width=8)
        arrow_left = Arrow(memory.get_left(), cpu.get_right(), color=RED, stroke_width=8)
        arrow_left.shift(DOWN * 0.3)
        arrow_right.shift(UP * 0.3)

        bottleneck_label = Text("Data Movement\nBottleneck", font_size=20, color=RED)
        bottleneck_label.next_to(arrow_right, UP)

        self.play(
            Create(cpu_group),
            Create(memory_group)
        )
        self.play(
            GrowArrow(arrow_right),
            GrowArrow(arrow_left),
            Write(bottleneck_label)
        )

        # Performance equation
        equation = MathTex(
            r"\text{Time} = \underbrace{\text{Compute Time}}_{\text{Fast}} + \underbrace{\text{Memory Access Time}}_{\text{Slow}}",
            font_size=32
        )
        equation.to_edge(DOWN)
        self.play(Write(equation))

        # Highlight the problem
        problem = Text("80% of AI workload time spent on memory access!",
                      font_size=28, color=RED)
        problem.next_to(equation, UP, buff=0.5)
        self.play(Write(problem))

    def show_memory_centric_paradigm(self):
        subtitle = Text("Memory-Centric Computing Paradigm", font_size=36)
        subtitle.to_edge(UP, buff=0.8)
        self.play(Write(subtitle))

        # Unified compute-memory block
        unified = RoundedRectangle(width=6, height=4, corner_radius=0.5,
                                  color=GREEN, fill_opacity=0.3)

        # Internal components
        compute_region = Rectangle(width=2.5, height=1.5, color=BLUE, fill_opacity=0.5)
        compute_region.move_to(unified.get_center() + LEFT * 1.5 + UP * 0.5)
        compute_label = Text("Compute", font_size=20).move_to(compute_region)

        memory_region = Rectangle(width=2.5, height=1.5, color=ORANGE, fill_opacity=0.5)
        memory_region.move_to(unified.get_center() + RIGHT * 1.5 + UP * 0.5)
        memory_label = Text("Memory", font_size=20).move_to(memory_region)

        # Short interconnects
        interconnects = VGroup(*[
            Line(compute_region.get_right() + UP * i * 0.3,
                 memory_region.get_left() + UP * i * 0.3,
                 color=YELLOW, stroke_width=4)
            for i in np.linspace(-1, 1, 5)
        ])

        title_label = Text("In-Memory\nComputing", font_size=28, color=GREEN)
        title_label.next_to(unified, UP, buff=0.3)

        self.play(Create(unified), Write(title_label))
        self.play(
            Create(VGroup(compute_region, compute_label)),
            Create(VGroup(memory_region, memory_label))
        )
        self.play(Create(interconnects))

        # Benefits
        benefits = VGroup(
            Text("✓ Minimize data movement", font_size=24, color=GREEN),
            Text("✓ Maximize parallelism", font_size=24, color=GREEN),
            Text("✓ Energy efficiency", font_size=24, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        benefits.to_edge(DOWN)

        self.play(Write(benefits))

        # Performance improvement
        speedup = MathTex(
            r"\text{Speedup} \approx 10\text{-}100\times",
            font_size=36,
            color=GOLD
        )
        speedup.next_to(benefits, UP, buff=0.5)
        self.play(Write(speedup))

    def show_upduino_platform(self):
        subtitle = Text("UPduino v3.1: FPGA AI Prototyping", font_size=36)
        subtitle.to_edge(UP, buff=0.8)
        self.play(Write(subtitle))

        # UPduino board representation
        board = RoundedRectangle(width=8, height=5, corner_radius=0.3,
                                color=BLUE_D, fill_opacity=0.2, stroke_width=3)

        # FPGA chip
        fpga = Rectangle(width=3, height=3, color=PURPLE, fill_opacity=0.4)
        fpga_label = Text("iCE40 UP5K\nFPGA", font_size=28, weight=BOLD)
        fpga_label.move_to(fpga)
        fpga_group = VGroup(fpga, fpga_label)

        # Specifications
        specs = VGroup(
            Text("• 5,280 LUTs", font_size=20, color=BLUE),
            Text("• 120 Kb SPRAM", font_size=20, color=ORANGE),
            Text("• 8 DSP blocks", font_size=20, color=GREEN),
            Text("• 39 I/O pins", font_size=20, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        specs.next_to(fpga, RIGHT, buff=0.8)

        # Memory blocks
        spram = Rectangle(width=1.5, height=0.8, color=ORANGE, fill_opacity=0.5)
        spram_label = Text("SPRAM", font_size=16).move_to(spram)
        spram_group = VGroup(spram, spram_label).next_to(fpga, DOWN, buff=0.3)

        # DSP blocks
        dsp_blocks = VGroup(*[
            Rectangle(width=0.5, height=0.5, color=GREEN, fill_opacity=0.5)
            for _ in range(4)
        ]).arrange(RIGHT, buff=0.1)
        dsp_label = Text("DSP", font_size=12).next_to(dsp_blocks, DOWN, buff=0.1)
        dsp_group = VGroup(dsp_blocks, dsp_label).next_to(fpga, LEFT, buff=0.5)

        self.play(Create(board))
        self.play(Create(fpga_group))
        self.play(Write(specs))
        self.play(
            Create(spram_group),
            Create(dsp_group)
        )

        # Use case
        use_case = Text(
            "Ideal for prototyping AI accelerators and in-memory computing",
            font_size=24,
            color=GOLD
        )
        use_case.to_edge(DOWN)
        self.play(Write(use_case))


class FPGAResourceMapping(ThreeDScene):
    """Scene 2: FPGA Resource Mapping to Neural Network Components"""

    def construct(self):
        # Setup 3D camera
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        title = Text("FPGA Resource → Neural Network Mapping", font_size=42)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait()

        # Section 1: LUTs to Logic Gates to NN Layers
        self.show_lut_mapping()
        self.wait(2)

        # Section 2: SPRAM to Weight Storage
        self.show_spram_mapping()
        self.wait(2)

        # Section 3: DSP to Systolic Array
        self.show_dsp_mapping()
        self.wait(2)

    def show_lut_mapping(self):
        # Create LUT representation
        lut = Cube(side_length=1, fill_color=BLUE, fill_opacity=0.5)
        lut_label = Text("LUT", font_size=24).next_to(lut, DOWN)

        self.play(Create(lut), Write(lut_label))
        self.wait()

        # Transform to logic gates
        and_gate = Polygon(
            LEFT, RIGHT, RIGHT + DOWN, LEFT + DOWN,
            color=GREEN, fill_opacity=0.5
        ).scale(0.5).shift(RIGHT * 2)
        or_gate = Circle(radius=0.5, color=GREEN, fill_opacity=0.5).shift(RIGHT * 2 + UP)
        gates_label = Text("Logic Gates", font_size=20).next_to(and_gate, DOWN)

        self.play(
            Transform(lut.copy(), VGroup(and_gate, or_gate)),
            Write(gates_label)
        )
        self.wait()

        # Transform to NN layer
        neurons = VGroup(*[
            Circle(radius=0.3, color=PURPLE, fill_opacity=0.5)
            for _ in range(4)
        ]).arrange(DOWN, buff=0.3).shift(RIGHT * 4)

        connections = VGroup(*[
            Line(and_gate.get_right(), neuron.get_left(), color=YELLOW, stroke_width=2)
            for neuron in neurons
        ])

        layer_label = Text("NN Layer", font_size=20).next_to(neurons, RIGHT)

        self.play(
            Create(neurons),
            Create(connections),
            Write(layer_label)
        )

        # Equation
        equation = MathTex(
            r"\text{5,280 LUTs} \rightarrow \text{Configurable Logic} \rightarrow \text{NN Layers}",
            font_size=28
        ).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(equation)
        self.play(Write(equation))

    def show_spram_mapping(self):
        self.clear()

        # SPRAM blocks
        spram_blocks = VGroup(*[
            Cube(side_length=0.5, fill_color=ORANGE, fill_opacity=0.5)
            for _ in range(8)
        ]).arrange_in_grid(rows=2, cols=4, buff=0.2)

        spram_label = Text("120 Kb SPRAM", font_size=24).next_to(spram_blocks, DOWN)

        self.play(Create(spram_blocks), Write(spram_label))
        self.wait()

        # Weight storage representation
        weight_matrix = Matrix(
            [[r"w_{11}", r"w_{12}", r"w_{13}"],
             [r"w_{21}", r"w_{22}", r"w_{23}"],
             [r"w_{31}", r"w_{32}", r"w_{33}"]],
            h_buff=1.5
        ).scale(0.7).shift(RIGHT * 3)

        weight_label = Text("Weight Storage", font_size=20).next_to(weight_matrix, DOWN)

        self.play(
            Create(weight_matrix),
            Write(weight_label)
        )

        # In-memory compute concept
        compute_arrow = Arrow(
            spram_blocks.get_right(), weight_matrix.get_left(),
            color=GREEN, stroke_width=6
        )
        compute_label = Text("In-Memory\nCompute", font_size=18, color=GREEN)
        compute_label.next_to(compute_arrow, UP)

        self.play(
            GrowArrow(compute_arrow),
            Write(compute_label)
        )

        # Memory calculation
        calc = MathTex(
            r"\text{120 Kb} = 15,000 \text{ weights (8-bit)}",
            font_size=28
        ).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(calc)
        self.play(Write(calc))

    def show_dsp_mapping(self):
        self.clear()

        # DSP blocks
        dsp_blocks = VGroup(*[
            Prism(dimensions=[0.5, 0.5, 0.5], fill_color=GREEN, fill_opacity=0.6)
            for _ in range(8)
        ]).arrange_in_grid(rows=2, cols=4, buff=0.3)

        dsp_label = Text("8 DSP Blocks", font_size=24).next_to(dsp_blocks, DOWN)

        self.play(Create(dsp_blocks), Write(dsp_label))
        self.wait()

        # MAC units
        mac_equation = MathTex(
            r"MAC: a \times b + c",
            font_size=32
        ).shift(RIGHT * 3 + UP)

        self.add_fixed_in_frame_mobjects(mac_equation)
        self.play(Write(mac_equation))

        # Systolic array representation
        pe_array = VGroup(*[
            Square(side_length=0.4, color=PURPLE, fill_opacity=0.5)
            for _ in range(16)
        ]).arrange_in_grid(rows=4, cols=4, buff=0.15).shift(RIGHT * 3 + DOWN)

        pe_label = Text("Systolic Array PEs", font_size=20).next_to(pe_array, DOWN)

        self.add_fixed_in_frame_mobjects(pe_label)
        self.play(Create(pe_array), Write(pe_label))

        # Data flow arrows
        input_arrows = VGroup(*[
            Arrow(pe_array[i].get_left() + LEFT * 0.5, pe_array[i].get_left(),
                  color=BLUE, stroke_width=3)
            for i in [0, 4, 8, 12]
        ])

        weight_arrows = VGroup(*[
            Arrow(pe_array[i].get_top() + UP * 0.5, pe_array[i].get_top(),
                  color=ORANGE, stroke_width=3)
            for i in range(4)
        ])

        self.play(
            Create(input_arrows),
            Create(weight_arrows)
        )

        # Performance metric
        perf = MathTex(
            r"\text{8 DSP} \times 100\text{ MHz} = 800\text{ MMAC/s}",
            font_size=28
        ).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(perf)
        self.play(Write(perf))


class DataflowAnimation(Scene):
    """Scene 3: Animated Dataflow Through Neural Network"""

    def construct(self):
        title = Text("Neural Network Dataflow on FPGA", font_size=42)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Build systolic array
        self.build_systolic_array()
        self.wait()

        # Animate input feature map
        self.animate_input_flow()
        self.wait()

        # Animate weight stationary operation
        self.animate_weight_stationary()
        self.wait()

        # Animate partial sum accumulation
        self.animate_partial_sums()
        self.wait()

        # Animate activation function
        self.animate_activation()
        self.wait()

    def build_systolic_array(self):
        # Create 4x4 processing element array
        self.pe_size = 0.8
        self.pe_array = VGroup(*[
            Square(side_length=self.pe_size, color=BLUE, fill_opacity=0.3)
            for _ in range(16)
        ]).arrange_in_grid(rows=4, cols=4, buff=0.2)

        # Add MAC labels
        self.pe_labels = VGroup(*[
            Text("MAC", font_size=16).move_to(pe)
            for pe in self.pe_array
        ])

        # Add weight storage in each PE
        self.weights = VGroup(*[
            MathTex(r"w_{" + str(i) + "}", font_size=14, color=ORANGE)
            .next_to(pe, UP, buff=0.05)
            for i, pe in enumerate(self.pe_array)
        ])

        self.play(
            Create(self.pe_array),
            Write(self.pe_labels)
        )
        self.play(Write(self.weights))

        # Add labels
        input_label = Text("Input", font_size=20, color=BLUE).next_to(
            self.pe_array, LEFT, buff=0.5
        )
        output_label = Text("Output", font_size=20, color=PURPLE).next_to(
            self.pe_array, RIGHT, buff=0.5
        )

        self.play(Write(input_label), Write(output_label))

    def animate_input_flow(self):
        # Create input feature map data
        input_data = VGroup(*[
            Circle(radius=0.15, color=AI_COLORS["input"], fill_opacity=0.8)
            for _ in range(4)
        ])

        # Position inputs to the left of first column
        for i, data in enumerate(input_data):
            data.move_to(self.pe_array[i * 4].get_left() + LEFT * 1.5)
            data_label = MathTex(r"x_{" + str(i) + "}", font_size=12).next_to(data, LEFT, buff=0.1)
            data.add(data_label)

        self.play(Create(input_data))

        # Animate flow through first column
        for i, data in enumerate(input_data):
            path = Line(
                data.get_center(),
                self.pe_array[i * 4].get_left(),
                color=AI_COLORS["data_flow"]
            )

            self.play(
                MoveAlongPath(data, path),
                run_time=0.5
            )

            # Show computation happening
            self.pe_array[i * 4].set_fill(GREEN, opacity=0.6)
            self.wait(0.2)
            self.pe_array[i * 4].set_fill(BLUE, opacity=0.3)

    def animate_weight_stationary(self):
        # Highlight that weights stay in place
        weight_box = SurroundingRectangle(
            self.weights,
            color=GOLD,
            buff=0.1,
            stroke_width=3
        )

        stationary_text = Text(
            "Weight Stationary: Weights remain in PEs",
            font_size=24,
            color=GOLD
        ).to_edge(DOWN)

        self.play(
            Create(weight_box),
            Write(stationary_text)
        )
        self.wait(2)
        self.play(
            FadeOut(weight_box),
            FadeOut(stationary_text)
        )

    def animate_partial_sums(self):
        # Show partial sum accumulation
        partial_sum_label = Text(
            "Partial Sum Accumulation",
            font_size=28,
            color=GREEN
        ).to_edge(DOWN)

        self.play(Write(partial_sum_label))

        # Animate accumulation through rows
        for row in range(4):
            accumulator = Circle(radius=0.2, color=GREEN, fill_opacity=0.7)
            accumulator.move_to(self.pe_array[row * 4].get_center())

            self.play(Create(accumulator))

            for col in range(1, 4):
                pe_index = row * 4 + col
                path = Line(
                    self.pe_array[pe_index - 1].get_right(),
                    self.pe_array[pe_index].get_left()
                )

                # Grow accumulator to show accumulation
                self.play(
                    MoveAlongPath(accumulator, path),
                    accumulator.animate.scale(1.2),
                    run_time=0.3
                )
                self.pe_array[pe_index].set_fill(GREEN, opacity=0.6)
                self.wait(0.1)
                self.pe_array[pe_index].set_fill(BLUE, opacity=0.3)

            # Move to output
            final_path = Line(
                self.pe_array[row * 4 + 3].get_right(),
                self.pe_array[row * 4 + 3].get_right() + RIGHT * 1.5
            )
            self.play(
                MoveAlongPath(accumulator, final_path),
                FadeOut(accumulator),
                run_time=0.5
            )

        self.play(FadeOut(partial_sum_label))

    def animate_activation(self):
        # Create activation function visualization
        activation_label = Text(
            "Activation Function: ReLU(x) = max(0, x)",
            font_size=28,
            color=PURPLE
        ).to_edge(DOWN)

        self.play(Write(activation_label))

        # Create ReLU function graph
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-0.5, 2, 1],
            x_length=4,
            y_length=3,
            tips=False
        ).scale(0.6).to_edge(RIGHT)

        relu_graph = axes.plot(
            lambda x: max(0, x),
            color=PURPLE,
            x_range=[-2, 2]
        )

        graph_label = Text("ReLU", font_size=20, color=PURPLE).next_to(axes, UP)

        self.play(
            Create(axes),
            Create(relu_graph),
            Write(graph_label)
        )

        # Show transformation
        input_val = MathTex(r"x = -0.5", font_size=24).next_to(axes, LEFT)
        output_val = MathTex(r"y = 0", font_size=24, color=GREEN).next_to(input_val, DOWN)

        self.play(Write(input_val))
        self.wait()
        self.play(Write(output_val))

        self.wait(2)


class MemoryAsInference(ThreeDScene):
    """Scene 4: Memory-as-Inference and Analog Computing Concepts"""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)

        title = Text("In-Memory Computing: Memory as Inference", font_size=40)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait()

        # Show crossbar array
        self.show_crossbar_array()
        self.wait(2)

        # Show analog computation
        self.show_analog_computation()
        self.wait(2)

        # Show FPGA digital emulation
        self.show_fpga_emulation()
        self.wait(2)

    def show_crossbar_array(self):
        # Create crossbar array structure
        rows = 4
        cols = 4
        spacing = 1.5

        # Horizontal wires (word lines)
        h_wires = VGroup(*[
            Line(LEFT * 3, RIGHT * 3, color=BLUE, stroke_width=6)
            .shift(UP * (rows/2 - i) * spacing)
            for i in range(rows)
        ])

        # Vertical wires (bit lines)
        v_wires = VGroup(*[
            Line(UP * 3, DOWN * 3, color=ORANGE, stroke_width=6)
            .shift(LEFT * (cols/2 - i - 0.5) * spacing)
            for i in range(cols)
        ])

        self.play(Create(h_wires), Create(v_wires))

        # Add memristor/resistor elements at intersections
        memristors = VGroup()
        for i in range(rows):
            for j in range(cols):
                pos = np.array([
                    (j - cols/2 + 0.5) * spacing,
                    (rows/2 - i) * spacing,
                    0
                ])

                # Memristor represented as small sphere
                mem = Sphere(radius=0.15, color=GREEN, resolution=(8, 8))
                mem.move_to(pos)
                memristors.add(mem)

        self.play(Create(memristors))

        # Labels
        wl_label = Text("Word Lines (Inputs)", font_size=20, color=BLUE)
        wl_label.next_to(h_wires, LEFT, buff=0.3)

        bl_label = Text("Bit Lines (Outputs)", font_size=20, color=ORANGE)
        bl_label.next_to(v_wires, DOWN, buff=0.3)

        mem_label = Text("Memristors\n(Weights)", font_size=18, color=GREEN)
        mem_label.to_corner(UR)

        self.add_fixed_in_frame_mobjects(wl_label, bl_label, mem_label)
        self.play(Write(wl_label), Write(bl_label), Write(mem_label))

    def show_analog_computation(self):
        self.clear()

        subtitle = Text("Analog Computation via Ohm's Law", font_size=32)
        subtitle.to_edge(UP, buff=1.5)
        self.add_fixed_in_frame_mobjects(subtitle)
        self.play(Write(subtitle))

        # Show Ohm's law
        ohms_law = MathTex(
            r"I = V \times G = V \times \frac{1}{R}",
            font_size=40
        ).shift(UP * 2)

        self.add_fixed_in_frame_mobjects(ohms_law)
        self.play(Write(ohms_law))

        # Show how conductance encodes weights
        weight_encoding = MathTex(
            r"G \propto w \quad \Rightarrow \quad I = V \times w",
            font_size=40
        )

        self.add_fixed_in_frame_mobjects(weight_encoding)
        self.play(Write(weight_encoding))

        # Show MAC operation
        mac_operation = VGroup(
            MathTex(r"\text{Multiply: } I_i = V_i \times w_i", font_size=32),
            MathTex(r"\text{Accumulate: } I_{total} = \sum_i I_i", font_size=32),
            MathTex(r"\text{(Kirchhoff's Current Law)}", font_size=24, color=YELLOW)
        ).arrange(DOWN, buff=0.3).shift(DOWN * 1.5)

        self.add_fixed_in_frame_mobjects(mac_operation)
        self.play(Write(mac_operation))

        # Advantage box
        advantage = VGroup(
            Text("Single-cycle MAC operation!", font_size=28, color=GREEN),
            Text("Energy: ~0.1 pJ per operation", font_size=24, color=GREEN)
        ).arrange(DOWN, buff=0.2).to_edge(DOWN)

        self.add_fixed_in_frame_mobjects(advantage)
        self.play(Write(advantage))

    def show_fpga_emulation(self):
        self.clear()

        subtitle = Text("FPGA Digital Emulation of Analog Compute", font_size=32)
        subtitle.to_edge(UP, buff=1.5)
        self.add_fixed_in_frame_mobjects(subtitle)
        self.play(Write(subtitle))

        # Create comparison diagram
        analog_side = VGroup(
            Text("Analog (Ideal)", font_size=28, color=GOLD).shift(UP * 2),
            Text("• Continuous values", font_size=20).shift(UP * 1.2),
            Text("• Instant MAC", font_size=20).shift(UP * 0.6),
            Text("• Low energy", font_size=20),
            Text("• Noise sensitive", font_size=20, color=RED).shift(DOWN * 0.6)
        ).shift(LEFT * 3.5)

        digital_side = VGroup(
            Text("FPGA (Practical)", font_size=28, color=BLUE).shift(UP * 2),
            Text("• Quantized (8-bit)", font_size=20).shift(UP * 1.2),
            Text("• Pipelined MAC", font_size=20).shift(UP * 0.6),
            Text("• Programmable", font_size=20),
            Text("• Robust", font_size=20, color=GREEN).shift(DOWN * 0.6)
        ).shift(RIGHT * 3.5)

        separator = Line(UP * 3, DOWN * 3, color=WHITE, stroke_width=2)

        self.add_fixed_in_frame_mobjects(analog_side, digital_side, separator)
        self.play(
            Write(analog_side),
            Create(separator),
            Write(digital_side)
        )

        # Verilog code snippet
        code_title = Text("Verilog: Digital MAC Unit", font_size=24, color=YELLOW)
        code_title.to_edge(DOWN, buff=2)

        verilog_code = Code(
            code="""module mac_unit (
    input [7:0] a, b,
    input [15:0] c,
    output reg [15:0] out
);
    always @(*) begin
        out = a * b + c;
    end
endmodule""",
            language="verilog",
            font="Monospace",
            font_size=16,
            background="window",
            style="monokai"
        ).scale(0.5).next_to(code_title, DOWN, buff=0.2)

        self.add_fixed_in_frame_mobjects(code_title, verilog_code)
        self.play(Write(code_title), Create(verilog_code))


class PerformanceComparison(Scene):
    """Scene 5: Performance Comparison Charts"""

    def construct(self):
        title = Text("Performance Comparison: FPGA vs CPU vs GPU", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Throughput comparison
        self.show_throughput_chart()
        self.wait(3)

        self.clear()
        title.to_edge(UP)
        self.add(title)

        # Energy efficiency
        self.show_energy_chart()
        self.wait(3)

        self.clear()
        title.to_edge(UP)
        self.add(title)

        # Latency vs throughput tradeoffs
        self.show_tradeoffs()
        self.wait(3)

    def show_throughput_chart(self):
        # Bar chart data (operations per second)
        platforms = ["CPU\n(i7)", "GPU\n(RTX3060)", "FPGA\n(UPduino)"]
        throughputs = [100, 1000, 800]  # Relative values (GOPS)
        colors = [BLUE, GREEN, ORANGE]

        # Create axes
        axes = Axes(
            x_range=[0, len(platforms), 1],
            y_range=[0, 1200, 200],
            x_length=8,
            y_length=4,
            axis_config={"include_tip": False},
            y_axis_config={"label_direction": LEFT}
        ).shift(DOWN * 0.5)

        # Y-axis label
        y_label = Text("GOPS (Giga-Ops/Sec)", font_size=24).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.5)

        self.play(Create(axes), Write(y_label))

        # Create bars
        bars = VGroup()
        bar_labels = VGroup()

        for i, (platform, throughput, color) in enumerate(zip(platforms, throughputs, colors)):
            bar = Rectangle(
                width=0.8,
                height=throughput / 1200 * 4,
                fill_color=color,
                fill_opacity=0.7,
                stroke_width=2
            )
            bar.move_to(axes.c2p(i + 0.5, throughput / 2))

            value_label = Text(f"{throughput}", font_size=20, color=color)
            value_label.next_to(bar, UP, buff=0.1)

            platform_label = Text(platform, font_size=18)
            platform_label.next_to(bar, DOWN, buff=0.3)

            bars.add(bar)
            bar_labels.add(value_label, platform_label)

        self.play(
            *[GrowFromEdge(bar, DOWN) for bar in bars],
            run_time=1.5
        )
        self.play(Write(bar_labels))

        # Add note
        note = Text(
            "* UPduino achieves competitive throughput with 100x lower cost",
            font_size=20,
            color=YELLOW
        ).to_edge(DOWN)
        self.play(Write(note))

    def show_energy_chart(self):
        # Energy efficiency (GOPS/Watt)
        platforms = ["CPU\n(i7)", "GPU\n(RTX3060)", "FPGA\n(UPduino)"]
        efficiency = [5, 20, 150]  # GOPS/Watt
        colors = [BLUE, GREEN, ORANGE]

        # Create axes (log scale effect)
        axes = Axes(
            x_range=[0, len(platforms), 1],
            y_range=[0, 160, 40],
            x_length=8,
            y_length=4,
            axis_config={"include_tip": False},
            y_axis_config={"label_direction": LEFT}
        ).shift(DOWN * 0.5)

        # Y-axis label
        y_label = Text("GOPS/Watt", font_size=24).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.5)

        self.play(Create(axes), Write(y_label))

        # Create bars
        bars = VGroup()
        bar_labels = VGroup()

        for i, (platform, eff, color) in enumerate(zip(platforms, efficiency, colors)):
            bar = Rectangle(
                width=0.8,
                height=eff / 160 * 4,
                fill_color=color,
                fill_opacity=0.7,
                stroke_width=2
            )
            bar.move_to(axes.c2p(i + 0.5, eff / 2))

            value_label = Text(f"{eff}", font_size=20, color=color)
            value_label.next_to(bar, UP, buff=0.1)

            platform_label = Text(platform, font_size=18)
            platform_label.next_to(bar, DOWN, buff=0.3)

            bars.add(bar)
            bar_labels.add(value_label, platform_label)

        self.play(
            *[GrowFromEdge(bar, DOWN) for bar in bars],
            run_time=1.5
        )
        self.play(Write(bar_labels))

        # Highlight FPGA advantage
        highlight = SurroundingRectangle(bars[2], color=GOLD, buff=0.1, stroke_width=4)
        advantage_text = Text(
            "30x more energy efficient!",
            font_size=28,
            color=GOLD
        ).to_edge(DOWN)

        self.play(
            Create(highlight),
            Write(advantage_text)
        )

    def show_tradeoffs(self):
        # Latency vs Throughput scatter plot
        chart_title = Text("Latency vs Throughput Tradeoffs", font_size=32)
        chart_title.shift(UP * 2.5)
        self.play(Write(chart_title))

        # Create axes
        axes = Axes(
            x_range=[0, 1000, 200],
            y_range=[0, 100, 20],
            x_length=7,
            y_length=4,
            axis_config={"include_tip": True},
            tips=True
        )

        x_label = Text("Throughput (GOPS)", font_size=20).next_to(axes.x_axis, DOWN)
        y_label = Text("Latency (ms)", font_size=20).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Plot points
        cpu_point = Dot(axes.c2p(100, 50), color=BLUE, radius=0.15)
        cpu_label = Text("CPU", font_size=18, color=BLUE).next_to(cpu_point, UR, buff=0.1)

        gpu_point = Dot(axes.c2p(1000, 80), color=GREEN, radius=0.15)
        gpu_label = Text("GPU", font_size=18, color=GREEN).next_to(gpu_point, UR, buff=0.1)

        fpga_point = Dot(axes.c2p(800, 15), color=ORANGE, radius=0.15)
        fpga_label = Text("FPGA", font_size=18, color=ORANGE).next_to(fpga_point, DR, buff=0.1)

        self.play(
            Create(cpu_point), Write(cpu_label),
            Create(gpu_point), Write(gpu_label),
            Create(fpga_point), Write(fpga_label)
        )

        # Draw pareto frontier
        pareto_curve = axes.plot(
            lambda x: 10000 / x,  # Hyperbola showing tradeoff
            x_range=[100, 1000],
            color=YELLOW,
            stroke_width=3
        )
        pareto_label = Text("Ideal Frontier", font_size=16, color=YELLOW)
        pareto_label.next_to(axes.c2p(500, 20), UP)

        self.play(Create(pareto_curve), Write(pareto_label))

        # Conclusion
        conclusion = VGroup(
            Text("✓ FPGA: Best latency + competitive throughput", font_size=20, color=GREEN),
            Text("✓ Ideal for real-time edge inference", font_size=20, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(DOWN)

        self.play(Write(conclusion))


class TestingWorkflow(Scene):
    """Scene 6: Swarm-Based Testing Workflow"""

    def construct(self):
        title = Text("AI Swarm-Based Verification Workflow", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Show swarm architecture
        self.show_swarm_architecture()
        self.wait(2)

        # Show parallel test execution
        self.show_parallel_testing()
        self.wait(2)

        # Show results aggregation
        self.show_results_aggregation()
        self.wait(2)

    def show_swarm_architecture(self):
        # Coordinator agent
        coordinator = Circle(radius=0.8, color=GOLD, fill_opacity=0.5)
        coordinator_label = Text("Coordinator\nAgent", font_size=20, weight=BOLD)
        coordinator_label.move_to(coordinator)
        coordinator_group = VGroup(coordinator, coordinator_label)

        self.play(Create(coordinator_group))

        # Worker agents in a circle
        num_workers = 6
        worker_names = ["Tester 1", "Tester 2", "Tester 3",
                       "Analyzer", "Validator", "Reporter"]
        worker_colors = [BLUE, BLUE, BLUE, GREEN, PURPLE, ORANGE]

        workers = VGroup()
        for i in range(num_workers):
            angle = i * TAU / num_workers
            pos = np.array([np.cos(angle), np.sin(angle), 0]) * 3

            worker = Circle(radius=0.5, color=worker_colors[i], fill_opacity=0.4)
            worker.move_to(pos)

            label = Text(worker_names[i], font_size=14).move_to(pos)

            # Connection to coordinator
            connection = Line(
                coordinator.get_center(),
                worker.get_center(),
                color=YELLOW,
                stroke_width=2
            )

            workers.add(VGroup(connection, worker, label))

        self.play(
            *[Create(worker) for worker in workers],
            run_time=2
        )

        # Show communication
        for worker in workers:
            self.play(
                worker[0].animate.set_color(GREEN),
                run_time=0.3
            )
            self.play(
                worker[0].animate.set_color(YELLOW),
                run_time=0.3
            )

    def show_parallel_testing(self):
        self.clear()

        subtitle = Text("Parallel Test Execution", font_size=32)
        subtitle.to_edge(UP, buff=1.2)
        self.play(Write(subtitle))

        # Test categories
        test_categories = [
            "Unit Tests",
            "Integration Tests",
            "Performance Tests",
            "Hardware Tests"
        ]

        test_colors = [BLUE, GREEN, ORANGE, PURPLE]

        # Create test lanes
        lanes = VGroup()
        for i, (category, color) in enumerate(zip(test_categories, test_colors)):
            y_pos = 2 - i * 1.2

            # Lane label
            label = Text(category, font_size=20, color=color)
            label.to_edge(LEFT, buff=0.5).shift(UP * y_pos)

            # Timeline
            timeline = Line(LEFT * 2, RIGHT * 5, color=color, stroke_width=3)
            timeline.shift(UP * y_pos)

            # Test blocks
            num_tests = np.random.randint(3, 6)
            test_blocks = VGroup(*[
                Rectangle(width=0.5, height=0.4, color=color, fill_opacity=0.6)
                .move_to(timeline.point_from_proportion(j / num_tests))
                for j in range(num_tests)
            ])

            lanes.add(VGroup(label, timeline, test_blocks))

        # Animate tests running in parallel
        for lane in lanes:
            self.play(Write(lane[0]), Create(lane[1]))

        # Run tests simultaneously
        all_tests = VGroup(*[lane[2] for lane in lanes])
        self.play(
            *[Create(test_block) for lane in lanes for test_block in lane[2]],
            run_time=2
        )

        # Show completion
        for lane in lanes:
            for test in lane[2]:
                self.play(
                    test.animate.set_fill(GREEN, opacity=0.8),
                    run_time=0.1
                )

        # Completion message
        completion = Text(
            "All tests passed in parallel! 4x speedup",
            font_size=28,
            color=GREEN
        ).to_edge(DOWN)
        self.play(Write(completion))

    def show_results_aggregation(self):
        self.clear()

        subtitle = Text("Results Aggregation & Reporting", font_size=32)
        subtitle.to_edge(UP, buff=1.2)
        self.play(Write(subtitle))

        # Individual test results flowing in
        result_sources = VGroup(*[
            Circle(radius=0.3, color=BLUE, fill_opacity=0.6)
            .shift(LEFT * 5 + UP * (1.5 - i * 1.0))
            for i in range(4)
        ])

        source_labels = VGroup(*[
            Text(f"Test {i+1}", font_size=16).next_to(source, LEFT, buff=0.1)
            for i, source in enumerate(result_sources)
        ])

        self.play(
            Create(result_sources),
            Write(source_labels)
        )

        # Aggregator
        aggregator = Rectangle(width=2, height=3, color=GREEN, fill_opacity=0.4)
        aggregator_label = Text("Result\nAggregator", font_size=20).move_to(aggregator)
        aggregator_group = VGroup(aggregator, aggregator_label)

        self.play(Create(aggregator_group))

        # Data flowing to aggregator
        for source in result_sources:
            data_packet = Square(side_length=0.2, color=YELLOW, fill_opacity=0.8)
            data_packet.move_to(source.get_center())

            path = Line(source.get_center(), aggregator.get_left())

            self.play(
                MoveAlongPath(data_packet, path),
                run_time=0.5
            )
            self.remove(data_packet)

        # Final report
        report = Rectangle(width=3, height=4, color=PURPLE, fill_opacity=0.3)
        report.shift(RIGHT * 4)

        report_content = VGroup(
            Text("Test Report", font_size=24, weight=BOLD, color=PURPLE),
            Text("━━━━━━━━━━━━", font_size=16),
            Text("✓ 142 tests passed", font_size=18, color=GREEN),
            Text("✗ 3 tests failed", font_size=18, color=RED),
            Text("⚠ 5 tests skipped", font_size=18, color=YELLOW),
            Text("━━━━━━━━━━━━", font_size=16),
            Text("Coverage: 94%", font_size=18, color=GREEN),
            Text("Time: 2.3s", font_size=18, color=BLUE)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).scale(0.7).move_to(report)

        self.play(Create(report))
        self.play(Write(report_content))

        # Export arrow
        export_arrow = Arrow(
            aggregator.get_right(),
            report.get_left(),
            color=GREEN,
            stroke_width=6
        )
        self.play(GrowArrow(export_arrow))


class FullPresentation(Scene):
    """Combined presentation with all scenes"""

    def construct(self):
        # Opening title
        main_title = Text(
            "AI-on-Chip Architecture\nUPduino FPGA Platform",
            font_size=52,
            gradient=(BLUE, PURPLE)
        )

        subtitle = Text(
            "Memory-Centric AI Acceleration for Edge Computing",
            font_size=28,
            color=GOLD
        )
        subtitle.next_to(main_title, DOWN, buff=0.5)

        self.play(
            Write(main_title),
            Write(subtitle)
        )
        self.wait(3)
        self.clear()

        # Scene transitions with section markers
        scenes = [
            ("AI Hardware Overview", AIChipOverview),
            ("FPGA Resource Mapping", FPGAResourceMapping),
            ("Dataflow Animation", DataflowAnimation),
            ("Memory-as-Inference", MemoryAsInference),
            ("Performance Comparison", PerformanceComparison),
            ("Testing Workflow", TestingWorkflow)
        ]

        for section_name, scene_class in scenes:
            # Section marker
            section_title = Text(f"Section: {section_name}", font_size=36, color=GOLD)
            section_underline = Line(LEFT * 4, RIGHT * 4, color=GOLD)
            section_underline.next_to(section_title, DOWN, buff=0.2)

            self.play(
                Write(section_title),
                Create(section_underline)
            )
            self.wait(1)
            self.clear()

            # Note: In actual use, you would instantiate and run each scene
            # This is a conceptual structure
            info_text = Text(
                f"Run: manim -pqh ai_chip_visualization.py {scene_class.__name__}",
                font_size=24,
                color=YELLOW
            )
            self.play(Write(info_text))
            self.wait(2)
            self.clear()

        # Closing
        closing = VGroup(
            Text("Thank You", font_size=52, gradient=(BLUE, PURPLE)),
            Text("Questions?", font_size=36, color=GOLD)
        ).arrange(DOWN, buff=0.5)

        self.play(Write(closing))
        self.wait(3)


# Additional utility function for rendering all scenes
def render_all_scenes():
    """
    Utility to render all scenes at high quality.
    Run with: python ai_chip_visualization.py
    """
    import os

    scenes = [
        "AIChipOverview",
        "FPGAResourceMapping",
        "DataflowAnimation",
        "MemoryAsInference",
        "PerformanceComparison",
        "TestingWorkflow"
    ]

    for scene in scenes:
        print(f"\nRendering {scene}...")
        os.system(f"manim -pqh ai_chip_visualization.py {scene}")

    print("\nAll scenes rendered successfully!")


if __name__ == "__main__":
    print("""
    AI-on-Chip Visualization Suite
    ==============================

    Available scenes:
    1. AIChipOverview - Hardware evolution and paradigm shift
    2. FPGAResourceMapping - Resource to neural network mapping
    3. DataflowAnimation - Animated dataflow through network
    4. MemoryAsInference - In-memory computing concepts
    5. PerformanceComparison - Performance metrics and charts
    6. TestingWorkflow - Swarm-based testing workflow
    7. FullPresentation - Complete presentation

    Usage:
        manim -pql ai_chip_visualization.py <SceneName>     # Low quality preview
        manim -pqh ai_chip_visualization.py <SceneName>     # High quality render
        manim -pqk ai_chip_visualization.py <SceneName>     # 4K quality render

    Render all:
        python ai_chip_visualization.py
    """)
