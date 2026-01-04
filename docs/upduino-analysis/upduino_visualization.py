"""
UPduino v3.0/3.1 FPGA Board Visualization
==========================================

A comprehensive Manim visualization script for new UPduino board users.
This script creates educational animations covering hardware architecture,
pin layouts, development workflow, and getting started guides.

Requirements:
    pip install manim

Usage:
    manim -pql upduino_visualization.py TitleScene
    manim -pql upduino_visualization.py HardwareArchitectureScene
    manim -pql upduino_visualization.py PinLayoutScene
    manim -pql upduino_visualization.py DevelopmentWorkflowScene
    manim -pql upduino_visualization.py TaskMappingScene
    manim -pql upduino_visualization.py GettingStartedScene

    # Render all scenes:
    manim -pql upduino_visualization.py
"""

from manim import *


# ============================================================================
# SCENE 1: TITLE SCENE
# ============================================================================

class TitleScene(Scene):
    """
    Opening title scene introducing the UPduino v3.0/3.1 board
    with animated text and board outline.
    """

    def construct(self):
        # Main title
        title = Text("UPduino v3.0/3.1", font_size=72, gradient=(BLUE, PURPLE))
        subtitle = Text("Lattice iCE40 UltraPlus FPGA", font_size=36, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.5)

        # Feature highlights
        features = VGroup(
            Text("✓ 5.3K LUT Logic Elements", font_size=24, color=GREEN),
            Text("✓ 1Mb SPRAM + 120Kb DPRAM", font_size=24, color=GREEN),
            Text("✓ Open-Source Toolchain", font_size=24, color=GREEN),
            Text("✓ USB Programmable", font_size=24, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        features.next_to(subtitle, DOWN, buff=1)

        # Board outline representation
        board_outline = Rectangle(
            width=4, height=2.5,
            color=WHITE,
            stroke_width=3
        ).shift(RIGHT * 3 + DOWN * 0.5)

        # FPGA chip representation
        fpga_chip = Square(
            side_length=1.5,
            color=BLUE,
            fill_opacity=0.3
        ).move_to(board_outline.get_center())
        fpga_label = Text("ICE40UP5K", font_size=20, color=WHITE)
        fpga_label.move_to(fpga_chip.get_center())

        # USB connector
        usb = Rectangle(
            width=0.3, height=0.5,
            color=YELLOW,
            fill_opacity=0.5
        ).move_to(board_outline.get_left() + RIGHT * 0.15)
        usb_label = Text("USB", font_size=12, color=YELLOW)
        usb_label.next_to(usb, LEFT, buff=0.1)

        # RGB LED
        led = Circle(radius=0.15, color=RED, fill_opacity=0.8)
        led.move_to(board_outline.get_top() + DOWN * 0.3)
        led_label = Text("RGB", font_size=10, color=RED)
        led_label.next_to(led, UP, buff=0.1)

        # Animations
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle), run_time=1)
        self.wait(0.5)

        self.play(
            Create(board_outline),
            Create(fpga_chip),
            Write(fpga_label),
            run_time=2
        )

        self.play(
            FadeIn(usb), Write(usb_label),
            FadeIn(led), Write(led_label),
            run_time=1.5
        )

        # LED blink animation
        for _ in range(3):
            self.play(
                led.animate.set_color(RED),
                run_time=0.3
            )
            self.play(
                led.animate.set_color(GREEN),
                run_time=0.3
            )
            self.play(
                led.animate.set_color(BLUE),
                run_time=0.3
            )

        self.play(FadeIn(features, shift=UP), run_time=2)
        self.wait(2)

        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)


# ============================================================================
# SCENE 2: HARDWARE ARCHITECTURE
# ============================================================================

class HardwareArchitectureScene(Scene):
    """
    Detailed breakdown of the UPduino hardware components and their
    interconnections, showing the ICE40UP5K, USB interface, memory, etc.
    """

    def construct(self):
        # Scene title
        title = Text("Hardware Architecture", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # ===== FPGA CHIP (Center) =====
        fpga = RoundedRectangle(
            width=3, height=2.5,
            corner_radius=0.2,
            color=BLUE,
            fill_opacity=0.2,
            stroke_width=3
        ).shift(UP * 0.3)

        fpga_title = Text("ICE40UP5K FPGA", font_size=28, color=BLUE, weight=BOLD)
        fpga_title.move_to(fpga.get_top() + DOWN * 0.3)

        # FPGA specs
        fpga_specs = VGroup(
            Text("5.3K LUTs", font_size=18, color=WHITE),
            Text("1Mb SPRAM", font_size=18, color=YELLOW),
            Text("120Kb DPRAM", font_size=18, color=YELLOW),
            Text("8× DSP (16×16)", font_size=18, color=GREEN),
            Text("2× I2C, 2× SPI", font_size=18, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        fpga_specs.scale(0.8)
        fpga_specs.move_to(fpga.get_center() + DOWN * 0.3)

        self.play(Create(fpga), Write(fpga_title))
        self.play(FadeIn(fpga_specs, shift=UP), run_time=1.5)
        self.wait(1)

        # ===== USB INTERFACE (Left) =====
        usb_box = RoundedRectangle(
            width=2, height=1.2,
            corner_radius=0.15,
            color=YELLOW,
            fill_opacity=0.15,
            stroke_width=2
        ).shift(LEFT * 4.5 + UP * 1.5)

        usb_title = Text("FTDI FT232H", font_size=20, color=YELLOW, weight=BOLD)
        usb_title.move_to(usb_box.get_top() + DOWN * 0.25)

        usb_info = Text("USB-UART/JTAG", font_size=14, color=WHITE)
        usb_info.move_to(usb_box.get_center() + DOWN * 0.15)

        # Connection from USB to FPGA
        usb_arrow = Arrow(
            usb_box.get_right(),
            fpga.get_left() + UP * 0.5,
            color=YELLOW,
            stroke_width=3,
            buff=0.1
        )
        usb_label = Text("SPI/UART", font_size=12, color=YELLOW)
        usb_label.next_to(usb_arrow, UP, buff=0.1)

        self.play(
            Create(usb_box),
            Write(usb_title),
            Write(usb_info)
        )
        self.play(GrowArrow(usb_arrow), Write(usb_label))
        self.wait(1)

        # ===== FLASH MEMORY (Right) =====
        flash_box = RoundedRectangle(
            width=2, height=1.2,
            corner_radius=0.15,
            color=PURPLE,
            fill_opacity=0.15,
            stroke_width=2
        ).shift(RIGHT * 4.5 + UP * 1.5)

        flash_title = Text("4MB SPI Flash", font_size=20, color=PURPLE, weight=BOLD)
        flash_title.move_to(flash_box.get_top() + DOWN * 0.25)

        flash_info = Text("Config Storage", font_size=14, color=WHITE)
        flash_info.move_to(flash_box.get_center() + DOWN * 0.15)

        flash_arrow = Arrow(
            flash_box.get_left(),
            fpga.get_right() + UP * 0.5,
            color=PURPLE,
            stroke_width=3,
            buff=0.1
        )
        flash_label = Text("SPI", font_size=12, color=PURPLE)
        flash_label.next_to(flash_arrow, UP, buff=0.1)

        self.play(
            Create(flash_box),
            Write(flash_title),
            Write(flash_info)
        )
        self.play(GrowArrow(flash_arrow), Write(flash_label))
        self.wait(1)

        # ===== RGB LED (Bottom Left) =====
        led_box = RoundedRectangle(
            width=1.8, height=0.8,
            corner_radius=0.15,
            color=RED,
            fill_opacity=0.15,
            stroke_width=2
        ).shift(LEFT * 3.5 + DOWN * 1.8)

        led_title = Text("RGB LED", font_size=18, color=RED, weight=BOLD)
        led_title.move_to(led_box.get_center())

        # LED circles
        led_r = Circle(radius=0.12, color=RED, fill_opacity=0.8)
        led_g = Circle(radius=0.12, color=GREEN, fill_opacity=0.8)
        led_b = Circle(radius=0.12, color=BLUE, fill_opacity=0.8)
        leds = VGroup(led_r, led_g, led_b).arrange(RIGHT, buff=0.15)
        leds.next_to(led_box, DOWN, buff=0.2)

        led_arrow = Arrow(
            fpga.get_bottom() + LEFT * 0.8,
            led_box.get_top(),
            color=RED,
            stroke_width=2,
            buff=0.1
        )
        led_label = Text("PWM", font_size=10, color=RED)
        led_label.next_to(led_arrow, LEFT, buff=0.1)

        self.play(Create(led_box), Write(led_title))
        self.play(FadeIn(leds), GrowArrow(led_arrow), Write(led_label))
        self.wait(0.5)

        # ===== OSCILLATOR (Bottom Center) =====
        osc_box = RoundedRectangle(
            width=1.8, height=0.8,
            corner_radius=0.15,
            color=ORANGE,
            fill_opacity=0.15,
            stroke_width=2
        ).shift(DOWN * 1.8)

        osc_title = Text("12MHz OSC", font_size=18, color=ORANGE, weight=BOLD)
        osc_title.move_to(osc_box.get_center())

        osc_arrow = Arrow(
            fpga.get_bottom(),
            osc_box.get_top(),
            color=ORANGE,
            stroke_width=2,
            buff=0.1
        )
        osc_label = Text("CLK", font_size=10, color=ORANGE)
        osc_label.next_to(osc_arrow, RIGHT, buff=0.1)

        self.play(Create(osc_box), Write(osc_title))
        self.play(GrowArrow(osc_arrow), Write(osc_label))
        self.wait(0.5)

        # ===== VOLTAGE REGULATORS (Bottom Right) =====
        vreg_box = RoundedRectangle(
            width=2, height=0.8,
            corner_radius=0.15,
            color=GREEN,
            fill_opacity=0.15,
            stroke_width=2
        ).shift(RIGHT * 3.5 + DOWN * 1.8)

        vreg_title = Text("Power Regs", font_size=18, color=GREEN, weight=BOLD)
        vreg_title.move_to(vreg_box.get_top() + DOWN * 0.2)

        vreg_info = Text("3.3V & 1.2V", font_size=12, color=WHITE)
        vreg_info.move_to(vreg_box.get_center() + DOWN * 0.15)

        vreg_arrow = Arrow(
            fpga.get_bottom() + RIGHT * 0.8,
            vreg_box.get_top(),
            color=GREEN,
            stroke_width=2,
            buff=0.1
        )
        vreg_label = Text("PWR", font_size=10, color=GREEN)
        vreg_label.next_to(vreg_arrow, RIGHT, buff=0.1)

        self.play(Create(vreg_box), Write(vreg_title), Write(vreg_info))
        self.play(GrowArrow(vreg_arrow), Write(vreg_label))
        self.wait(2)

        # Highlight data flow
        self.play(
            Indicate(usb_box, color=YELLOW, scale_factor=1.1),
            Indicate(fpga, color=BLUE, scale_factor=1.05),
            Indicate(flash_box, color=PURPLE, scale_factor=1.1),
            run_time=2
        )

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)


# ============================================================================
# SCENE 3: PIN LAYOUT
# ============================================================================

class PinLayoutScene(Scene):
    """
    Comprehensive pin layout showing all 32 GPIO pins, power pins,
    and special function pins with color coding and labels.
    """

    def construct(self):
        # Scene title
        title = Text("Pin Layout & Configuration", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Board representation
        board = Rectangle(width=6, height=8, color=WHITE, stroke_width=3)
        board.shift(DOWN * 0.3)

        self.play(Create(board))
        self.wait(0.5)

        # ===== LEFT SIDE PINS (GPIO 0-15) =====
        left_pins = VGroup()
        pin_configs_left = [
            ("GPIO 0", BLUE, "General I/O"),
            ("GPIO 1", BLUE, "General I/O"),
            ("GPIO 2", BLUE, "General I/O"),
            ("GPIO 3", BLUE, "General I/O"),
            ("GPIO 4", ORANGE, "SPI_SCK"),
            ("GPIO 5", ORANGE, "SPI_SS"),
            ("GPIO 6", ORANGE, "SPI_MISO"),
            ("GPIO 7", ORANGE, "SPI_MOSI"),
            ("GPIO 8", BLUE, "General I/O"),
            ("GPIO 9", BLUE, "General I/O"),
            ("GPIO 10", BLUE, "General I/O"),
            ("GPIO 11", BLUE, "General I/O"),
            ("GND", RED, "Ground"),
            ("3.3V", GREEN, "Power"),
            ("GPIO 14", BLUE, "General I/O"),
            ("GPIO 15", BLUE, "General I/O"),
        ]

        y_start = board.get_top()[1] - 0.5
        for i, (name, color, function) in enumerate(pin_configs_left):
            y_pos = y_start - i * 0.45

            # Pin dot
            pin_dot = Circle(radius=0.08, color=color, fill_opacity=1)
            pin_dot.move_to(board.get_left() + RIGHT * 0.15 + UP * y_pos)

            # Pin label
            pin_label = Text(name, font_size=12, color=color)
            pin_label.next_to(pin_dot, LEFT, buff=0.15)

            # Function label
            func_label = Text(function, font_size=8, color=GRAY)
            func_label.next_to(pin_dot, RIGHT, buff=0.15)

            pin_group = VGroup(pin_dot, pin_label, func_label)
            left_pins.add(pin_group)

        # ===== RIGHT SIDE PINS (GPIO 16-31) =====
        right_pins = VGroup()
        pin_configs_right = [
            ("GPIO 16", BLUE, "General I/O"),
            ("GPIO 17", BLUE, "General I/O"),
            ("GPIO 18", BLUE, "General I/O"),
            ("GPIO 19", BLUE, "General I/O"),
            ("GPIO 20", PURPLE, "I2C_SDA"),
            ("GPIO 21", PURPLE, "I2C_SCL"),
            ("GPIO 22", BLUE, "General I/O"),
            ("GPIO 23", BLUE, "General I/O"),
            ("RGB_R", RED, "LED Red"),
            ("RGB_G", GREEN, "LED Green"),
            ("RGB_B", BLUE, "LED Blue"),
            ("GPIO 27", BLUE, "General I/O"),
            ("GPIO 28", BLUE, "General I/O"),
            ("GND", RED, "Ground"),
            ("5V", GREEN, "Power In"),
            ("GPIO 31", BLUE, "General I/O"),
        ]

        for i, (name, color, function) in enumerate(pin_configs_right):
            y_pos = y_start - i * 0.45

            pin_dot = Circle(radius=0.08, color=color, fill_opacity=1)
            pin_dot.move_to(board.get_right() + LEFT * 0.15 + UP * y_pos)

            pin_label = Text(name, font_size=12, color=color)
            pin_label.next_to(pin_dot, RIGHT, buff=0.15)

            func_label = Text(function, font_size=8, color=GRAY)
            func_label.next_to(pin_dot, LEFT, buff=0.15)

            pin_group = VGroup(pin_dot, pin_label, func_label)
            right_pins.add(pin_group)

        # Animate pin creation
        self.play(
            LaggedStart(*[FadeIn(pin) for pin in left_pins], lag_ratio=0.05),
            run_time=3
        )
        self.wait(0.5)

        self.play(
            LaggedStart(*[FadeIn(pin) for pin in right_pins], lag_ratio=0.05),
            run_time=3
        )
        self.wait(1)

        # ===== LEGEND =====
        legend_title = Text("Pin Types:", font_size=20, color=WHITE, weight=BOLD)
        legend_title.to_edge(DOWN, buff=1.5).shift(LEFT * 4)

        legend_items = VGroup(
            VGroup(
                Circle(radius=0.08, color=BLUE, fill_opacity=1),
                Text("GPIO", font_size=14, color=WHITE)
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Circle(radius=0.08, color=ORANGE, fill_opacity=1),
                Text("SPI", font_size=14, color=WHITE)
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Circle(radius=0.08, color=PURPLE, fill_opacity=1),
                Text("I2C", font_size=14, color=WHITE)
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Circle(radius=0.08, color=RED, fill_opacity=1),
                Text("LED/GND", font_size=14, color=WHITE)
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Circle(radius=0.08, color=GREEN, fill_opacity=1),
                Text("Power", font_size=14, color=WHITE)
            ).arrange(RIGHT, buff=0.2),
        ).arrange(RIGHT, buff=0.5)
        legend_items.next_to(legend_title, DOWN, buff=0.3)

        self.play(Write(legend_title))
        self.play(FadeIn(legend_items, shift=UP), run_time=1.5)
        self.wait(1)

        # Highlight special function pins
        spi_pins = VGroup(*[left_pins[i] for i in [4, 5, 6, 7]])
        self.play(Indicate(spi_pins, color=ORANGE, scale_factor=1.3), run_time=2)
        self.wait(0.5)

        i2c_pins = VGroup(*[right_pins[i] for i in [4, 5]])
        self.play(Indicate(i2c_pins, color=PURPLE, scale_factor=1.3), run_time=2)
        self.wait(0.5)

        led_pins = VGroup(*[right_pins[i] for i in [8, 9, 10]])
        self.play(Indicate(led_pins, color=YELLOW, scale_factor=1.3), run_time=2)

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)


# ============================================================================
# SCENE 4: DEVELOPMENT WORKFLOW
# ============================================================================

class DevelopmentWorkflowScene(Scene):
    """
    Step-by-step visualization of the FPGA development workflow from
    toolchain installation to programming and testing.
    """

    def construct(self):
        # Scene title
        title = Text("Development Workflow", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Workflow steps container
        steps_container = VGroup()

        # ===== STEP 1: Install Toolchain =====
        step1_box = self.create_step_box(
            "1. Install Toolchain",
            ["IceStorm - Place & Route", "Yosys - Synthesis", "NextPNR - Timing", "APIO - Build System"],
            BLUE,
            position=UP * 2.5
        )
        steps_container.add(step1_box)

        self.play(FadeIn(step1_box, shift=RIGHT), run_time=1.5)
        self.wait(1)

        # ===== STEP 2: Write HDL Code =====
        step2_box = self.create_step_box(
            "2. Write HDL Code",
            ["Verilog/SystemVerilog", "VHDL", "Chisel/SpinalHDL", "Amaranth (Python)"],
            GREEN,
            position=UP * 0.8
        )
        steps_container.add(step2_box)

        # Arrow from step 1 to step 2
        arrow1 = Arrow(
            step1_box.get_bottom(),
            step2_box.get_top(),
            color=WHITE,
            stroke_width=3,
            buff=0.1
        )

        self.play(GrowArrow(arrow1))
        self.play(FadeIn(step2_box, shift=RIGHT), run_time=1.5)
        self.wait(1)

        # Code example popup
        code_example = Code(
            code="""module blink (
    input clk,
    output reg led
);
    reg [23:0] counter;
    always @(posedge clk) begin
        counter <= counter + 1;
        led <= counter[23];
    end
endmodule""",
            language="verilog",
            font_size=14,
            background="window",
            insert_line_no=False
        ).scale(0.6).shift(RIGHT * 3.5 + UP * 0.8)

        self.play(FadeIn(code_example, shift=LEFT), run_time=1)
        self.wait(1.5)
        self.play(FadeOut(code_example), run_time=0.8)

        # ===== STEP 3: Synthesize & P&R =====
        step3_box = self.create_step_box(
            "3. Synthesize & Place-Route",
            ["yosys synth_ice40", "nextpnr-ice40", "icepack bitstream", "Timing analysis"],
            ORANGE,
            position=DOWN * 0.9
        )
        steps_container.add(step3_box)

        arrow2 = Arrow(
            step2_box.get_bottom(),
            step3_box.get_top(),
            color=WHITE,
            stroke_width=3,
            buff=0.1
        )

        self.play(GrowArrow(arrow2))
        self.play(FadeIn(step3_box, shift=RIGHT), run_time=1.5)
        self.wait(1)

        # ===== STEP 4: Program via USB =====
        step4_box = self.create_step_box(
            "4. Program FPGA",
            ["iceprog bitstream.bin", "Upload to Flash/SRAM", "USB connection", "Verify programming"],
            PURPLE,
            position=DOWN * 2.6
        )
        steps_container.add(step4_box)

        arrow3 = Arrow(
            step3_box.get_bottom(),
            step4_box.get_top(),
            color=WHITE,
            stroke_width=3,
            buff=0.1
        )

        self.play(GrowArrow(arrow3))
        self.play(FadeIn(step4_box, shift=RIGHT), run_time=1.5)
        self.wait(1)

        # ===== STEP 5: Test & Debug =====
        # Position to the right side
        step5_box = self.create_step_box(
            "5. Test & Debug",
            ["Logic analyzer", "UART debugging", "LED indicators", "Iterate & improve"],
            RED,
            position=RIGHT * 3.5
        )

        # Curved arrow from step 4 to step 5
        arrow4 = CurvedArrow(
            step4_box.get_right(),
            step5_box.get_bottom(),
            color=WHITE,
            stroke_width=3
        )

        self.play(GrowArrow(arrow4))
        self.play(FadeIn(step5_box, shift=UP), run_time=1.5)
        self.wait(1)

        # Feedback loop arrow
        feedback_arrow = CurvedArrow(
            step5_box.get_top(),
            step2_box.get_right(),
            color=YELLOW,
            stroke_width=3
        )
        feedback_label = Text("Iterate", font_size=16, color=YELLOW)
        feedback_label.next_to(feedback_arrow, UP, buff=0.1)

        self.play(GrowArrow(feedback_arrow), Write(feedback_label))
        self.wait(1)

        # Highlight the complete workflow
        self.play(
            Indicate(steps_container, color=BLUE, scale_factor=1.05),
            run_time=2
        )

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)

    def create_step_box(self, title_text, items, color, position):
        """Helper method to create a workflow step box"""
        box = RoundedRectangle(
            width=3.5, height=1.3,
            corner_radius=0.15,
            color=color,
            fill_opacity=0.15,
            stroke_width=2
        ).shift(position)

        title = Text(title_text, font_size=18, color=color, weight=BOLD)
        title.move_to(box.get_top() + DOWN * 0.25)

        item_list = VGroup(*[
            Text(f"• {item}", font_size=10, color=WHITE)
            for item in items
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        item_list.scale(0.9)
        item_list.move_to(box.get_center() + DOWN * 0.15)

        return VGroup(box, title, item_list)


# ============================================================================
# SCENE 5: TASK MAPPING
# ============================================================================

class TaskMappingScene(Scene):
    """
    Maps different project complexity levels to appropriate tasks,
    from beginner LED blink to advanced RISC-V processors.
    """

    def construct(self):
        # Scene title
        title = Text("Project Ideas & Task Mapping", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Difficulty scale
        difficulty_scale = self.create_difficulty_scale()
        difficulty_scale.shift(DOWN * 3)
        self.play(Create(difficulty_scale), run_time=1.5)
        self.wait(0.5)

        # ===== BEGINNER TASKS =====
        beginner_box = self.create_task_box(
            "Beginner Tasks",
            [
                "LED Blink - Basic clocking",
                "Button Input - Debouncing",
                "PWM LED - Analog simulation",
                "7-Segment Display",
                "Binary Counter"
            ],
            GREEN,
            position=LEFT * 4 + UP * 1.5
        )

        beginner_icon = Text("🌱", font_size=40)
        beginner_icon.next_to(beginner_box, UP, buff=0.2)

        self.play(
            FadeIn(beginner_box, shift=UP),
            FadeIn(beginner_icon, shift=DOWN),
            run_time=1.5
        )

        # Connect to difficulty scale
        beginner_arrow = Arrow(
            beginner_box.get_bottom(),
            difficulty_scale.get_left() + UP * 0.3 + RIGHT * 0.5,
            color=GREEN,
            stroke_width=2,
            buff=0.1
        )
        self.play(GrowArrow(beginner_arrow))
        self.wait(1)

        # ===== INTERMEDIATE TASKS =====
        intermediate_box = self.create_task_box(
            "Intermediate Tasks",
            [
                "UART Communication",
                "SPI Interface - Flash R/W",
                "I2C Sensors",
                "VGA Signal Generator",
                "Simple CPU (4-bit)"
            ],
            YELLOW,
            position=UP * 1.5
        )

        intermediate_icon = Text("🔧", font_size=40)
        intermediate_icon.next_to(intermediate_box, UP, buff=0.2)

        self.play(
            FadeIn(intermediate_box, shift=UP),
            FadeIn(intermediate_icon, shift=DOWN),
            run_time=1.5
        )

        intermediate_arrow = Arrow(
            intermediate_box.get_bottom(),
            difficulty_scale.get_center() + UP * 0.3,
            color=YELLOW,
            stroke_width=2,
            buff=0.1
        )
        self.play(GrowArrow(intermediate_arrow))
        self.wait(1)

        # ===== ADVANCED TASKS =====
        advanced_box = self.create_task_box(
            "Advanced Tasks",
            [
                "RISC-V Soft Processor",
                "DSP - Audio/Signal Proc",
                "DMA Controllers",
                "Custom Accelerators",
                "Multi-core Systems"
            ],
            RED,
            position=RIGHT * 4 + UP * 1.5
        )

        advanced_icon = Text("🚀", font_size=40)
        advanced_icon.next_to(advanced_box, UP, buff=0.2)

        self.play(
            FadeIn(advanced_box, shift=UP),
            FadeIn(advanced_icon, shift=DOWN),
            run_time=1.5
        )

        advanced_arrow = Arrow(
            advanced_box.get_bottom(),
            difficulty_scale.get_right() + UP * 0.3 + LEFT * 0.5,
            color=RED,
            stroke_width=2,
            buff=0.1
        )
        self.play(GrowArrow(advanced_arrow))
        self.wait(1)

        # ===== RESOURCE REQUIREMENTS =====
        resources = VGroup(
            Text("Resource Usage:", font_size=20, color=WHITE, weight=BOLD),
            VGroup(
                Text("Beginner: ", font_size=16, color=GREEN),
                Text("<10% LUTs", font_size=16, color=WHITE)
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Text("Intermediate: ", font_size=16, color=YELLOW),
                Text("10-50% LUTs", font_size=16, color=WHITE)
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Text("Advanced: ", font_size=16, color=RED),
                Text("50-100% LUTs", font_size=16, color=WHITE)
            ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        resources.to_edge(DOWN, buff=0.5).shift(RIGHT * 3)

        self.play(FadeIn(resources, shift=UP), run_time=1.5)
        self.wait(2)

        # Highlight progression
        self.play(
            Indicate(beginner_box, color=GREEN, scale_factor=1.1),
            run_time=1
        )
        self.play(
            Indicate(intermediate_box, color=YELLOW, scale_factor=1.1),
            run_time=1
        )
        self.play(
            Indicate(advanced_box, color=RED, scale_factor=1.1),
            run_time=1
        )

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)

    def create_difficulty_scale(self):
        """Create a horizontal difficulty scale bar"""
        scale_bar = Rectangle(width=8, height=0.5, color=WHITE, stroke_width=2)

        # Gradient fill (simulated with multiple rectangles)
        gradient_parts = VGroup()
        num_parts = 20
        for i in range(num_parts):
            part_width = 8 / num_parts
            color_interp = interpolate_color(GREEN, RED, i / num_parts)
            part = Rectangle(
                width=part_width,
                height=0.5,
                color=color_interp,
                fill_opacity=0.6,
                stroke_width=0
            ).shift(LEFT * 4 + RIGHT * (i * part_width + part_width / 2))
            gradient_parts.add(part)

        # Labels
        easy_label = Text("Easy", font_size=16, color=GREEN)
        easy_label.next_to(scale_bar, LEFT, buff=0.3)

        hard_label = Text("Advanced", font_size=16, color=RED)
        hard_label.next_to(scale_bar, RIGHT, buff=0.3)

        return VGroup(gradient_parts, scale_bar, easy_label, hard_label)

    def create_task_box(self, title_text, tasks, color, position):
        """Helper method to create a task category box"""
        box = RoundedRectangle(
            width=3, height=2,
            corner_radius=0.15,
            color=color,
            fill_opacity=0.15,
            stroke_width=3
        ).shift(position)

        title = Text(title_text, font_size=20, color=color, weight=BOLD)
        title.move_to(box.get_top() + DOWN * 0.35)

        task_list = VGroup(*[
            Text(f"• {task}", font_size=12, color=WHITE)
            for task in tasks
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        task_list.move_to(box.get_center() + DOWN * 0.2)

        return VGroup(box, title, task_list)


# ============================================================================
# SCENE 6: GETTING STARTED CHECKLIST
# ============================================================================

class GettingStartedScene(Scene):
    """
    Interactive checklist for new users covering hardware inspection,
    software setup, first program, and verification steps.
    """

    def construct(self):
        # Scene title
        title = Text("Getting Started Checklist", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Introduction text
        intro = Text(
            "Follow these steps to start your FPGA journey!",
            font_size=20,
            color=GRAY
        )
        intro.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(intro))
        self.wait(1)

        # Checklist container
        checklist_items = [
            {
                "title": "1. Hardware Inspection",
                "items": [
                    "Inspect board for damage",
                    "Check USB connector",
                    "Verify RGB LED is intact",
                    "Ensure pins are not bent"
                ],
                "color": BLUE
            },
            {
                "title": "2. Driver Installation",
                "items": [
                    "Install FTDI drivers (Windows)",
                    "Install libftdi/libusb (Linux)",
                    "Test USB connection: lsusb",
                    "Verify device permissions"
                ],
                "color": GREEN
            },
            {
                "title": "3. Toolchain Setup",
                "items": [
                    "Install APIO: pip install apio",
                    "Install packages: apio install -a",
                    "Test: apio system --lsftdi",
                    "Verify iceprog is available"
                ],
                "color": ORANGE
            },
            {
                "title": "4. First Program",
                "items": [
                    "Create blink.v LED blinker",
                    "Write constraints file (.pcf)",
                    "Run: apio build",
                    "Upload: apio upload"
                ],
                "color": PURPLE
            },
            {
                "title": "5. Verification",
                "items": [
                    "Observe RGB LED blinking",
                    "Check timing reports",
                    "Verify resource usage",
                    "Test modifications"
                ],
                "color": RED
            }
        ]

        # Create checklist with animations
        y_position = 2
        all_checkboxes = []

        for section in checklist_items:
            # Section title
            section_title = Text(
                section["title"],
                font_size=22,
                color=section["color"],
                weight=BOLD
            )
            section_title.to_edge(LEFT, buff=0.5).shift(UP * y_position)

            self.play(Write(section_title), run_time=0.8)
            y_position -= 0.5

            # Section items with checkboxes
            for item_text in section["items"]:
                # Checkbox
                checkbox = Square(
                    side_length=0.25,
                    color=section["color"],
                    stroke_width=2
                )
                checkbox.to_edge(LEFT, buff=1).shift(UP * y_position)

                # Item text
                item = Text(item_text, font_size=14, color=WHITE)
                item.next_to(checkbox, RIGHT, buff=0.3)

                # Checkmark (initially invisible)
                checkmark = Text("✓", font_size=20, color=GREEN)
                checkmark.move_to(checkbox.get_center())
                checkmark.set_opacity(0)

                item_group = VGroup(checkbox, item, checkmark)
                all_checkboxes.append(item_group)

                self.play(FadeIn(item_group), run_time=0.4)
                y_position -= 0.35

            y_position -= 0.3  # Extra space between sections

        self.wait(1)

        # Animate checking off items sequentially
        self.play(FadeOut(intro), run_time=0.5)

        check_instruction = Text(
            "Completing checklist...",
            font_size=18,
            color=YELLOW
        )
        check_instruction.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(check_instruction))

        for item_group in all_checkboxes:
            checkbox, item, checkmark = item_group

            # Fill checkbox and show checkmark
            self.play(
                checkbox.animate.set_fill(GREEN, opacity=0.3),
                checkmark.animate.set_opacity(1),
                item.animate.set_color(GRAY),
                run_time=0.3
            )
            self.wait(0.1)

        self.play(FadeOut(check_instruction))
        self.wait(0.5)

        # Completion message
        completion = VGroup(
            Text("✓ Setup Complete!", font_size=36, color=GREEN, weight=BOLD),
            Text("You're ready to start developing!", font_size=20, color=WHITE),
            Text("Visit: github.com/gtjennings1/UPDuino-v3.0", font_size=16, color=BLUE)
        ).arrange(DOWN, buff=0.3)
        completion.next_to(title, DOWN, buff=0.5)

        self.play(FadeIn(completion, shift=UP), run_time=1.5)
        self.wait(2)

        # Final celebration effect
        stars = VGroup(*[
            Text("⭐", font_size=30, color=YELLOW).move_to(
                np.array([
                    np.random.uniform(-6, 6),
                    np.random.uniform(-3, 3),
                    0
                ])
            )
            for _ in range(15)
        ])

        self.play(
            LaggedStart(*[FadeIn(star, scale=0.5) for star in stars], lag_ratio=0.1),
            run_time=2
        )

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)


# ============================================================================
# MASTER SCENE - Renders all scenes in sequence
# ============================================================================

class AllScenes(Scene):
    """
    Meta-scene that renders all scenes in sequence for a complete presentation.
    Use this to render the entire visualization in one go.

    Usage:
        manim -pql upduino_visualization.py AllScenes
    """

    def construct(self):
        scenes = [
            TitleScene,
            HardwareArchitectureScene,
            PinLayoutScene,
            DevelopmentWorkflowScene,
            TaskMappingScene,
            GettingStartedScene
        ]

        for scene_class in scenes:
            scene = scene_class()
            scene.construct()
            self.wait(1)
