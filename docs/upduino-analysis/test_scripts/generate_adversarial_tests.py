#!/usr/bin/env python3
"""
Generate adversarial test inputs for FPGA AI hardware robustness testing

This script creates adversarial examples using various attack techniques
to test the robustness of quantized neural networks on FPGA hardware.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class AdversarialTestGenerator:
    """Generate adversarial test inputs for FPGA testing"""

    def __init__(self, bit_width: int = 8, input_shape: Tuple[int, ...] = (28, 28)):
        self.bit_width = bit_width
        self.input_shape = input_shape
        self.q_min = -(2 ** (bit_width - 1))
        self.q_max = 2 ** (bit_width - 1) - 1

    def quantize(self, data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Quantize floating-point data to specified bit width"""
        data_min, data_max = data.min(), data.max()

        if data_max - data_min < 1e-8:  # Avoid division by zero
            scale = 1.0
        else:
            scale = (data_max - data_min) / (self.q_max - self.q_min)

        zero_point = self.q_min - data_min / scale if scale > 0 else 0

        quantized = np.clip(
            np.round(data / scale + zero_point),
            self.q_min,
            self.q_max
        )

        return quantized.astype(np.int16), scale, zero_point

    def fgsm_attack(
        self,
        image: np.ndarray,
        epsilon: float = 0.1
    ) -> Dict:
        """
        Fast Gradient Sign Method attack

        Args:
            image: Input image (normalized to [0, 1])
            epsilon: Perturbation magnitude

        Returns:
            Adversarial test vector
        """
        # Generate random gradient direction (simplified FGSM without model)
        gradient = np.random.randn(*image.shape)
        gradient = gradient / (np.linalg.norm(gradient) + 1e-8)

        # Apply perturbation
        perturbation = epsilon * np.sign(gradient)
        adversarial = np.clip(image + perturbation, 0, 1)

        # Quantize
        quantized, scale, zero_point = self.quantize(adversarial)

        return {
            'type': 'adversarial',
            'attack': 'fgsm',
            'epsilon': epsilon,
            'input': quantized.tolist(),
            'metadata': {
                'original': image.tolist(),
                'perturbation': perturbation.tolist(),
                'scale': float(scale),
                'zero_point': float(zero_point),
                'bit_width': self.bit_width,
                'l2_norm': float(np.linalg.norm(perturbation))
            }
        }

    def random_noise_attack(
        self,
        image: np.ndarray,
        noise_level: float = 0.05
    ) -> Dict:
        """
        Add random Gaussian noise

        Args:
            image: Input image
            noise_level: Standard deviation of Gaussian noise

        Returns:
            Noisy test vector
        """
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = np.clip(image + noise, 0, 1)

        # Quantize
        quantized, scale, zero_point = self.quantize(noisy_image)

        return {
            'type': 'adversarial',
            'attack': 'random_noise',
            'noise_level': noise_level,
            'input': quantized.tolist(),
            'metadata': {
                'original': image.tolist(),
                'noise': noise.tolist(),
                'scale': float(scale),
                'zero_point': float(zero_point),
                'bit_width': self.bit_width,
                'snr_db': float(10 * np.log10(np.var(image) / np.var(noise)))
            }
        }

    def salt_and_pepper_noise(
        self,
        image: np.ndarray,
        density: float = 0.05
    ) -> Dict:
        """
        Add salt-and-pepper (impulse) noise

        Args:
            image: Input image
            density: Fraction of pixels to corrupt

        Returns:
            Corrupted test vector
        """
        noisy = image.copy()
        num_pixels = int(density * image.size)

        # Salt (white pixels)
        coords = [np.random.randint(0, i - 1, num_pixels // 2) for i in image.shape]
        noisy[tuple(coords)] = 1.0

        # Pepper (black pixels)
        coords = [np.random.randint(0, i - 1, num_pixels // 2) for i in image.shape]
        noisy[tuple(coords)] = 0.0

        # Quantize
        quantized, scale, zero_point = self.quantize(noisy)

        return {
            'type': 'adversarial',
            'attack': 'salt_pepper',
            'density': density,
            'input': quantized.tolist(),
            'metadata': {
                'original': image.tolist(),
                'corrupted_pixels': num_pixels,
                'scale': float(scale),
                'zero_point': float(zero_point),
                'bit_width': self.bit_width
            }
        }

    def occlusion_attack(
        self,
        image: np.ndarray,
        box_size: Tuple[int, int] = (5, 5),
        position: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Occlude part of the image with black box

        Args:
            image: Input image
            box_size: Size of occlusion box (height, width)
            position: Top-left corner of box (if None, random)

        Returns:
            Occluded test vector
        """
        occluded = image.copy()

        if position is None:
            max_h = image.shape[0] - box_size[0]
            max_w = image.shape[1] - box_size[1]
            position = (
                np.random.randint(0, max_h),
                np.random.randint(0, max_w)
            )

        h, w = position
        occluded[h:h+box_size[0], w:w+box_size[1]] = 0.0

        # Quantize
        quantized, scale, zero_point = self.quantize(occluded)

        return {
            'type': 'adversarial',
            'attack': 'occlusion',
            'box_size': box_size,
            'position': position,
            'input': quantized.tolist(),
            'metadata': {
                'original': image.tolist(),
                'occluded_area_pct': (box_size[0] * box_size[1]) / image.size * 100,
                'scale': float(scale),
                'zero_point': float(zero_point),
                'bit_width': self.bit_width
            }
        }

    def rotation_perturbation(
        self,
        image: np.ndarray,
        angle: float = 15.0
    ) -> Dict:
        """
        Rotate image (simplified - just shifts pixels)

        Args:
            image: Input image
            angle: Rotation angle in degrees

        Returns:
            Rotated test vector
        """
        # Simplified rotation: circular shift
        shift_h = int(image.shape[0] * angle / 360)
        shift_w = int(image.shape[1] * angle / 360)

        rotated = np.roll(image, (shift_h, shift_w), axis=(0, 1))

        # Quantize
        quantized, scale, zero_point = self.quantize(rotated)

        return {
            'type': 'adversarial',
            'attack': 'rotation',
            'angle': angle,
            'input': quantized.tolist(),
            'metadata': {
                'original': image.tolist(),
                'scale': float(scale),
                'zero_point': float(zero_point),
                'bit_width': self.bit_width
            }
        }

    def quantization_error_amplification(
        self,
        image: np.ndarray,
        amplification_factor: float = 2.0
    ) -> Dict:
        """
        Amplify quantization errors to test robustness

        Args:
            image: Input image
            amplification_factor: Error amplification factor

        Returns:
            Error-amplified test vector
        """
        # Quantize and dequantize
        quantized, scale, zero_point = self.quantize(image)
        dequantized = (quantized - zero_point) * scale

        # Calculate error
        error = image - dequantized

        # Amplify error
        amplified = image + error * (amplification_factor - 1)
        amplified = np.clip(amplified, 0, 1)

        # Re-quantize
        quantized_amp, scale_amp, zero_point_amp = self.quantize(amplified)

        return {
            'type': 'adversarial',
            'attack': 'quantization_error',
            'amplification_factor': amplification_factor,
            'input': quantized_amp.tolist(),
            'metadata': {
                'original': image.tolist(),
                'quantization_error_norm': float(np.linalg.norm(error)),
                'scale': float(scale_amp),
                'zero_point': float(zero_point_amp),
                'bit_width': self.bit_width
            }
        }

    def generate_comprehensive_suite(
        self,
        num_samples: int = 100
    ) -> List[Dict]:
        """
        Generate comprehensive adversarial test suite

        Args:
            num_samples: Number of base samples to generate

        Returns:
            List of adversarial test vectors
        """
        test_vectors = []

        print(f"Generating {num_samples} adversarial test vectors...")

        for i in range(num_samples):
            # Generate random base image
            base_image = np.random.rand(*self.input_shape).astype(np.float32)

            # Apply different attacks
            attacks = [
                ('fgsm_weak', lambda: self.fgsm_attack(base_image, epsilon=0.01)),
                ('fgsm_medium', lambda: self.fgsm_attack(base_image, epsilon=0.05)),
                ('fgsm_strong', lambda: self.fgsm_attack(base_image, epsilon=0.1)),
                ('noise_low', lambda: self.random_noise_attack(base_image, noise_level=0.02)),
                ('noise_high', lambda: self.random_noise_attack(base_image, noise_level=0.1)),
                ('salt_pepper', lambda: self.salt_and_pepper_noise(base_image, density=0.05)),
                ('occlusion_small', lambda: self.occlusion_attack(base_image, box_size=(3, 3))),
                ('occlusion_large', lambda: self.occlusion_attack(base_image, box_size=(7, 7))),
                ('rotation', lambda: self.rotation_perturbation(base_image, angle=15)),
                ('quant_error', lambda: self.quantization_error_amplification(base_image, amplification_factor=2.0))
            ]

            # Select random attack (one per sample)
            attack_name, attack_func = attacks[i % len(attacks)]

            vector = attack_func()
            vector['id'] = f'adversarial_{i:05d}_{attack_name}'

            test_vectors.append(vector)

            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{num_samples} vectors")

        return test_vectors


def main():
    parser = argparse.ArgumentParser(
        description='Generate adversarial test inputs for FPGA AI hardware'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of adversarial examples to generate'
    )
    parser.add_argument(
        '--bit-width',
        type=int,
        default=8,
        choices=[4, 8, 16, 32],
        help='Quantization bit width'
    )
    parser.add_argument(
        '--output',
        default='adversarial_tests.json',
        help='Output file path'
    )
    parser.add_argument(
        '--shape',
        nargs=2,
        type=int,
        default=[28, 28],
        help='Input shape (height width)'
    )

    args = parser.parse_args()

    print("=== Adversarial Test Generator ===")
    print(f"Configuration:")
    print(f"  Count: {args.count}")
    print(f"  Bit Width: {args.bit_width}")
    print(f"  Shape: {args.shape}")
    print(f"  Output: {args.output}")
    print()

    # Create generator
    generator = AdversarialTestGenerator(
        bit_width=args.bit_width,
        input_shape=tuple(args.shape)
    )

    # Generate test suite
    test_vectors = generator.generate_comprehensive_suite(num_samples=args.count)

    # Statistics
    attack_types = {}
    for vector in test_vectors:
        attack = vector.get('attack', 'unknown')
        attack_types[attack] = attack_types.get(attack, 0) + 1

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'bit_width': args.bit_width,
            'input_shape': args.shape,
            'total_vectors': len(test_vectors),
            'attack_distribution': attack_types
        },
        'vectors': test_vectors
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nGenerated {len(test_vectors)} adversarial test vectors")
    print(f"Saved to: {output_path}")
    print("\nAttack Distribution:")
    for attack, count in sorted(attack_types.items()):
        print(f"  - {attack}: {count} vectors")


if __name__ == '__main__':
    main()
