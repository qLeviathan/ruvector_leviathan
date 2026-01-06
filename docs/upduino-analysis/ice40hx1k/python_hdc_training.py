#!/usr/bin/env python3
"""
HDC Training Script for iCE40HX1K Implementation

This script generates prototype hypervectors for classification tasks
and outputs them in a format compatible with the FPGA implementation.

Features:
- Generate random basis vectors for encoding
- Train class prototypes from sample data
- Export prototypes as Verilog memory initialization
- Generate test vectors for validation

Author: HDC Design Team
Date: 2026-01-06
"""

import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import json


class HDCTrainer:
    """Binary Hyperdimensional Computing trainer for 1024-bit vectors."""

    def __init__(self, dimension: int = 1024, num_classes: int = 32):
        """
        Initialize HDC trainer.

        Args:
            dimension: Hypervector dimension (must be 1024 for iCE40HX1K)
            num_classes: Number of classes (max 64 for 1024-bit vectors)
        """
        self.dimension = dimension
        self.num_classes = num_classes
        self.basis_vectors = {}
        self.prototypes = np.zeros((num_classes, dimension), dtype=np.int32)

    def generate_basis_vectors(self, num_features: int, seed: int = 42):
        """
        Generate random binary basis vectors for encoding.

        Args:
            num_features: Number of input features
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.basis_vectors = {
            f"feature_{i}": np.random.randint(0, 2, self.dimension)
            for i in range(num_features)
        }
        print(f"Generated {num_features} basis vectors of dimension {self.dimension}")

    def bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Binding operation: XOR."""
        return np.bitwise_xor(v1, v2)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundling operation: Majority vote."""
        # Sum all vectors
        summed = np.sum(vectors, axis=0)
        # Threshold at half the number of vectors
        threshold = len(vectors) / 2
        return (summed >= threshold).astype(np.int32)

    def encode_sample(self, features: Dict[str, float]) -> np.ndarray:
        """
        Encode a sample into a hypervector.

        Args:
            features: Dictionary of feature_name -> value

        Returns:
            1024-bit binary hypervector
        """
        encoded = []

        for feature_name, value in features.items():
            if feature_name not in self.basis_vectors:
                continue

            basis = self.basis_vectors[feature_name]

            # Simple encoding: scale value and bind with basis
            # For binary features, just use basis directly
            if value > 0.5:
                encoded.append(basis)
            else:
                # Negate by flipping all bits
                encoded.append(1 - basis)

        # Bundle all encoded features
        if len(encoded) == 0:
            return np.zeros(self.dimension, dtype=np.int32)
        return self.bundle(encoded)

    def train(self, training_data: Dict[int, List[Dict[str, float]]]):
        """
        Train class prototypes from samples.

        Args:
            training_data: Dict mapping class_id -> list of feature dicts

        Example:
            training_data = {
                0: [{"feature_0": 0.1, "feature_1": 0.9}, ...],
                1: [{"feature_0": 0.8, "feature_1": 0.2}, ...],
            }
        """
        for class_id, samples in training_data.items():
            if class_id >= self.num_classes:
                print(f"Warning: Skipping class {class_id} (exceeds num_classes)")
                continue

            # Encode all samples for this class
            encoded_samples = [self.encode_sample(sample) for sample in samples]

            # Bundle to create prototype
            self.prototypes[class_id] = self.bundle(encoded_samples)

            print(f"Trained class {class_id} with {len(samples)} samples")

    def hamming_distance(self, v1: np.ndarray, v2: np.ndarray) -> int:
        """Compute Hamming distance between two vectors."""
        return np.sum(np.bitwise_xor(v1, v2))

    def classify(self, query: np.ndarray) -> Tuple[int, int]:
        """
        Classify a query vector.

        Args:
            query: 1024-bit query vector

        Returns:
            (class_id, min_distance)
        """
        distances = [
            self.hamming_distance(query, proto)
            for proto in self.prototypes
        ]
        min_idx = np.argmin(distances)
        return min_idx, distances[min_idx]

    def to_verilog_hex(self, output_file: Path):
        """
        Export prototypes as Verilog $readmemh format.

        Args:
            output_file: Path to output .mem file
        """
        with open(output_file, 'w') as f:
            for class_id, prototype in enumerate(self.prototypes):
                # Convert 1024-bit vector to 32 × 32-bit hex words
                for word_idx in range(32):
                    # Extract 32 bits
                    start_bit = word_idx * 32
                    end_bit = start_bit + 32
                    word_bits = prototype[start_bit:end_bit]

                    # Convert to 32-bit integer
                    word_value = 0
                    for i, bit in enumerate(word_bits):
                        word_value |= (int(bit) << i)

                    # Write as 8-digit hex
                    f.write(f"{word_value:08x}\n")

        print(f"Exported prototypes to {output_file}")

    def to_binary_file(self, output_file: Path):
        """
        Export prototypes as raw binary file for FPGA loading.

        Args:
            output_file: Path to output .bin file
        """
        # Pack bits into bytes
        packed = np.packbits(self.prototypes.flatten())
        packed.tofile(output_file)
        print(f"Exported {len(packed)} bytes to {output_file}")

    def generate_test_vectors(self, num_tests: int = 10) -> List[Dict]:
        """
        Generate test vectors for FPGA validation.

        Args:
            num_tests: Number of test cases to generate

        Returns:
            List of test dictionaries with query, expected_class, expected_distance
        """
        tests = []

        for class_id in range(min(num_tests, self.num_classes)):
            # Test 1: Exact match (distance = 0)
            tests.append({
                "query": self.prototypes[class_id].tolist(),
                "expected_class": class_id,
                "expected_distance": 0,
                "description": f"Exact match for class {class_id}"
            })

            # Test 2: Slight perturbation (distance = 1-5)
            perturbed = self.prototypes[class_id].copy()
            num_flips = np.random.randint(1, 6)
            flip_positions = np.random.choice(self.dimension, num_flips, replace=False)
            perturbed[flip_positions] = 1 - perturbed[flip_positions]

            tests.append({
                "query": perturbed.tolist(),
                "expected_class": class_id,
                "expected_distance": num_flips,
                "description": f"Class {class_id} with {num_flips} bit flips"
            })

        return tests


def main():
    """Main training workflow."""
    parser = argparse.ArgumentParser(description="HDC Training for iCE40HX1K")
    parser.add_argument("--dimension", type=int, default=1024, help="Vector dimension")
    parser.add_argument("--classes", type=int, default=32, help="Number of classes")
    parser.add_argument("--features", type=int, default=10, help="Number of features")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Initialize trainer
    trainer = HDCTrainer(dimension=args.dimension, num_classes=args.classes)

    # Generate basis vectors
    trainer.generate_basis_vectors(num_features=args.features, seed=args.seed)

    # Generate synthetic training data (for demonstration)
    print("\nGenerating synthetic training data...")
    training_data = {}

    for class_id in range(args.classes):
        samples = []
        for _ in range(20):  # 20 samples per class
            # Generate random feature values clustered around class-specific mean
            features = {
                f"feature_{i}": np.clip(
                    (class_id / args.classes) + np.random.randn() * 0.1,
                    0, 1
                )
                for i in range(args.features)
            }
            samples.append(features)
        training_data[class_id] = samples

    # Train
    print("\nTraining prototypes...")
    trainer.train(training_data)

    # Export
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\nExporting prototypes...")
    trainer.to_verilog_hex(output_dir / "prototypes.mem")
    trainer.to_binary_file(output_dir / "prototypes.bin")

    # Generate test vectors
    print("\nGenerating test vectors...")
    test_vectors = trainer.generate_test_vectors(num_tests=20)

    with open(output_dir / "test_vectors.json", 'w') as f:
        json.dump(test_vectors, f, indent=2)

    print(f"Exported {len(test_vectors)} test vectors")

    # Statistics
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Dimension: {args.dimension} bits")
    print(f"Classes: {args.classes}")
    print(f"Basis vectors: {args.features}")
    print(f"Memory size: {args.classes * args.dimension} bits = {args.classes * args.dimension / 8192:.1f} KB")
    print(f"\nOutput files in: {output_dir}")
    print("  - prototypes.mem (Verilog $readmemh format)")
    print("  - prototypes.bin (raw binary)")
    print("  - test_vectors.json (test cases)")
    print("\nNext steps:")
    print("  1. Load prototypes.mem into FPGA BRAM")
    print("  2. Run test_vectors.json for validation")
    print("  3. Deploy to UPduino v3.0 hardware")


if __name__ == "__main__":
    main()
