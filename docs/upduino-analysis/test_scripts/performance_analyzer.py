#!/usr/bin/env python3
"""
Performance Analysis Tool for FPGA AI Hardware Testing

Analyzes test results and generates comprehensive performance reports
including latency, throughput, power, accuracy, and resource efficiency metrics.
"""

import json
import sys
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class PerformanceAnalyzer:
    """Analyze FPGA AI hardware performance metrics"""

    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = self._load_results()

    def _load_results(self) -> Dict:
        """Load test results from JSON file"""
        with open(self.results_file) as f:
            return json.load(f)

    def analyze_latency(self) -> Dict:
        """Analyze inference latency metrics"""
        latencies = self.results.get('latencies_ms', [])

        if not latencies:
            return {'status': 'no_data'}

        latencies = np.array(latencies)

        analysis = {
            'mean_ms': float(np.mean(latencies)),
            'median_ms': float(np.median(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'coefficient_of_variation': float(np.std(latencies) / np.mean(latencies)),

            # Jitter analysis
            'jitter_ms': float(np.max(latencies) - np.min(latencies)),
            'jitter_pct': float((np.max(latencies) - np.min(latencies)) / np.mean(latencies) * 100),

            # Outlier detection (beyond 3 std deviations)
            'outliers': int(np.sum(np.abs(latencies - np.mean(latencies)) > 3 * np.std(latencies))),
            'outlier_pct': float(np.sum(np.abs(latencies - np.mean(latencies)) > 3 * np.std(latencies)) / len(latencies) * 100),

            # Target compliance (< 100ms target)
            'under_100ms_pct': float(np.sum(latencies < 100) / len(latencies) * 100),
            'target_met': bool(np.percentile(latencies, 95) < 100)
        }

        return analysis

    def analyze_throughput(self) -> Dict:
        """Analyze throughput metrics"""
        mean_latency = self.results.get('mean_latency_ms', 0)

        if mean_latency == 0:
            return {'status': 'no_data'}

        throughput_fps = 1000 / mean_latency

        analysis = {
            'fps': float(throughput_fps),
            'inferences_per_minute': float(throughput_fps * 60),
            'inferences_per_hour': float(throughput_fps * 3600),

            # Target compliance (> 10 fps target)
            'target_met': bool(throughput_fps > 10),
            'target_fps': 10,
            'fps_vs_target_pct': float(throughput_fps / 10 * 100)
        }

        return analysis

    def analyze_accuracy(self) -> Dict:
        """Analyze accuracy metrics"""
        total = self.results.get('total_tests', 0)
        passed = self.results.get('passed', 0)

        if total == 0:
            return {'status': 'no_data'}

        accuracy = passed / total

        analysis = {
            'accuracy': float(accuracy),
            'accuracy_pct': float(accuracy * 100),
            'total_tests': int(total),
            'passed': int(passed),
            'failed': int(self.results.get('failed', 0)),

            # Target compliance (> 90% accuracy target)
            'target_met': bool(accuracy > 0.90),
            'target_accuracy': 0.90,
            'accuracy_vs_target_pct': float(accuracy / 0.90 * 100),

            # Error rate
            'error_rate': float(1 - accuracy),
            'error_rate_pct': float((1 - accuracy) * 100)
        }

        return analysis

    def analyze_power(self) -> Dict:
        """Analyze power consumption metrics"""
        power_mw = self.results.get('power_mw', None)
        mean_latency_ms = self.results.get('mean_latency_ms', 0)

        if power_mw is None:
            return {
                'status': 'not_measured',
                'note': 'Power measurement requires external hardware'
            }

        energy_per_inference_mj = power_mw * (mean_latency_ms / 1000)

        analysis = {
            'average_power_mw': float(power_mw),
            'average_power_w': float(power_mw / 1000),
            'energy_per_inference_mj': float(energy_per_inference_mj),
            'energy_per_inference_uj': float(energy_per_inference_mj * 1000),

            # Battery life estimation (assuming 1000 mAh @ 3.7V battery)
            'battery_capacity_mwh': 1000 * 3.7,  # 3700 mWh
            'estimated_inferences_per_charge': int(3700 / energy_per_inference_mj) if energy_per_inference_mj > 0 else 0,
            'estimated_hours_per_charge': float(3700 / power_mw) if power_mw > 0 else 0,

            # Target compliance (< 50 mW target)
            'target_met': bool(power_mw < 50),
            'target_power_mw': 50,
            'power_vs_target_pct': float(power_mw / 50 * 100)
        }

        return analysis

    def analyze_resource_efficiency(self) -> Dict:
        """Analyze resource utilization efficiency"""
        # This would require synthesis/PnR reports
        # For now, return placeholder

        return {
            'status': 'requires_synthesis_report',
            'note': 'Resource efficiency requires synthesis log and timing reports',
            'metrics_to_analyze': [
                'LUT utilization',
                'BRAM utilization',
                'SPRAM utilization',
                'DSP blocks utilization',
                'TOPS per LUT',
                'GOPS per LUT',
                'TOPS per Watt'
            ]
        }

    def generate_summary(self) -> Dict:
        """Generate comprehensive performance summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'test_session': self.results.get('session_id', 'unknown'),
            'model': self.results.get('model', 'unknown'),
            'quantization': self.results.get('quantization', 'unknown'),

            'latency': self.analyze_latency(),
            'throughput': self.analyze_throughput(),
            'accuracy': self.analyze_accuracy(),
            'power': self.analyze_power(),
            'resource_efficiency': self.analyze_resource_efficiency()
        }

        # Overall health score (0-100)
        health_score = self._calculate_health_score(summary)
        summary['health_score'] = health_score

        return summary

    def _calculate_health_score(self, summary: Dict) -> Dict:
        """Calculate overall system health score"""
        scores = []

        # Latency score (target: < 100ms p95)
        latency = summary['latency']
        if latency.get('status') != 'no_data':
            if latency.get('target_met', False):
                scores.append(100)
            else:
                # Penalize linearly
                p95 = latency.get('p95_ms', 1000)
                scores.append(max(0, 100 - (p95 - 100)))

        # Throughput score (target: > 10 fps)
        throughput = summary['throughput']
        if throughput.get('status') != 'no_data':
            if throughput.get('target_met', False):
                scores.append(100)
            else:
                fps = throughput.get('fps', 0)
                scores.append(min(100, fps / 10 * 100))

        # Accuracy score (target: > 90%)
        accuracy = summary['accuracy']
        if accuracy.get('status') != 'no_data':
            if accuracy.get('target_met', False):
                scores.append(100)
            else:
                acc = accuracy.get('accuracy', 0)
                scores.append(min(100, acc / 0.90 * 100))

        # Power score (target: < 50 mW)
        power = summary['power']
        if power.get('status') != 'not_measured':
            if power.get('target_met', False):
                scores.append(100)
            else:
                power_mw = power.get('average_power_mw', 1000)
                scores.append(max(0, 100 - (power_mw - 50)))

        if not scores:
            return {'overall_score': 0, 'status': 'insufficient_data'}

        overall = np.mean(scores)

        return {
            'overall_score': float(overall),
            'grade': self._score_to_grade(overall),
            'component_scores': {
                'latency': scores[0] if len(scores) > 0 else None,
                'throughput': scores[1] if len(scores) > 1 else None,
                'accuracy': scores[2] if len(scores) > 2 else None,
                'power': scores[3] if len(scores) > 3 else None
            }
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    def generate_markdown_report(self, output_file: str):
        """Generate markdown performance report"""
        summary = self.generate_summary()

        report = []
        report.append("# FPGA AI Hardware Performance Analysis Report")
        report.append(f"\n**Generated:** {summary['timestamp']}")
        report.append(f"**Session:** {summary.get('test_session', 'N/A')}")
        report.append(f"**Model:** {summary.get('model', 'N/A')}")
        report.append(f"**Quantization:** {summary.get('quantization', 'N/A')}")

        # Health score
        health = summary['health_score']
        report.append(f"\n## Overall Health Score: {health['overall_score']:.1f}/100 (Grade: {health['grade']})\n")

        # Latency analysis
        report.append("## Latency Analysis\n")
        latency = summary['latency']
        if latency.get('status') != 'no_data':
            report.append(f"- **Mean Latency:** {latency['mean_ms']:.2f} ms")
            report.append(f"- **Median Latency:** {latency['median_ms']:.2f} ms")
            report.append(f"- **P95 Latency:** {latency['p95_ms']:.2f} ms")
            report.append(f"- **P99 Latency:** {latency['p99_ms']:.2f} ms")
            report.append(f"- **Jitter:** {latency['jitter_ms']:.2f} ms ({latency['jitter_pct']:.1f}%)")
            report.append(f"- **Target Met (<100ms):** {'✅ Yes' if latency['target_met'] else '❌ No'}")
        else:
            report.append("*No latency data available*")

        # Throughput analysis
        report.append("\n## Throughput Analysis\n")
        throughput = summary['throughput']
        if throughput.get('status') != 'no_data':
            report.append(f"- **Throughput:** {throughput['fps']:.2f} fps")
            report.append(f"- **Inferences/Minute:** {throughput['inferences_per_minute']:.0f}")
            report.append(f"- **Inferences/Hour:** {throughput['inferences_per_hour']:.0f}")
            report.append(f"- **Target Met (>10 fps):** {'✅ Yes' if throughput['target_met'] else '❌ No'}")
        else:
            report.append("*No throughput data available*")

        # Accuracy analysis
        report.append("\n## Accuracy Analysis\n")
        accuracy = summary['accuracy']
        if accuracy.get('status') != 'no_data':
            report.append(f"- **Accuracy:** {accuracy['accuracy_pct']:.2f}%")
            report.append(f"- **Total Tests:** {accuracy['total_tests']}")
            report.append(f"- **Passed:** {accuracy['passed']}")
            report.append(f"- **Failed:** {accuracy['failed']}")
            report.append(f"- **Error Rate:** {accuracy['error_rate_pct']:.2f}%")
            report.append(f"- **Target Met (>90%):** {'✅ Yes' if accuracy['target_met'] else '❌ No'}")
        else:
            report.append("*No accuracy data available*")

        # Power analysis
        report.append("\n## Power Analysis\n")
        power = summary['power']
        if power.get('status') != 'not_measured':
            report.append(f"- **Average Power:** {power['average_power_mw']:.2f} mW")
            report.append(f"- **Energy/Inference:** {power['energy_per_inference_uj']:.2f} µJ")
            report.append(f"- **Est. Battery Life:** {power['estimated_hours_per_charge']:.1f} hours (1000mAh @ 3.7V)")
            report.append(f"- **Target Met (<50mW):** {'✅ Yes' if power['target_met'] else '❌ No'}")
        else:
            report.append("*Power measurement not available (requires external measurement hardware)*")

        # Recommendations
        report.append("\n## Recommendations\n")
        recommendations = self._generate_recommendations(summary)
        for rec in recommendations:
            report.append(f"- {rec}")

        # Save report
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))

        print(f"Performance report generated: {output_path}")
        return '\n'.join(report)

    def _generate_recommendations(self, summary: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # Latency recommendations
        latency = summary['latency']
        if latency.get('status') != 'no_data':
            if not latency.get('target_met', True):
                recommendations.append(
                    f"⚠️ **Latency issue**: P95 latency ({latency['p95_ms']:.2f}ms) exceeds 100ms target. "
                    "Consider pipeline optimization or increasing clock frequency."
                )
            if latency.get('jitter_pct', 0) > 50:
                recommendations.append(
                    f"⚠️ **High jitter**: {latency['jitter_pct']:.1f}% jitter detected. "
                    "Investigate timing variations and pipeline stalls."
                )

        # Accuracy recommendations
        accuracy = summary['accuracy']
        if accuracy.get('status') != 'no_data':
            if not accuracy.get('target_met', True):
                acc = accuracy.get('accuracy_pct', 0)
                if acc < 70:
                    recommendations.append(
                        f"🚨 **Critical accuracy issue**: {acc:.1f}% accuracy is very low. "
                        "Verify model implementation and quantization settings."
                    )
                elif acc < 85:
                    recommendations.append(
                        f"⚠️ **Low accuracy**: {acc:.1f}% accuracy below target. "
                        "Consider increasing bit-width or reviewing quantization strategy."
                    )

        # Power recommendations
        power = summary['power']
        if power.get('status') != 'not_measured':
            if not power.get('target_met', True):
                recommendations.append(
                    f"⚠️ **High power consumption**: {power['average_power_mw']:.2f}mW exceeds 50mW target. "
                    "Review clock gating, power islands, and dynamic voltage scaling."
                )

        # Overall health recommendations
        health = summary['health_score']
        if health['overall_score'] < 70:
            recommendations.append(
                f"🚨 **Overall health low**: Score {health['overall_score']:.1f}/100 (Grade {health['grade']}). "
                "Review all metrics and prioritize optimization."
            )

        if not recommendations:
            recommendations.append("✅ **All metrics within target ranges** - System performing well!")

        return recommendations


def main():
    parser = argparse.ArgumentParser(
        description='Analyze FPGA AI hardware performance metrics'
    )
    parser.add_argument(
        '--results',
        required=True,
        help='Path to test results JSON file'
    )
    parser.add_argument(
        '--output',
        default='performance_report.md',
        help='Output report file path'
    )
    parser.add_argument(
        '--json-output',
        help='Optional JSON output file for structured results'
    )

    args = parser.parse_args()

    print("=== FPGA Performance Analyzer ===")
    print(f"Results file: {args.results}")
    print(f"Output file: {args.output}")
    print()

    # Analyze performance
    analyzer = PerformanceAnalyzer(args.results)

    # Generate reports
    report_text = analyzer.generate_markdown_report(args.output)
    summary = analyzer.generate_summary()

    # Print summary to console
    print(report_text)

    # Save JSON if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nJSON summary saved to: {args.json_output}")


if __name__ == '__main__':
    main()
