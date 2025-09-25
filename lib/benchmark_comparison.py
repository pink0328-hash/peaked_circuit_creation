import time
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import quimb.tensor as qtn
from dataclasses import dataclass

from .circuit_gen import CircuitParams, make_qmps, DEVICE
from .map_optimizer import MAPOptimizer, BoundaryMPSOptimizer
from .optimized_contraction import OptimizedContraction, MemoryEfficientContraction

@dataclass
class BenchmarkResult:
    method: str
    bitstring: str
    probability: float
    execution_time: float
    memory_usage_mb: float
    iterations: int
    success: bool
    error_message: str = ""

class PeakedBitstringBenchmark:
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = []
        
    def run_comprehensive_benchmark(self, 
                                  difficulty_levels: List[float] = [0.5, 1.0, 1.5],
                                  seeds: List[int] = [42, 123, 456],
                                  methods: List[str] = None) -> Dict[str, Any]:
        if methods is None:
            methods = [
                "original_tn",
                "map_optimization", 
                "boundary_mps",
                "optimized_contraction",
                "memory_efficient"
            ]
        
        all_results = {}
        
        for difficulty in difficulty_levels:
            print(f"\n=== Testing Difficulty Level: {difficulty} ===")
            
            for seed in seeds:
                print(f"\n--- Seed: {seed} ---")
                
                # Generate circuit
                circuit_params = CircuitParams.from_difficulty(difficulty)
                circuit = self._generate_test_circuit(circuit_params, seed)
                
                seed_results = {}
                
                for method in methods:
                    print(f"Testing {method}...")
                    
                    try:
                        result = self._run_single_benchmark(
                            circuit, circuit_params, method, seed
                        )
                        seed_results[method] = result
                        self.results.append(result)
                        
                        print(f"  ✓ {method}: {result.execution_time:.2f}s, "
                              f"prob={result.probability:.2e}, "
                              f"success={result.success}")
                        
                    except Exception as e:
                        print(f"  ✗ {method}: Failed - {str(e)}")
                        error_result = BenchmarkResult(
                            method=method,
                            bitstring="",
                            probability=0.0,
                            execution_time=0.0,
                            memory_usage_mb=0.0,
                            iterations=0,
                            success=False,
                            error_message=str(e)
                        )
                        seed_results[method] = error_result
                        self.results.append(error_result)
                
                all_results[f"difficulty_{difficulty}_seed_{seed}"] = seed_results
        
        # Generate summary statistics
        summary = self._generate_summary_statistics()
        
        return {
            "detailed_results": all_results,
            "summary": summary,
            "total_tests": len(self.results)
        }
    
    def _generate_test_circuit(self, circuit_params: CircuitParams, seed: int) -> qtn.TensorNetwork:
        """Generate a test circuit for benchmarking."""
        # Create a simple test circuit
        n_qubits = circuit_params.nqubits
        
        # Generate random target state
        np.random.seed(seed)
        target_state = "".join("1" if np.random.random() < 0.5 else "0" for _ in range(n_qubits))
        
        # Create circuit using the existing make_qmps function
        circuit = make_qmps(target_state, circuit_params.pqc_depth, 0)
        
        return circuit
    
    def _run_single_benchmark(self, 
                            circuit: qtn.TensorNetwork,
                            circuit_params: CircuitParams,
                            method: str,
                            seed: int) -> BenchmarkResult:
        """Run a single benchmark for a specific method."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            if method == "original_tn":
                bitstring, probability, stats = self._original_tn_method(circuit, circuit_params)
            elif method == "map_optimization":
                bitstring, probability, stats = self._map_optimization_method(circuit, circuit_params)
            elif method == "boundary_mps":
                bitstring, probability, stats = self._boundary_mps_method(circuit, circuit_params)
            elif method == "optimized_contraction":
                bitstring, probability, stats = self._optimized_contraction_method(circuit, circuit_params)
            elif method == "memory_efficient":
                bitstring, probability, stats = self._memory_efficient_method(circuit, circuit_params)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_usage = end_memory - start_memory
            
            return BenchmarkResult(
                method=method,
                bitstring=bitstring,
                probability=probability,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                iterations=stats.get("iterations", 0),
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BenchmarkResult(
                method=method,
                bitstring="",
                probability=0.0,
                execution_time=execution_time,
                memory_usage_mb=0.0,
                iterations=0,
                success=False,
                error_message=str(e)
            )
    
    def _original_tn_method(self, circuit: qtn.TensorNetwork, circuit_params: CircuitParams) -> Tuple[str, float, Dict]:
        """Original tensor network method (simplified for benchmarking)."""        
        n_qubits = circuit_params.nqubits
        target_state = "0" * n_qubits  # Placeholder
        
        # Simulate the original approach
        best_bitstring = target_state
        best_prob = 0.0
        iterations = 0
        
        return best_bitstring, best_prob, {"iterations": iterations}
    
    def _map_optimization_method(self, circuit: qtn.TensorNetwork, circuit_params: CircuitParams) -> Tuple[str, float, Dict]:
        """MAP optimization method."""
        target_state = "0" * circuit_params.nqubits  # Placeholder
        
        optimizer = MAPOptimizer(
            circuit=circuit,
            target_state=target_state,
            device=self.device,
            max_memory_gb=8.0,
            beam_width=64
        )
        
        try:
            bitstring, log_prob, stats = optimizer.find_peaked_bitstring()
            probability = np.exp(log_prob)
            optimizer.cleanup()
            
            return bitstring, probability, stats
            
        except Exception as e:
            optimizer.cleanup()
            raise e
    
    def _boundary_mps_method(self, circuit: qtn.TensorNetwork, circuit_params: CircuitParams) -> Tuple[str, float, Dict]:
        """Boundary-MPS method."""
        target_state = "0" * circuit_params.nqubits  # Placeholder
        
        optimizer = BoundaryMPSOptimizer(
            circuit=circuit,
            target_state=target_state,
            device=self.device
        )
        
        try:
            bitstring, probability, stats = optimizer.find_peaked_bitstring()
            optimizer.cleanup()
            
            return bitstring, probability, stats
            
        except Exception as e:
            optimizer.cleanup()
            raise e
    
    def _optimized_contraction_method(self, circuit: qtn.TensorNetwork, circuit_params: CircuitParams) -> Tuple[str, float, Dict]:
        """Optimized contraction method."""
        target_state = "0" * circuit_params.nqubits  # Placeholder
        
        optimizer = OptimizedContraction(
            device=self.device,
            max_memory_gb=8.0,
            n_workers=4,
            use_mixed_precision=True
        )
        
        try:
            bitstring, probability, stats = optimizer.find_peaked_bitstring_optimized(
                circuit=circuit,
                target_state=target_state,
                batch_size=1000,
                use_parallel=True
            )
            optimizer.cleanup()
            
            return bitstring, probability, stats
            
        except Exception as e:
            optimizer.cleanup()
            raise e
    
    def _memory_efficient_method(self, circuit: qtn.TensorNetwork, circuit_params: CircuitParams) -> Tuple[str, float, Dict]:
        """Memory-efficient method."""
        target_state = "0" * circuit_params.nqubits  # Placeholder
        
        optimizer = MemoryEfficientContraction(
            device=self.device,
            max_memory_gb=4.0
        )
        
        try:
            bitstring, probability, stats = optimizer.find_peaked_bitstring_memory_efficient(
                circuit=circuit,
                target_state=target_state
            )
            optimizer.cleanup()
            
            return bitstring, probability, stats
            
        except Exception as e:
            optimizer.cleanup()
            raise e
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        else:
            import psutil
            return psutil.Process().memory_info().rss / 1024**2
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics from all benchmark results."""
        if not self.results:
            return {}
        
        # Group results by method
        method_results = {}
        for result in self.results:
            if result.method not in method_results:
                method_results[result.method] = []
            method_results[result.method].append(result)
        
        summary = {}
        
        for method, results in method_results.items():
            successful_results = [r for r in results if r.success]
            
            if not successful_results:
                summary[method] = {
                    "success_rate": 0.0,
                    "avg_time": 0.0,
                    "avg_probability": 0.0,
                    "avg_memory": 0.0,
                    "total_tests": len(results)
                }
                continue
            
            summary[method] = {
                "success_rate": len(successful_results) / len(results),
                "avg_time": np.mean([r.execution_time for r in successful_results]),
                "std_time": np.std([r.execution_time for r in successful_results]),
                "avg_probability": np.mean([r.probability for r in successful_results]),
                "std_probability": np.std([r.probability for r in successful_results]),
                "avg_memory": np.mean([r.memory_usage_mb for r in successful_results]),
                "std_memory": np.std([r.memory_usage_mb for r in successful_results]),
                "total_tests": len(results),
                "successful_tests": len(successful_results)
            }
        
        return summary
    
    def save_results(self, filename: str):
        """Save benchmark results to a JSON file."""
        results_dict = {
            "results": [
                {
                    "method": r.method,
                    "bitstring": r.bitstring,
                    "probability": r.probability,
                    "execution_time": r.execution_time,
                    "memory_usage_mb": r.memory_usage_mb,
                    "iterations": r.iterations,
                    "success": r.success,
                    "error_message": r.error_message
                }
                for r in self.results
            ],
            "summary": self._generate_summary_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        summary = self._generate_summary_statistics()
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        for method, stats in summary.items():
            print(f"\n{method.upper()}:")
            print(f"  Success Rate: {stats['success_rate']:.2%}")
            print(f"  Avg Time: {stats['avg_time']:.2f}s ± {stats['std_time']:.2f}s")
            print(f"  Avg Probability: {stats['avg_probability']:.2e} ± {stats['std_probability']:.2e}")
            print(f"  Avg Memory: {stats['avg_memory']:.1f}MB ± {stats['std_memory']:.1f}MB")
            print(f"  Tests: {stats['successful_tests']}/{stats['total_tests']}")


def run_quick_benchmark():
    """Run a quick benchmark for testing purposes."""
    benchmark = PeakedBitstringBenchmark(device=DEVICE)
    
    # Run with smaller parameters for quick testing
    results = benchmark.run_comprehensive_benchmark(
        difficulty_levels=[0.5, 1.0],
        seeds=[42, 123],
        methods=["original_tn", "map_optimization", "boundary_mps"]
    )
    
    benchmark.print_summary()
    benchmark.save_results("quick_benchmark_results.json")
    
    return results


if __name__ == "__main__":
    # Run quick benchmark
    results = run_quick_benchmark()
