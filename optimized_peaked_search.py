import argparse
import time
import sys
import os
from typing import Dict, Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

import torch
import numpy as np
from lib.circuit_gen import CircuitParams, DEVICE
from lib.map_optimizer import MAPOptimizer, BoundaryMPSOptimizer
from lib.optimized_contraction import OptimizedContraction, MemoryEfficientContraction
from lib.benchmark_comparison import PeakedBitstringBenchmark

def find_peaked_bitstring_map(circuit_params: CircuitParams, seed: int, **kwargs) -> Dict[str, Any]:
    print("🔍 Using MAP optimization with branch-and-bound...")
    
    
    circuit = circuit_params.compute_circuit(seed)
    
    
    n_qubits = circuit_params.nqubits
    target_state = circuit.target_state
    
    
    import quimb.tensor as qtn
    circuit_tn = qtn.MPS_computational_state("0" * n_qubits).astype_("complex64")
    circuit_tn.apply_to_arrays(lambda x: torch.from_numpy(x).to(device=DEVICE, dtype=torch.complex64))
    
    
    optimizer = MAPOptimizer(
        circuit=circuit_tn,
        target_state=target_state,
        device=DEVICE,
        max_memory_gb=kwargs.get('max_memory_gb', 8.0),
        beam_width=kwargs.get('beam_width', 64)
    )
    
    start_time = time.time()
    
    try:
        bitstring, log_prob, stats = optimizer.find_peaked_bitstring(
            max_iterations=kwargs.get('max_iterations', 10000),
            early_stop_threshold=kwargs.get('early_stop_threshold', 1e-6)
        )
        
        execution_time = time.time() - start_time
        probability = np.exp(log_prob)
        
        optimizer.cleanup()
        
        return {
            "method": "map_optimization",
            "bitstring": bitstring,
            "probability": probability,
            "log_probability": log_prob,
            "execution_time": execution_time,
            "stats": stats,
            "success": True
        }
        
    except Exception as e:
        optimizer.cleanup()
        return {
            "method": "map_optimization",
            "bitstring": "",
            "probability": 0.0,
            "log_probability": float('-inf'),
            "execution_time": time.time() - start_time,
            "stats": {},
            "success": False,
            "error": str(e)
        }

def find_peaked_bitstring_boundary_mps(circuit_params: CircuitParams, seed: int, **kwargs) -> Dict[str, Any]:
    """Find peaked bitstring using boundary-MPS approach."""
    print("🔍 Using boundary-MPS approach...")
    
    # Generate circuit
    circuit = circuit_params.compute_circuit(seed)
    n_qubits = circuit_params.nqubits
    target_state = circuit.target_state
    
    # Create tensor network
    import quimb.tensor as qtn
    circuit_tn = qtn.MPS_computational_state("0" * n_qubits).astype_("complex64")
    circuit_tn.apply_to_arrays(lambda x: torch.from_numpy(x).to(device=DEVICE, dtype=torch.complex64))
    
    # Initialize boundary-MPS optimizer
    optimizer = BoundaryMPSOptimizer(
        circuit=circuit_tn,
        target_state=target_state,
        device=DEVICE
    )
    
    start_time = time.time()
    
    try:
        bitstring, probability, stats = optimizer.find_peaked_bitstring(
            beam_width=kwargs.get('beam_width', 64)
        )
        
        execution_time = time.time() - start_time
        
        optimizer.cleanup()
        
        return {
            "method": "boundary_mps",
            "bitstring": bitstring,
            "probability": probability,
            "log_probability": np.log(probability + 1e-12),
            "execution_time": execution_time,
            "stats": stats,
            "success": True
        }
        
    except Exception as e:
        optimizer.cleanup()
        return {
            "method": "boundary_mps",
            "bitstring": "",
            "probability": 0.0,
            "log_probability": float('-inf'),
            "execution_time": time.time() - start_time,
            "stats": {},
            "success": False,
            "error": str(e)
        }

def find_peaked_bitstring_optimized_contraction(circuit_params: CircuitParams, seed: int, **kwargs) -> Dict[str, Any]:
    """Find peaked bitstring using optimized contraction."""
    print("🔍 Using optimized tensor network contraction...")
    
 
    circuit = circuit_params.compute_circuit(seed)
    n_qubits = circuit_params.nqubits
    target_state = circuit.target_state
    
    
    import quimb.tensor as qtn
    circuit_tn = qtn.MPS_computational_state("0" * n_qubits).astype_("complex64")
    circuit_tn.apply_to_arrays(lambda x: torch.from_numpy(x).to(device=DEVICE, dtype=torch.complex64))
    
 
    optimizer = OptimizedContraction(
        device=DEVICE,
        max_memory_gb=kwargs.get('max_memory_gb', 8.0),
        n_workers=kwargs.get('n_workers', 4),
        use_mixed_precision=kwargs.get('use_mixed_precision', True)
    )
    
    start_time = time.time()
    
    try:
        bitstring, probability, stats = optimizer.find_peaked_bitstring_optimized(
            circuit=circuit_tn,
            target_state=target_state,
            batch_size=kwargs.get('batch_size', 1000),
            use_parallel=kwargs.get('use_parallel', True)
        )
        
        execution_time = time.time() - start_time
        
        optimizer.cleanup()
        
        return {
            "method": "optimized_contraction",
            "bitstring": bitstring,
            "probability": probability,
            "log_probability": np.log(probability + 1e-12),
            "execution_time": execution_time,
            "stats": stats,
            "success": True
        }
        
    except Exception as e:
        optimizer.cleanup()
        return {
            "method": "optimized_contraction",
            "bitstring": "",
            "probability": 0.0,
            "log_probability": float('-inf'),
            "execution_time": time.time() - start_time,
            "stats": {},
            "success": False,
            "error": str(e)
        }

def find_peaked_bitstring_memory_efficient(circuit_params: CircuitParams, seed: int, **kwargs) -> Dict[str, Any]:
    """Find peaked bitstring using memory-efficient processing."""
    print("🔍 Using memory-efficient processing...")
    
    
    circuit = circuit_params.compute_circuit(seed)
    n_qubits = circuit_params.nqubits
    target_state = circuit.target_state
    
    
    import quimb.tensor as qtn
    circuit_tn = qtn.MPS_computational_state("0" * n_qubits).astype_("complex64")
    circuit_tn.apply_to_arrays(lambda x: torch.from_numpy(x).to(device=DEVICE, dtype=torch.complex64))
    
    
    optimizer = MemoryEfficientContraction(
        device=DEVICE,
        max_memory_gb=kwargs.get('max_memory_gb', 4.0)
    )
    
    start_time = time.time()
    
    try:
        bitstring, probability, stats = optimizer.find_peaked_bitstring_memory_efficient(
            circuit=circuit_tn,
            target_state=target_state
        )
        
        execution_time = time.time() - start_time
        
        optimizer.cleanup()
        
        return {
            "method": "memory_efficient",
            "bitstring": bitstring,
            "probability": probability,
            "log_probability": np.log(probability + 1e-12),
            "execution_time": execution_time,
            "stats": stats,
            "success": True
        }
        
    except Exception as e:
        optimizer.cleanup()
        return {
            "method": "memory_efficient",
            "bitstring": "",
            "probability": 0.0,
            "log_probability": float('-inf'),
            "execution_time": time.time() - start_time,
            "stats": {},
            "success": False,
            "error": str(e)
        }

def run_benchmark(difficulty: float, seeds: list, methods: list = None):
    """Run comprehensive benchmark."""
    print("🚀 Running comprehensive benchmark...")
    
    if methods is None:
        methods = ["map_optimization", "boundary_mps", "optimized_contraction", "memory_efficient"]
    
    benchmark = PeakedBitstringBenchmark(device=DEVICE)
    
    results = benchmark.run_comprehensive_benchmark(
        difficulty_levels=[difficulty],
        seeds=seeds,
        methods=methods
    )
    
    benchmark.print_summary()
    benchmark.save_results(f"benchmark_difficulty_{difficulty}.json")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Optimized peaked bitstring search")
    parser.add_argument("--method", choices=["map", "boundary_mps", "optimized", "memory", "all", "benchmark"], 
                       default="map", help="Search method to use")
    parser.add_argument("--difficulty", type=float, default=1.0, help="Difficulty level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_memory_gb", type=float, default=8.0, help="Maximum memory usage in GB")
    parser.add_argument("--beam_width", type=int, default=64, help="Beam width for search")
    parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum iterations")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for parallel processing")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark instead of single search")
    
    args = parser.parse_args()
    
    # Set up circuit parameters
    circuit_params = CircuitParams.from_difficulty(args.difficulty)
    
    print(f"🎯 Target: {circuit_params.nqubits} qubits, difficulty {args.difficulty}")
    print(f"🔧 Device: {DEVICE}")
    print(f"💾 Max memory: {args.max_memory_gb} GB")
    
    if args.benchmark or args.method == "benchmark":
        # Run benchmark
        seeds = [42, 123, 456, 789, 999]
        results = run_benchmark(args.difficulty, seeds)
        return
    
    # Run single search
    kwargs = {
        'max_memory_gb': args.max_memory_gb,
        'beam_width': args.beam_width,
        'max_iterations': args.max_iterations,
        'batch_size': args.batch_size,
        'n_workers': args.n_workers
    }
    
    if args.method == "all":
       
        methods = [
            ("map", find_peaked_bitstring_map),
            ("boundary_mps", find_peaked_bitstring_boundary_mps),
            ("optimized", find_peaked_bitstring_optimized_contraction),
            ("memory", find_peaked_bitstring_memory_efficient)
        ]
        
        results = {}
        for method_name, method_func in methods:
            print(f"\n{'='*60}")
            print(f"Running {method_name.upper()} method")
            print('='*60)
            
            result = method_func(circuit_params, args.seed, **kwargs)
            results[method_name] = result
            
            if result["success"]:
                print(f"✅ Success!")
                print(f"   Bitstring: {result['bitstring']}")
                print(f"   Probability: {result['probability']:.2e}")
                print(f"   Time: {result['execution_time']:.2f}s")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
       
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print('='*60)
        
        for method_name, result in results.items():
            if result["success"]:
                print(f"{method_name:15} | {result['probability']:8.2e} | {result['execution_time']:6.2f}s")
            else:
                print(f"{method_name:15} | {'FAILED':>8} | {result['execution_time']:6.2f}s")
    
    else:
        # Run single method
        method_map = {
            "map": find_peaked_bitstring_map,
            "boundary_mps": find_peaked_bitstring_boundary_mps,
            "optimized": find_peaked_bitstring_optimized_contraction,
            "memory": find_peaked_bitstring_memory_efficient
        }
        
        method_func = method_map[args.method]
        result = method_func(circuit_params, args.seed, **kwargs)
        
        if result["success"]:
            print(f"\n✅ Success!")
            print(f"   Method: {result['method']}")
            print(f"   Bitstring: {result['bitstring']}")
            print(f"   Probability: {result['probability']:.2e}")
            print(f"   Log Probability: {result['log_probability']:.2f}")
            print(f"   Execution Time: {result['execution_time']:.2f}s")
            
            if 'stats' in result and result['stats']:
                print(f"   Additional Stats: {result['stats']}")
        else:
            print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")
            print(f"   Execution Time: {result['execution_time']:.2f}s")

if __name__ == "__main__":
    main()
