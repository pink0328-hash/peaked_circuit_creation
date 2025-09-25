import sys
import os
import time
import torch
import numpy as np
from typing import Dict, Any, Tuple

# Add lib directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.circuit_gen import CircuitParams, DEVICE
from lib.map_optimizer import MAPOptimizer, BoundaryMPSOptimizer
from lib.optimized_contraction import OptimizedContraction

def create_demo_circuit(n_qubits: int = 40, difficulty: float = 1.0, seed: int = 42) -> Tuple[CircuitParams, Any]:
    """Create a demo circuit for testing."""
    print(f"🔧 Creating {n_qubits}-qubit demo circuit...")
    print(f"   Difficulty: {difficulty}")
    print(f"   Seed: {seed}")
    
    # Create circuit parameters
    circuit_params = CircuitParams.from_difficulty(difficulty)
    
    # Override n_qubits if needed
    if n_qubits != circuit_params.nqubits:
        circuit_params.nqubits = n_qubits
        print(f"   Overriding n_qubits to {n_qubits}")
    
    # Generate the circuit
    start_time = time.time()
    circuit = circuit_params.compute_circuit(seed)
    circuit_time = time.time() - start_time
    
    print(f"Circuit generated in {circuit_time:.2f}s")
    print(f"Target state: {circuit.target_state}")
    print(f"Peak probability: {circuit.peak_prob:.2e}")
    print(f"Number of gates: {len(circuit.gates)}")
    
    return circuit_params, circuit

def create_tensor_network_from_circuit(circuit, n_qubits: int) -> Any:
    """Convert the circuit to a tensor network for optimization."""
    import quimb.tensor as qtn
    
    print(f"Converting circuit to tensor network...")
    
    # Create initial state |0...0⟩
    circuit_tn = qtn.MPS_computational_state("0" * n_qubits).astype_("complex64")
    circuit_tn.apply_to_arrays(lambda x: torch.from_numpy(x).to(device=DEVICE, dtype=torch.complex64))
    
    # Add physical indices
    for k in range(n_qubits):
        circuit_tn[k].modify(left_inds=[f"k{k}"], tags=[f"I{k}", "MPS", "physical"])
    
    # Apply gates from the circuit
    gate_count = 0
    for gate in circuit.gates:
        try:
            # Convert gate to tensor and apply
            if hasattr(gate, 'to_uni'):
                # For SU4 gates
                uni = gate.to_uni()
                circuit_tn.gate_(torch.from_numpy(uni).to(device=DEVICE, dtype=torch.complex64), 
                               (gate.target0, gate.target1), tags=[f"gate_{gate_count}"])
                gate_count += 1
            elif hasattr(gate, 'to_pauli_rots'):
                # For U3 gates, convert to Pauli rotations
                for subgate in gate.to_pauli_rots():
                    if hasattr(subgate, 'to_uni'):
                        uni = subgate.to_uni()
                        circuit_tn.gate_(torch.from_numpy(uni).to(device=DEVICE, dtype=torch.complex64), 
                                       (subgate.target,), tags=[f"gate_{gate_count}"])
                        gate_count += 1
        except Exception as e:
            print(f"Skipping gate {gate_count}: {e}")
            continue
    
    print(f"Applied {gate_count} gates to tensor network")
    print(f"Tensor network has {len(circuit_tn.tensors)} tensors")
    
    return circuit_tn

def demo_map_optimization(circuit_tn, target_state: str, n_qubits: int) -> Dict[str, Any]:
    """Demonstrate MAP optimization."""
    print(f"\n🔍 MAP Optimization Demo ({n_qubits} qubits)")
    print("=" * 60)
    
    # Calculate search space size
    search_space = 2 ** n_qubits
    print(f"Search space: 2^{n_qubits} = {search_space:,} bitstrings")
    print(f"Original method would take ~{search_space / 2000 / 3600:.1f} hours at 2000 it/s")
    
    # Initialize MAP optimizer
    optimizer = MAPOptimizer(
        circuit=circuit_tn,
        target_state=target_state,
        device=DEVICE,
        max_memory_gb=8.0,
        beam_width=64
    )
    
    print(f"Starting MAP optimization...")
    start_time = time.time()
    
    try:
        bitstring, log_prob, stats = optimizer.find_peaked_bitstring(
            max_iterations=10000,
            early_stop_threshold=1e-6
        )
        
        execution_time = time.time() - start_time
        probability = np.exp(log_prob)
        
        print(f"MAP Optimization Complete!")
        print(f"Found bitstring: {bitstring}")
        print(f"Probability: {probability:.2e}")
        print(f"Log probability: {log_prob:.2f}")
        print(f"Execution time: {execution_time:.2f}s")
        print(f"Iterations: {stats.get('iterations', 'N/A')}")
        print(f"Pruned: {stats.get('pruned_count', 'N/A')}")
        print(f"Queue size: {stats.get('queue_size', 'N/A')}")
        
        # Calculate speedup
        original_time_hours = search_space / 2000 / 3600
        speedup = original_time_hours * 3600 / execution_time if execution_time > 0 else float('inf')
        print(f"Speedup: {speedup:.1f}x faster than original method")
        print(f"Time saved: {original_time_hours - execution_time/3600:.1f} hours")
        
        optimizer.cleanup()
        
        return {
            "method": "map_optimization",
            "bitstring": bitstring,
            "probability": probability,
            "log_probability": log_prob,
            "execution_time": execution_time,
            "speedup": speedup,
            "stats": stats,
            "success": True
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ MAP Optimization Failed: {e}")
        optimizer.cleanup()
        
        return {
            "method": "map_optimization",
            "bitstring": "",
            "probability": 0.0,
            "log_probability": float('-inf'),
            "execution_time": execution_time,
            "speedup": 0.0,
            "stats": {},
            "success": False,
            "error": str(e)
        }

def demo_boundary_mps(circuit_tn, target_state: str, n_qubits: int) -> Dict[str, Any]:
    """Demonstrate Boundary-MPS optimization."""
    print(f"\n🔍 Boundary-MPS Demo ({n_qubits} qubits)")
    print("=" * 60)
    
    # Initialize Boundary-MPS optimizer
    optimizer = BoundaryMPSOptimizer(
        circuit=circuit_tn,
        target_state=target_state,
        device=DEVICE
    )
    
    print(f"🚀 Starting Boundary-MPS optimization...")
    start_time = time.time()
    
    try:
        bitstring, probability, stats = optimizer.find_peaked_bitstring(beam_width=64)
        execution_time = time.time() - start_time
        
        print(f"Boundary-MPS Complete!")
        print(f"Found bitstring: {bitstring}")
        print(f"Probability: {probability:.2e}")
        print(f"Execution time: {execution_time:.2f}s")
        print(f"Stats: {stats}")
        
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
        execution_time = time.time() - start_time
        print(f"Boundary-MPS Failed: {e}")
        optimizer.cleanup()
        
        return {
            "method": "boundary_mps",
            "bitstring": "",
            "probability": 0.0,
            "log_probability": float('-inf'),
            "execution_time": execution_time,
            "stats": {},
            "success": False,
            "error": str(e)
        }

def demo_optimized_contraction(circuit_tn, target_state: str, n_qubits: int) -> Dict[str, Any]:
    """Demonstrate optimized contraction."""
    print(f"\nOptimized Contraction Demo ({n_qubits} qubits)")
    print("=" * 60)
    
    # Initialize optimized contraction
    optimizer = OptimizedContraction(
        device=DEVICE,
        max_memory_gb=8.0,
        n_workers=4,
        use_mixed_precision=True
    )
    
    print(f"Starting optimized contraction...")
    start_time = time.time()
    
    try:
        bitstring, probability, stats = optimizer.find_peaked_bitstring_optimized(
            circuit=circuit_tn,
            target_state=target_state,
            batch_size=1000,
            use_parallel=True
        )
        execution_time = time.time() - start_time
        
        print(f"Optimized Contraction Complete!")
        print(f"Found bitstring: {bitstring}")
        print(f"Probability: {probability:.2e}")
        print(f"Execution time: {execution_time:.2f}s")
        print(f"Stats: {stats}")
        
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
        execution_time = time.time() - start_time
        print(f"Optimized Contraction Failed: {e}")
        optimizer.cleanup()
        
        return {
            "method": "optimized_contraction",
            "bitstring": "",
            "probability": 0.0,
            "log_probability": float('-inf'),
            "execution_time": execution_time,
            "stats": {},
            "success": False,
            "error": str(e)
        }

def run_demo(n_qubits: int = 40, difficulty: float = 1.0, seed: int = 42, methods: list = None):
    """Run the complete demo."""
    print("40-Qubit Peak Circuit Optimization Demo")
    print("=" * 80)
    print(f"Target: {n_qubits} qubits")
    print(f"Device: {DEVICE}")
    print(f"Difficulty: {difficulty}")
    print(f"Seed: {seed}")
    print("=" * 80)
    
    if methods is None:
        methods = ["map", "boundary_mps", "optimized"]
    
    # Create demo circuit
    circuit_params, circuit = create_demo_circuit(n_qubits, difficulty, seed)
    
    # Convert to tensor network
    circuit_tn = create_tensor_network_from_circuit(circuit, n_qubits)
    
    # Run optimizations
    results = {}
    
    if "map" in methods:
        results["map"] = demo_map_optimization(circuit_tn, circuit.target_state, n_qubits)
    
    if "boundary_mps" in methods:
        results["boundary_mps"] = demo_boundary_mps(circuit_tn, circuit.target_state, n_qubits)
    
    if "optimized" in methods:
        results["optimized"] = demo_optimized_contraction(circuit_tn, circuit.target_state, n_qubits)
    
    # Print summary
    print(f"\nDEMO SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} | {'Time (s)':<10} | {'Probability':<12} | {'Success':<8}")
    print("-" * 80)
    
    for method, result in results.items():
        if result["success"]:
            print(f"{method:<20} | {result['execution_time']:<10.2f} | {result['probability']:<12.2e} | {'✅':<8}")
        else:
            print(f"{method:<20} | {result['execution_time']:<10.2f} | {'FAILED':<12} | {'❌':<8}")
    
    # Calculate theoretical original time
    search_space = 2 ** n_qubits
    original_time_hours = search_space / 2000 / 3600
    print(f"\nPERFORMANCE COMPARISON")
    print("-" * 80)
    print(f"Original method (theoretical): {original_time_hours:.1f} hours")
    print(f"Search space: 2^{n_qubits} = {search_space:,} bitstrings")
    
    for method, result in results.items():
        if result["success"] and "speedup" in result:
            print(f"{method:<20}: {result['speedup']:.1f}x faster ({result['execution_time']:.2f}s)")
        elif result["success"]:
            speedup = original_time_hours * 3600 / result['execution_time']
            print(f"{method:<20}: {speedup:.1f}x faster ({result['execution_time']:.2f}s)")
    
    return results

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="40-qubit peak circuit optimization demo")
    parser.add_argument("--n_qubits", type=int, default=40, help="Number of qubits")
    parser.add_argument("--difficulty", type=float, default=1.0, help="Difficulty level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--methods", nargs="+", default=["map"], 
                       choices=["map", "boundary_mps", "optimized"],
                       help="Methods to test")
    
    args = parser.parse_args()
    
    # Run demo
    results = run_demo(
        n_qubits=args.n_qubits,
        difficulty=args.difficulty,
        seed=args.seed,
        methods=args.methods
    )
    
    return results

if __name__ == "__main__":
    main()
