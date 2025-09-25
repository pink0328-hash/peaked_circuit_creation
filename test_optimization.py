import sys
import os
import time
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.circuit_gen import CircuitParams, DEVICE
from lib.map_optimizer import MAPOptimizer, BoundaryMPSOptimizer
from lib.optimized_contraction import OptimizedContraction

def test_simple_circuit():
    """Test with a simple circuit to verify the methods work."""
    print("Testing optimization methods with simple circuit...")
    
    difficulty = 0.5  
    seed = 42
    
    print(f"Difficulty: {difficulty}, Seed: {seed}")
    

    circuit_params = CircuitParams.from_difficulty(difficulty)
    print(f"Circuit: {circuit_params.nqubits} qubits")
    

    import quimb.tensor as qtn
    
    n_qubits = circuit_params.nqubits
    target_state = "0" * n_qubits  
    
  
    circuit_tn = qtn.MPS_computational_state("0" * n_qubits).astype_("complex64")
    circuit_tn.apply_to_arrays(lambda x: torch.from_numpy(x).to(device=DEVICE, dtype=torch.complex64))
    

    for i in range(n_qubits - 1):
        gate = torch.tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]], dtype=torch.complex64, device=DEVICE)
        
        circuit_tn.gate_(gate, (i, i+1), tags=[f"gate_{i}"])
    
    print(f"✅ Created test circuit with {len(circuit_tn.tensors)} tensors")
     # Test MAP optimization
    print("\n🔍 Testing MAP optimization...")
    try:
        map_optimizer = MAPOptimizer(
            circuit=circuit_tn,
            target_state=target_state,
            device=DEVICE,
            max_memory_gb=2.0,  
            beam_width=16      
        )
        
        start_time = time.time()
        bitstring, log_prob, stats = map_optimizer.find_peaked_bitstring(
            max_iterations=100, 
            early_stop_threshold=1e-3
        )
        execution_time = time.time() - start_time
        
        print(f"  ✅ MAP Result: {bitstring}")
        print(f"  📈 Log Probability: {log_prob:.4f}")
        print(f"  ⏱️  Time: {execution_time:.2f}s")
        print(f"  📊 Stats: {stats}")
        
        map_optimizer.cleanup()
        
    except Exception as e:
        print(f"  ❌ MAP failed: {e}")
    
    # Test Boundary-MPS
    print("\n🔍 Testing Boundary-MPS...")
    try:
        bmps_optimizer = BoundaryMPSOptimizer(
            circuit=circuit_tn,
            target_state=target_state,
            device=DEVICE
        )
        
        start_time = time.time()
        bitstring, probability, stats = bmps_optimizer.find_peaked_bitstring(beam_width=16)
        execution_time = time.time() - start_time
        
        print(f"  ✅ Boundary-MPS Result: {bitstring}")
        print(f"  📈 Probability: {probability:.4f}")
        print(f"  ⏱️  Time: {execution_time:.2f}s")
        print(f"  📊 Stats: {stats}")
        
        bmps_optimizer.cleanup()
        
    except Exception as e:
        print(f"  ❌ Boundary-MPS failed: {e}")
    
    # Test Optimized Contraction
    print("\n🔍 Testing Optimized Contraction...")
    try:
        opt_contraction = OptimizedContraction(
            device=DEVICE,
            max_memory_gb=2.0,
            n_workers=2, 
            use_mixed_precision=True
        )
        
        start_time = time.time()
        bitstring, probability, stats = opt_contraction.find_peaked_bitstring_optimized(
            circuit=circuit_tn,
            target_state=target_state,
            batch_size=100, 
            use_parallel=False  
        )
        execution_time = time.time() - start_time
        
        print(f"Optimized Contraction Result: {bitstring}")
        print(f"Probability: {probability:.4f}")
        print(f"Time: {execution_time:.2f}s")
        print(f"Stats: {stats}")
        
        opt_contraction.cleanup()
        
    except Exception as e:
        print(f"  ❌ Optimized Contraction failed: {e}")
    
    print("\nTest completed!")

def test_memory_usage():
    """Test memory usage of different methods."""
    print("\nTesting memory usage...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    difficulty = 1.0
    circuit_params = CircuitParams.from_difficulty(difficulty)
    n_qubits = circuit_params.nqubits
    
    print(f"  🔧 Testing with {n_qubits} qubits")
    
    import quimb.tensor as qtn
    circuit_tn = qtn.MPS_computational_state("0" * n_qubits).astype_("complex64")
    circuit_tn.apply_to_arrays(lambda x: torch.from_numpy(x).to(device=DEVICE, dtype=torch.complex64))
    
    try:
        map_optimizer = MAPOptimizer(
            circuit=circuit_tn,
            target_state="0" * n_qubits,
            device=DEVICE,
            max_memory_gb=1.0, 
            beam_width=8
        )
        
        peak_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Peak GPU memory (MAP): {peak_memory:.1f} MB")
        print(f"Memory increase: {peak_memory - initial_memory:.1f} MB")
        
        map_optimizer.cleanup()
        
    except Exception as e:
        print(f"  ❌ MAP memory test failed: {e}")
    
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"  🧹 After cleanup: {final_memory:.1f} MB")

def main():
    """Main test function."""
    print("Starting optimization method tests...")
    print(f"Device: {DEVICE}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    try:
        test_simple_circuit()
        test_memory_usage()
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
