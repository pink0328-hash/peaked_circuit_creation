import sys
import os
import time
import torch
import numpy as np

# Add lib directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

def quick_demo():
    """Run a quick demo with a simple circuit."""
    print("🚀 Quick 40-Qubit Peak Circuit Demo")
    print("=" * 50)
    
    # Check if we can import the required modules
    try:
        from lib.circuit_gen import CircuitParams, DEVICE
        from lib.map_optimizer import MAPOptimizer
        import quimb.tensor as qtn
        print(f"✅ All modules imported successfully")
        print(f"🔧 Device: {DEVICE}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r peaked_circuits/requirements.txt")
        print("pip install cotengra optuna")
        return
    
    # Create a simple test circuit
    n_qubits = 40
    print(f"\n🔧 Creating {n_qubits}-qubit test circuit...")
    
    # Create a simple MPS circuit
    circuit_tn = qtn.MPS_computational_state("0" * n_qubits).astype_("complex64")
    circuit_tn.apply_to_arrays(lambda x: torch.from_numpy(x).to(device=DEVICE, dtype=torch.complex64))
    
    # Add physical indices
    for k in range(n_qubits):
        circuit_tn[k].modify(left_inds=[f"k{k}"], tags=[f"I{k}", "MPS", "physical"])
    
    # Add some simple gates to create a peaked circuit
    print("   Adding gates to create peaked circuit...")
    gate_count = 0
    
    # Add some random gates to create structure
    np.random.seed(42)
    for i in range(min(20, n_qubits - 1)):  # Limit gates for demo
        # Create a simple 2-qubit gate
        gate = torch.randn(4, 4, dtype=torch.complex64, device=DEVICE)
        gate = gate + gate.conj().T  # Make it Hermitian
        gate = torch.exp(1j * gate)  # Make it unitary-ish
        
        # Apply gate
        qubit1 = i
        qubit2 = (i + 1) % n_qubits
        circuit_tn.gate_(gate, (qubit1, qubit2), tags=[f"gate_{gate_count}"])
        gate_count += 1
    
    print(f"   ✅ Added {gate_count} gates")
    
    # Create a target state (random for demo)
    target_state = "".join("1" if np.random.random() < 0.5 else "0" for _ in range(n_qubits))
    print(f"   🎯 Target state: {target_state}")
    
    # Calculate search space
    search_space = 2 ** n_qubits
    print(f"\n📊 Search Space Analysis")
    print(f"   Total bitstrings: 2^{n_qubits} = {search_space:,}")
    print(f"   Original method time: ~{search_space / 2000 / 3600:.1f} hours at 2000 it/s")
    
    # Test MAP optimization
    print(f"\n🔍 Testing MAP Optimization...")
    print("-" * 50)
    
    try:
        optimizer = MAPOptimizer(
            circuit=circuit_tn,
            target_state=target_state,
            device=DEVICE,
            max_memory_gb=4.0,  # Conservative for demo
            beam_width=32       # Smaller for demo
        )
        
        start_time = time.time()
        bitstring, log_prob, stats = optimizer.find_peaked_bitstring(
            max_iterations=1000,  # Small number for demo
            early_stop_threshold=1e-3
        )
        execution_time = time.time() - start_time
        
        print(f"✅ MAP Optimization Complete!")
        print(f"   🎯 Found bitstring: {bitstring}")
        print(f"   📈 Probability: {np.exp(log_prob):.2e}")
        print(f"   ⏱️  Time: {execution_time:.2f}s")
        print(f"   🔄 Iterations: {stats.get('iterations', 'N/A')}")
        print(f"   ✂️  Pruned: {stats.get('pruned_count', 'N/A')}")
        
        # Calculate speedup
        original_time_hours = search_space / 2000 / 3600
        speedup = original_time_hours * 3600 / execution_time if execution_time > 0 else float('inf')
        print(f"   🚀 Speedup: {speedup:.1f}x faster than original")
        print(f"   ⏰ Time saved: {original_time_hours - execution_time/3600:.1f} hours")
        
        optimizer.cleanup()
        
    except Exception as e:
        print(f"❌ MAP Optimization Failed: {e}")
        print("This might be due to:")
        print("  - Insufficient GPU memory")
        print("  - Circuit complexity")
        print("  - Missing dependencies")
        
        # Try with smaller circuit
        print(f"\n🔄 Trying with smaller circuit (20 qubits)...")
        try_smaller_demo()
    
    print(f"\n🎉 Demo completed!")

def try_smaller_demo():
    """Try demo with smaller circuit if 40-qubit fails."""
    print("🔧 Creating 20-qubit test circuit...")
    
    n_qubits = 20
    import quimb.tensor as qtn
    from lib.map_optimizer import MAPOptimizer
    from lib.circuit_gen import DEVICE
    
    # Create smaller circuit
    circuit_tn = qtn.MPS_computational_state("0" * n_qubits).astype_("complex64")
    circuit_tn.apply_to_arrays(lambda x: torch.from_numpy(x).to(device=DEVICE, dtype=torch.complex64))
    
    # Add physical indices
    for k in range(n_qubits):
        circuit_tn[k].modify(left_inds=[f"k{k}"], tags=[f"I{k}", "MPS", "physical"])
    
    # Add some gates
    np.random.seed(42)
    for i in range(min(10, n_qubits - 1)):
        gate = torch.randn(4, 4, dtype=torch.complex64, device=DEVICE)
        gate = gate + gate.conj().T
        gate = torch.exp(1j * gate)
        
        qubit1 = i
        qubit2 = (i + 1) % n_qubits
        circuit_tn.gate_(gate, (qubit1, qubit2), tags=[f"gate_{i}"])
    
    target_state = "".join("1" if np.random.random() < 0.5 else "0" for _ in range(n_qubits))
    search_space = 2 ** n_qubits
    
    print(f"   🎯 Target state: {target_state}")
    print(f"   📊 Search space: 2^{n_qubits} = {search_space:,}")
    
    try:
        optimizer = MAPOptimizer(
            circuit=circuit_tn,
            target_state=target_state,
            device=DEVICE,
            max_memory_gb=2.0,
            beam_width=16
        )
        
        start_time = time.time()
        bitstring, log_prob, stats = optimizer.find_peaked_bitstring(
            max_iterations=500,
            early_stop_threshold=1e-3
        )
        execution_time = time.time() - start_time
        
        print(f"✅ 20-qubit MAP Optimization Complete!")
        print(f"   🎯 Found bitstring: {bitstring}")
        print(f"   📈 Probability: {np.exp(log_prob):.2e}")
        print(f"   ⏱️  Time: {execution_time:.2f}s")
        
        original_time_hours = search_space / 2000 / 3600
        speedup = original_time_hours * 3600 / execution_time if execution_time > 0 else float('inf')
        print(f"   🚀 Speedup: {speedup:.1f}x faster than original")
        
        optimizer.cleanup()
        
    except Exception as e:
        print(f"❌ Even 20-qubit demo failed: {e}")
        print("Please check your installation and dependencies.")

if __name__ == "__main__":
    quick_demo()
