import heapq
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import quimb.tensor as qtn
import cotengra as ctg
from collections import defaultdict

@dataclass
class PartialAssignment:
    assignment: Dict[int, int]  # qubit_index -> bit_value (0 or 1)
    bound: float  # upper bound on log probability
    unassigned_qubits: List[int]  # qubits not yet assigned
    
    def __lt__(self, other):
        # For max-heap, we want higher bounds first
        return self.bound > other.bound

class MAPOptimizer:
    
    def __init__(self, 
                 circuit: qtn.TensorNetwork,
                 target_state: str,
                 device: str = "cuda",
                 max_memory_gb: float = 8.0,
                 beam_width: int = 64):
        self.circuit = circuit
        self.target_state = target_state
        self.n_qubits = len(target_state)
        self.device = device
        self.max_memory_gb = max_memory_gb
        self.beam_width = beam_width
        
        # Initialize cotengra optimizer
        self.opti = ctg.ReusableHyperOptimizer(
            progbar=False,
            methods=["greedy", "kahypar"],
            reconf_opts={},
            max_repeats=50,
            optlib="optuna",
        )
        
        # Build double-layer tensor network
        self.double_layer_tn = self._build_double_layer_network()
        
        # Cache for contraction environments
        self.environment_cache = {}
        
    def _build_double_layer_network(self) -> qtn.TensorNetwork:
        """Build the double-layer tensor network for |⟨x|U|0…0⟩|^2."""
        # Create computational basis state |0...0⟩
        init_state = qtn.MPS_computational_state("0" * self.n_qubits).astype_("complex64")
        init_state.apply_to_arrays(lambda x: torch.from_numpy(x).to(device=self.device, dtype=torch.complex64))
        
        # Add physical indices
        for k in range(self.n_qubits):
            init_state[k].modify(left_inds=[f"k{k}"], tags=[f"I{k}", "MPS"])
        
        # Create double-layer: U ⊗ U*
        circuit_conj = self.circuit.conj()
        
        # Contract to get the full double-layer network
        double_layer = init_state & self.circuit & circuit_conj
        
        return double_layer
    
    def find_peaked_bitstring(self, 
                             max_iterations: int = 10000,
                             early_stop_threshold: float = 1e-6) -> Tuple[str, float, Dict[str, Any]]:

        start_time = time.time()
        
        
        initial_assignment = PartialAssignment(
            assignment={},
            bound=self._compute_upper_bound({}),
            unassigned_qubits=list(range(self.n_qubits))
        )
        
       
        pq = [initial_assignment]
        best_bitstring = None
        best_log_prob = float('-inf')
        iterations = 0
        pruned_count = 0
        
        while pq and iterations < max_iterations:
            iterations += 1
            

            current = heapq.heappop(pq)
            
         
            if current.bound <= best_log_prob + early_stop_threshold:
                pruned_count += 1
                continue
            
           
            if not current.unassigned_qubits:
                bitstring = self._assignment_to_bitstring(current.assignment)
                log_prob = self._compute_exact_log_prob(bitstring)
                
                if log_prob > best_log_prob:
                    best_bitstring = bitstring
                    best_log_prob = log_prob
                continue
            
  
            next_qubit = current.unassigned_qubits[0]
            
            for bit_value in [0, 1]:
                new_assignment = current.assignment.copy()
                new_assignment[next_qubit] = bit_value
                
                new_unassigned = current.unassigned_qubits[1:]
                new_bound = self._compute_upper_bound(new_assignment)
                
          
                if new_bound > best_log_prob + early_stop_threshold:
                    new_partial = PartialAssignment(
                        assignment=new_assignment,
                        bound=new_bound,
                        unassigned_qubits=new_unassigned
                    )
                    heapq.heappush(pq, new_partial)
            
    
            if len(pq) > self.beam_width:
                pq = heapq.nlargest(self.beam_width, pq)
                heapq.heapify(pq)
        
        total_time = time.time() - start_time
        
        stats = {
            "iterations": iterations,
            "pruned_count": pruned_count,
            "total_time": total_time,
            "queue_size": len(pq),
            "best_log_prob": best_log_prob
        }
        
        return best_bitstring or "0" * self.n_qubits, best_log_prob, stats
    
    def _compute_upper_bound(self, partial_assignment: Dict[int, int]) -> float:
        if not partial_assignment:
            # No assignments yet - use full relaxed bound
            return self._compute_relaxed_bound({})
        
        # Create a relaxed network for the bound
        relaxed_tn = self._create_relaxed_network(partial_assignment)
        
        try:
            # Contract the relaxed network
            result = relaxed_tn.contract(optimize=self.opti)
            return float(torch.log(torch.abs(result) + 1e-12))
        except Exception as e:
            # If contraction fails, return a conservative bound
            return -1e6
    
    def _create_relaxed_network(self, partial_assignment: Dict[int, int]) -> qtn.TensorNetwork:
        """Create a relaxed tensor network for bound computation."""
        # Start with a copy of the double-layer network
        relaxed_tn = self.double_layer_tn.copy()
        
        # For assigned qubits, fix their values
        for qubit, bit_value in partial_assignment.items():
            # Create a delta function tensor for this qubit
            delta_tensor = torch.zeros(2, device=self.device, dtype=torch.complex64)
            delta_tensor[bit_value] = 1.0
            
            # Replace the physical leg with this delta
            relaxed_tn.replace_with_identity(
                f"k{qubit}", 
                delta_tensor,
                inplace=True
            )
        

        for qubit in range(self.n_qubits):
            if qubit not in partial_assignment:
            
                max_tensor = torch.ones(2, device=self.device, dtype=torch.complex64)
             
                relaxed_tn.replace_with_identity(
                    f"k{qubit}",
                    max_tensor,
                    inplace=True
                )
        
        return relaxed_tn
    
    def _compute_relaxed_bound(self, partial_assignment: Dict[int, int]) -> float:
        return 0.0 
    
    def _compute_exact_log_prob(self, bitstring: str) -> float:
        delta_tensors = {}
        for i, bit in enumerate(bitstring):
            delta = torch.zeros(2, device=self.device, dtype=torch.complex64)
            delta[int(bit)] = 1.0
            delta_tensors[f"k{i}"] = delta
        
        exact_tn = self.double_layer_tn.copy()
        for qubit, delta in delta_tensors.items():
            exact_tn.replace_with_identity(qubit, delta, inplace=True)
        
        try:
            result = exact_tn.contract(optimize=self.opti)
            return float(torch.log(torch.abs(result) + 1e-12))
        except Exception as e:
            return float('-inf')
    
    def _assignment_to_bitstring(self, assignment: Dict[int, int]) -> str:
        bitstring = ['0'] * self.n_qubits
        for qubit, bit_value in assignment.items():
            bitstring[qubit] = str(bit_value)
        return ''.join(bitstring)
    
    def cleanup(self):
        self.opti.cleanup()
        self.environment_cache.clear()


class BoundaryMPSOptimizer:
    
    def __init__(self, circuit: qtn.TensorNetwork, target_state: str, device: str = "cuda"):
        self.circuit = circuit
        self.target_state = target_state
        self.n_qubits = len(target_state)
        self.device = device
        
        # Initialize optimizer
        self.opti = ctg.ReusableHyperOptimizer(
            progbar=False,
            methods=["greedy"],
            reconf_opts={},
            max_repeats=20,
            optlib="optuna",
        )
    
    def find_peaked_bitstring(self, beam_width: int = 64) -> Tuple[str, float, Dict[str, Any]]:

        start_time = time.time()
        
        current_prefix = {}
        bitstring = ['0'] * self.n_qubits
        
        for qubit in range(self.n_qubits):
    
            prob_0 = self._compute_conditional_prob(current_prefix, qubit, 0)
            prob_1 = self._compute_conditional_prob(current_prefix, qubit, 1)
            
           
            if prob_1 > prob_0:
                bitstring[qubit] = '1'
                current_prefix[qubit] = 1
            else:
                bitstring[qubit] = '0'
                current_prefix[qubit] = 0
        
        final_prob = self._compute_exact_prob(''.join(bitstring))
        
        total_time = time.time() - start_time
        
        stats = {
            "total_time": total_time,
            "method": "boundary_mps",
            "final_prob": final_prob
        }
        
        return ''.join(bitstring), final_prob, stats
    
    def _compute_conditional_prob(self, prefix: Dict[int, int], qubit: int, bit_value: int) -> float:
        test_assignment = prefix.copy()
        test_assignment[qubit] = bit_value
        
        return self._compute_exact_prob(self._assignment_to_bitstring(test_assignment))
    
    def _compute_exact_prob(self, bitstring: str) -> float:
        delta_tensors = {}
        for i, bit in enumerate(bitstring):
            delta = torch.zeros(2, device=self.device, dtype=torch.complex64)
            delta[int(bit)] = 1.0
            delta_tensors[f"k{i}"] = delta
        
        exact_tn = self.circuit.copy()
        for qubit, delta in delta_tensors.items():
            exact_tn.replace_with_identity(qubit, delta, inplace=True)
        
        try:
            result = exact_tn.contract(optimize=self.opti)
            return float(torch.abs(result) ** 2)
        except Exception as e:
            return 0.0
    
    def _assignment_to_bitstring(self, assignment: Dict[int, int]) -> str:
        bitstring = ['0'] * self.n_qubits
        for qubit, bit_value in assignment.items():
            bitstring[qubit] = str(bit_value)
        return ''.join(bitstring)
    
    def cleanup(self):
        self.opti.cleanup()
