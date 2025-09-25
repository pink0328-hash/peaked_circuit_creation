#!/usr/bin/env python3
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="40-qubit peak circuit optimization demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--qubits", type=int, default=40, 
                       help="Number of qubits (default: 40)")
    parser.add_argument("--difficulty", type=float, default=1.0,
                       help="Difficulty level (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--method", choices=["map", "boundary_mps", "optimized", "all"], 
                       default="map", help="Optimization method (default: map)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick demo with simple circuit")
    
    args = parser.parse_args()
    
    print("Peak Circuit Optimization Demo")
    print("=" * 50)
    print(f"Qubits: {args.qubits}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Seed: {args.seed}")
    print(f"Method: {args.method}")
    print("=" * 50)
    
    if args.quick:
        # Run quick demo
        print("🏃 Running quick demo...")
        try:
            from quick_demo import quick_demo
            quick_demo()
        except ImportError as e:
            print(f"❌ Error importing quick_demo: {e}")
            print("Make sure quick_demo.py is in the current directory")
            sys.exit(1)
    else:
        # Run full demo
        print("🔧 Running full demo...")
        try:
            from demo_40qubit_optimization import run_demo
            
            methods = [args.method] if args.method != "all" else ["map", "boundary_mps", "optimized"]
            results = run_demo(
                n_qubits=args.qubits,
                difficulty=args.difficulty,
                seed=args.seed,
                methods=methods
            )
            
            print(f"\n✅ Demo completed successfully!")
            print(f"📊 Results: {len(results)} methods tested")
            
        except ImportError as e:
            print(f"❌ Error importing demo: {e}")
            print("Make sure all required files are present:")
            print("  - demo_40qubit_optimization.py")
            print("  - lib/map_optimizer.py")
            print("  - lib/optimized_contraction.py")
            print("  - lib/circuit_gen.py")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print("Try running with --quick flag for a simpler test")
            sys.exit(1)

if __name__ == "__main__":
    main()
