# A Python Quantum Computing/ Quantum Computational Chemistry Library

## It contains:
- A general quantum circuit constructor: lib/Quantum_Gate.py
- Hartree Fock Computation Code: lib/HartreeFock.py
- Fermionic Operator & Pauli String class to implement abstract operator computations.


## Demos:
- `H2_VQE.ipynb`: A direct application of `lib/Quantum_Gate.py` for finding the ground state of H2 molecule via VQE and H2 molecules' 1st excited state by the SSVQE.

- `SCF.ipynb`: A direct application of `lib/HartreeFock.py`. It shows 
    - Basic elements in SCF computations for H2 and HeH+ molecules.
    - A plot of the potential curve of H2 and HeH+ molecules.

- `fermionic_operator_example.ipynb`: An example file showing basic usage of the `lib/FermionicOperator.py` file and the `lib/PauliString.py` file.

- `Quantum_Gates_Example.ipynb`: Showing construction of basic quantum gates with `lib/Quantum_Gate.py`.


## To Improve:
- In principle, the fermionic Hamiltonian could be constructed using `lib/HartreeFock.py` and `lib/FermionicOperator.py`.
- `FermionicOperator.py` and `PauliString.py` should be inherited from an abstract Operator class.
- Quantum Gates should be constructed using `PauliString.py` instead of computational basis matrix representation.