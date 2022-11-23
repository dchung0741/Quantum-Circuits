from functools import reduce
import itertools
from numpy import array, kron, pi, exp, sqrt, sin, cos, eye
from typing import List, Dict, Tuple

I = 1j

bit_map = {0: array([[1, 0]]).T, 1: array([[0, 1]]).T}

X = array([[0, 1], [1, 0]])
Y = array([[0, -I], [I, 0]])
Z = array([[1, 0], [0, -1]])

H = (X + Z)/sqrt(2)
T = array([[1, 0], [0, exp(I * pi/4)]])
S = array([[1, 0], [0, I]])
Id2 = eye(2)


def Rx(theta):
    return cos(theta/2) * eye(2) + I * X * sin(theta/2)

def RY(theta):
    return cos(theta/2) * eye(2) + I * Y * sin(theta/2)

def RZ(theta):
    return cos(theta/2) * eye(2) + I * Z * sin(theta/2)


class Ket:

    def __init__(self) -> None:
        pass


class BasisKet:

    def __init__(self, *label) -> None:
        self.dim = len(label)
        self.label = label
        self.matrix_rep = self.get_mat_rep()
        
    
    def get_mat_rep(self) -> array:
        ket_mat = [bit_map[bit] for bit in self.label]
        return reduce(kron, ket_mat)
        

    
    @property
    def T(self):
        return self.matrix_rep.T

    def __eq__(self, other):
        if isinstance(other, BasisKet):
            return self.label == other.label

    def __hash__(self) -> int:
        return hash(self.label)

    def __repr__(self) -> str:
        return self.matrix_rep.__repr__()
    
    def __str__(self) -> str:
        return self.matrix_rep.__str__()


def generate_basis(dim: int) -> List[BasisKet]:

    basis_kets = [[0, 1]] * dim
    basis_kets = map(lambda lab: BasisKet(*lab), itertools.product(*basis_kets))
    basis_kets = list(basis_kets)

    return basis_kets


class DensityMatrix:

    def __init__(self) -> None:
        pass



class QuantumGate:

    def __init__(self, n: int, free_bits_dict: Dict[int, array], target_bits_dict: Dict[int, array], controlled_bits_dict: Dict[int, int]) -> None:

        """
        channel label starts from 0
        n: total number of channels
        free_bits_dict: {channel label: gate operator}
        target_bits_dict: {channel label: gate operator}
        controlled_bits_dict: {channel label: activate_on 0 or 1}
        """
        
        self.n = n

        for i in range(self.n):
            if (i not in free_bits_dict) and (i not in target_bits_dict) and (i not in controlled_bits_dict):
                free_bits_dict[i] = Id2

        self.basis_kets = generate_basis(self.n)

        self.free_bits_dict = free_bits_dict
        self.free_bits_idx = free_bits_dict.keys()
        self.free_bits_val = free_bits_dict.values()

        self.target_bits_dict = target_bits_dict
        self.target_bits_idx = self.target_bits_dict.keys()
        
        self.controlled_bits_dict = controlled_bits_dict
        self.controlled_bits_idx = self.controlled_bits_dict.keys()
        self.controlled_bits_val = self.controlled_bits_dict.values()

        self.matrix_rep = self.compute_matrix_rep()
        

    def compute_matrix_rep(self) -> array:

        def matrix_el(i, j):
            
            out_proj = self.basis_kets[i].matrix_rep.T
            in_state = self.basis_kets[j]

            in_state_control_bit = tuple([in_state.label[i] for i in self.controlled_bits_idx])

            if in_state_control_bit != tuple(self.controlled_bits_val):
                operator = [self.free_bits_dict[i] if i in self.free_bits_dict else Id2 for i in range(self.n)]
            else:
                activated_gate = self.free_bits_dict|self.target_bits_dict
                operator = [activated_gate[i] if i in activated_gate else Id2 for i in range(self.n)]
            
            operator = reduce(kron, operator)
            out_state = operator @ in_state.matrix_rep
            

            return (out_proj @ out_state)[0, 0]
            

        matrix = array([[matrix_el(i, j) for j in range(2**self.n)] for i in range(2**self.n)])

        return matrix
    

    @classmethod
    def fromTruthTable(cls, truth_table: Dict[array, array]):
        

        pass

        



class QuantumCircuit:

    def __init__(self) -> None:
        pass



if __name__ == '__main__':
    pass