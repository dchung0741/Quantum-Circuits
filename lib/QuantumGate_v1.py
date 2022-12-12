import numpy as np
import mpmath as mp
import sympy as sp
from sympy import sqrt, Rational, I, Matrix, kronecker_product, conjugate
from functools import reduce
from typing import Dict, Tuple


class KetState:

    def __init__(self, state: Tuple[Tuple[int]]) -> None:
        
        # state: | (a_1, b_1), (a_2, b_2), ..., (a_n, b_n) >
        # where (a_1, b_1) = a_1 |0> + b_1 |1>
        self.state_tuple = state
        
        # check if state is valid
        if not self.__is_normalized():
            raise ValueError('Invalid state')
        
        # matrix representation of state
        self.matrix_rep = self.__gen_matrix_rep()


    def __is_normalized(self):
        
        for state in self.state_tuple:
            state_norm = state[0] * conjugate(state[0]) + state[1] * conjugate(state[1])
            if state_norm != 1:
                return False
        
        return True
    
    def __is_zero(self):
        pass

    def __gen_matrix_rep(self):
        state_0 = Matrix([[1], [0]])
        state_1 = Matrix([[0], [1]])
        state_mat = tuple(map(lambda c: c[0] * state_0 + c[1] * state_1, self.state_tuple))
        state_mat = reduce(lambda ket_1, ket_2: kronecker_product(ket_1, ket_2), state_mat)
        return state_mat
    

    def __eq__(self, other) -> bool:
        if isinstance(other, KetState):
            return self.matrix_rep == other.matrix_rep
        else:
            raise TypeError(f'StateKet cannot be compared with {type(other)}.')


    def __hash__(self) -> int:
        return hash(self.state_tuple)


    # ------------------------------ Basic arithmetics -----------------------------
    def __add__(self, other):
        
        if isinstance(other, KetState):
            new_state = [(ket_1[0] + ket_2[0], ket_1[1] + ket_2[1]) for ket_1, ket_2 in zip(self.state_tuple, other.state_tuple)]
            new_state = tuple(new_state)
            return KetState(new_state)
        else:
            raise Exception


    def __radd__(self, other):
        return self + other

    def __neg__(self):
        pass

    def __sub__(self, other):

        if isinstance(other, KetState):
            new_state = [(ket_1[0] - ket_2[0], ket_1[1] - ket_2[1]) for ket_1, ket_2 in zip(self.state_tuple, other.state_tuple)]
            new_state = tuple(new_state)
            return KetState(new_state)
        else:
            raise Exception

    def __rsub__(self, other):
        if isinstance(other, KetState):
            new_state = [(- ket_1[0] + ket_2[0], - ket_1[1] + ket_2[1]) for ket_1, ket_2 in zip(self.state_tuple, other.state_tuple)]
            new_state = tuple(new_state)
            return KetState(new_state)
        else:
            raise Exception
    
    def __matmul__(self):
        pass

    def __rmatmul__(self):
        pass

    def __truediv__(self):
        pass

    def kronecker_product(self):
        pass
    

    # ------------------------------ Reps -----------------------------
    def __repr__(self) -> str:
        return self.matrix_rep.__repr__()
    
    def __str__(self) -> str:
        return self.__repr__()




class NBasisQubit(KetState):

    # state: (1, 0, 0, 1, 0) = | 1, 0, 0, 1, 0 >

    def __init__(self, n: int, state: Tuple[int]) -> None:
        super.__init__(self, state)




class ControlledGate:

    def __init__(self, gate_n: int, 
                 controlled_n: int,
                 unitary_matrix: Matrix) -> None:
        
        self.gate_n = gate_n
        self.n = controlled_n
        self.unitary_matrix = unitary_matrix
        
        self.matrix_rep = self.buildMatrixRep()


    def buildMatrixRep(self) -> Matrix:
        pass



class System:

    def __init__(self) -> None:
        pass


# ------------------------------------------------------------------------------------------------
X = Matrix([[0, 1], [1, 0]])
Y = Matrix([[0, -I], [I, 0]])
Z = Matrix([[1, 0], [0, -1]])
Id_2 = Matrix([[1, 0], [0, 1]])


qubit_0 = KetState(((1, 0), ))
qubit_1 = KetState(((0, 1), ))


if __name__ == '__main__':
    state_1 = KetState( ((1, 0), (1/sqrt(2), I/sqrt(2)), (sqrt(2)/sqrt(3), -I/sqrt(3))) )
    state_2 = KetState( ((1, 0), (1/sqrt(2), I/sqrt(2)), (sqrt(2)/sqrt(3), -I/sqrt(3))) )
    print(state_1 == state_2)
    print(state_1.matrix_rep)