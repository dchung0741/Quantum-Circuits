from Operator_and_State import SingleOperator, Operator

from functools import reduce
from itertools import product
from numpy import array, eye, kron
from math import prod


def toPauliBasis(matrix: array):
    assert matrix.shape == (2, 2), 'this only works for 2d matrix'
    
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[1, 0]
    d = matrix[1, 1]

    return 0.5 * (a + d), 0.5 * (b + c), 0.5j * (b - c), 0.5 * (a - d)


class SinglePauliString(SingleOperator):

    Id2 = eye(2)
    X = array([[0., 1.], [1., 0.]])
    Y = array([[0., -1j], [1j, 0.]])
    Z = array([[1., 0.], [0., -1.]])

    pauli_dict_map = {'I': Id2, 'X': X, 'Y': Y, 'Z': Z}

    def __init__(self, operator_tup: tuple, coefficient: complex = 1, is_identity: bool = False) -> None:
        
        super().__init__(operator_tup, coefficient, is_identity = is_identity,
                         add_to_type = PauliString, 
                         act_on_type = (PauliString, SinglePauliString), 
                         act_by_type = (PauliString, SinglePauliString))

    def matrix_rep(self):
        
        if type(self.operator_tup[0]) is tuple:
            ps = self.simplify()
        
        else:
            ps = self
        
        ps = ps.operator_tup
        ps = map(lambda x: self.pauli_dict_map[x], ps)
        ps = reduce(lambda x, y: kron(x, y), ps)
        
        return ps * self.coefficient


    def simplify(self):

        op_kron_form = []

        for op in zip(*self.operator_tup):
            op = map(lambda x: self.pauli_dict_map[x], op)
            op = reduce(lambda x, y: x @ y, op)
            op = toPauliBasis(op)
            
            op_kron_form.append(op)

        final_operator = []
        pauli_matrices = [('I', 'X', 'Y', 'Z')] * len(op_kron_form)
        
        for c_tup, op in zip(product(*op_kron_form), product(*pauli_matrices)):
            c = prod(c_tup) * self.coefficient
            if c != 0:
                final_operator.append(SinglePauliString(operator_tup=op, coefficient=c))
        
        return sum(final_operator)



class PauliString(Operator):

    def __init__(self, *SingleFOs) -> None:
        super().__init__(*SingleFOs, 
                         single_type = SinglePauliString, 
                         act_on_type = (PauliString, SinglePauliString), 
                         act_by_type = (PauliString, SinglePauliString))

    def matrix_rep(self):
        simped_op_list = [op.matrix_rep() for op in self.rep]
        return sum(simped_op_list)

    def simplify(self):
        simped_op_list = [op.simplify() for op in self.rep]
        return PauliString(*simped_op_list)


if __name__ == '__main__':

    sp1 = SinglePauliString(operator_tup = (('X', 'I', 'X', 'Z'), ('Y', 'Z', 'X', 'Z')), coefficient = -3.5j)
    sp2 = SinglePauliString(operator_tup = (('X', 'Y', 'I', 'X'),), coefficient = 0.5j)
    sp3 = SinglePauliString(operator_tup = (('Z', 'Y', 'I', 'Y'),), coefficient = 0.5j)
    

    sp4 = sp2 @ (sp1 + sp2) @ (sp3 + sp2)

    print('sp1:')
    print(sp1)
    print(sp1.simplify())
    # print(sp1.matrix_rep())
    
    print('sp2:')
    print(sp2)
    print(sp2.simplify())
    
    print('sp3:')
    print(sp3)
    print(sp3.simplify())

    print('------------------------------- sp4:')
    print(sp1 + sp2)
    print(sp3 + sp2)
    print(sp4)
    print(sp4.simplify())
    print(sp4.matrix_rep())

    
    
    