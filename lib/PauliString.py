from .Operator_and_State import SingleOperator, Operator, SingleKet, Ket

from functools import reduce
from itertools import product
from numpy import array, eye, kron, diag
from numpy.random import choice

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
                         act_on_type = (PauliString, SinglePauliString, SingleQuantumState, QuantumState), 
                         act_by_type = (PauliString, SinglePauliString, SingleQuantumState, QuantumState))

    def matrix_rep(self):
        
        ps = self.simplify()
        c = ps.coefficient
        ps = ps.operator_tup[0]
        ps = map(lambda x: self.pauli_dict_map[x], ps)
        ps = reduce(lambda x, y: kron(x, y), ps)
        
        return ps * c


    def simplify(self, with_map = None):

        op_kron_form = []
        op_dict = self.pauli_dict_map if with_map is None else self.pauli_dict_map|with_map
        for op in zip(*self.operator_tup):
            op = map(lambda x: op_dict[x], op)
            op = reduce(lambda x, y: x @ y, op)
            op = toPauliBasis(op.round(10))
            
            op_kron_form.append(op)

        final_operator = []
        pauli_matrices = [('I', 'X', 'Y', 'Z')] * len(op_kron_form)
        
        for c_tup, op in zip(product(*op_kron_form), product(*pauli_matrices)):
            c = prod(c_tup) * self.coefficient
            if c != 0:
                final_operator.append(SinglePauliString(operator_tup=(op,), coefficient=c))
        
        return sum(final_operator)



class PauliString(Operator):

    def __init__(self, *SingleFOs) -> None:
        super().__init__(*SingleFOs, 
                         single_type = SinglePauliString, 
                         act_on_type = (PauliString, SinglePauliString, SingleQuantumState, QuantumState), 
                         act_by_type = (PauliString, SinglePauliString, SingleQuantumState, QuantumState))

    def matrix_rep(self):
        simped_op_list = [op.matrix_rep() for op in self.rep]
        return sum(simped_op_list)

    def simplify(self, with_map = None):
        simped_op_list = [op.simplify(with_map = with_map) for op in self.rep]
        return sum(simped_op_list)


class SingleQuantumState(SingleKet):

    state_map = {0: array([[1], [0]]), 1: array([[0], [1]])}

    def __init__(self, label_tup: tuple, coefficient: complex = 1, operator: SinglePauliString = None) -> None:
        if operator is None:
            n_bit = len(label_tup)
            operator = SinglePauliString((('I',) * n_bit,), is_identity=True)
        super().__init__(label_tup, coefficient, operator = operator, 
                         add_to_type = QuantumState, 
                         act_by_type = (PauliString, SinglePauliString))


    def evaluate(self):
        operator = self.operator if len(self.operator.operator_tup) == 1 else self.operator.simplify()
        res = [SinglePauliString.pauli_dict_map[o] @ self.state_map[l] for o, l in zip(operator.operator_tup[0], self.label_tup)]
        overall_coeff = operator.coefficient * self.coefficient
        n = len(res)
        res = [SingleQuantumState(label_tup=comb, coefficient = c * overall_coeff) for comb in product(range(2), repeat=n) if (c := prod(array(res)[range(n), comb, 0])) !=0 ]

        if len(res) == 0:
            return 0
        elif len(res) == 1:
            return res[0]
        else:
            return QuantumState(*res)


class QuantumState(Ket):

    def __init__(self, *SingleQSs) -> None:
        super().__init__(*SingleQSs, 
                         single_type = SingleQuantumState, 
                         act_by_type = (PauliString, SinglePauliString))
    
    def evaluate(self):

        res = [sket.evaluate() for sket in self.rep]
        return sum(res)

    def sample(self, n):

        c_list = array([s.coefficient for s in self.rep])
        p_list = (c_list.conj() * c_list).real
        label_dict = {i: s.label_tup for i, s in enumerate(self.rep)}

        out_sample = choice(list(label_dict.keys()), size=n, p=p_list)
        
        return map(lambda s: label_dict[s], out_sample)


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
    # print(sp4.matrix_rep())
    

    swap = SinglePauliString(operator_tup=(('I', 'I', 'I', 'I'),), coefficient=0.5) 
    swap = swap + SinglePauliString(operator_tup=(('X', 'I', 'X', 'I'),), coefficient=0.5) 
    swap = swap + SinglePauliString(operator_tup=(('Y', 'I', 'Y', 'I'),), coefficient=0.5) 
    swap = swap + SinglePauliString(operator_tup=(('Z', 'I', 'Z', 'I'),), coefficient=0.5) 
    
    sqs1 = SingleQuantumState(label_tup=(1, 0, 0, 1), coefficient=1)
    sqs2 = SingleQuantumState(label_tup=(0, 1, 0, 1), coefficient=3)
    sqs3 = SingleQuantumState(label_tup=(1, 1, 0, 1), coefficient=-1j)
    
    print((swap @ (sqs1 + sqs2 + sqs3)).evaluate())

    print(type(swap @ sqs1))

    
    print(sqs1)
    print(sqs1.evaluate())
    import timeit

    def test():
        (swap @ sqs1).evaluate()
    
    # print(timeit.timeit(test, number=10000))

    sp4 = SinglePauliString(operator_tup = (('Z', 'Y', 'P0', 'Y'),), coefficient = 0.5j)
    print(swap @ sp4)
    print((swap @ sp4).simplify(with_map={'P0': array([[1, 0], [0, 0]])}))
    