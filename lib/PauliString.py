from functools import reduce
from itertools import product
from numpy import array, eye
from math import prod



def toPauliBasis(matrix: array):
        assert matrix.shape == (2, 2), 'this only works for 2d matrix'
        
        a = matrix[0, 0]
        b = matrix[0, 1]
        c = matrix[1, 0]
        d = matrix[1, 1]

        return 0.5 * (a + d), 0.5 * (b + c), 0.5j * (b - c), 0.5 * (a - d)


class PauliString:

    def __init__(self, *SingleFOs) -> None:
        
        """
        self.single_fermionic_operators: of SingleFermionicOperator
        """
        self.single_fermionic_operators = SingleFOs
        self.dict_adder = lambda d1, d2: {k: v for k in set(d1)|set(d2) if (v := d1.get(k, 0) + d2.get(k, 0)) != 0}
        
        single_fos = map(lambda fo: fo.rep, SingleFOs)
        self.fermionic_operator = reduce(self.dict_adder, single_fos)
        
        self.rep = set([SinglePauliString( operator_tup=op_tup, coefficient=op_c) for op_tup, op_c in self.fermionic_operator.items()])


    def simplify(self):
        simped_op_list = [op.simplify() for op in self.rep]
        return PauliString(*simped_op_list)

    def __len__(self):
        return len(self.rep)

    def __add__(self, other):
        assert isinstance(other, PauliString) or isinstance(other, SinglePauliString) or other == 0, '123'
        
        if isinstance(other, PauliString):
            fo1 = tuple(self.rep)
            fo2 = tuple(other.rep)
            new_op = PauliString(*(fo1 + fo2))

        elif isinstance(other, SinglePauliString):
            fo1 = tuple(self.rep)
            fo2 = (other,)
            new_op = PauliString(*(fo1 + fo2))

        elif other == 0:
            new_op = self
        
        else:
            pass
        
        """---------------------------------------------------------"""
        if len(new_op.rep) == 0:
            return 0
        
        elif len(new_op.rep) == 1:
            return list(new_op.rep)[0]
        
        else:
            return new_op


    def __radd__(self, other):
        return self + other

    def __neg__(self):
        new_rep = map(lambda x: -x, self.rep)
        return PauliString(*new_rep)

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return  - self + other
    
    def __mul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        new_rep = map(lambda x: other * x, self.rep)
        return PauliString(*new_rep)

    def __rmul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return self * other
    
    def __truediv__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return 1/other * self
    
    def __matmul__(self, other):
        assert isinstance(other, SinglePauliString) or isinstance(other, PauliString), 'Not SingleFermionicOperator or FermionicOperator'
        if isinstance(other, SinglePauliString):
            new_rep = map(lambda x: x @ other, self.rep)
        
        elif isinstance(other, PauliString):
            new_rep = [op1 @ op2 for op1, op2 in product(self.rep, other.rep)]
        
        return PauliString(*new_rep)
    
    def __rmatmul__(self, other):
        assert isinstance(other, SinglePauliString) or isinstance(other, PauliString), 'Not SingleFermionicOperator or FermionicOperator'
        
        if isinstance(other, SinglePauliString):
            new_rep = map(lambda x: other@x, self.rep)
            return PauliString(*new_rep)
        
        else:
            return other @ self
    

    def __repr__(self) -> str:
        str_rep = ''
        for s in self.rep:
            str_rep += str(s)
            str_rep += '\n'
        return str_rep
    
    def __str__(self) -> str:
        return self.__repr__()


class SinglePauliString:

    Id2 = eye(2)
    X = array([[0., 1.], [1., 0.]])
    Y = array([[0., -1j], [1j, 0.]])
    Z = array([[1., 0.], [0., -1.]])

    pauli_dict_map = {'I': Id2, 'X': X, 'Y': Y, 'Z': Z}


    def __init__(self, operator_tup: tuple, coefficient: complex) -> None:
        
        """
        operator_tup = (('X', 'Y'), ('I', 'Z'))
        """
        self.operator_tup = operator_tup
        self.coefficient = coefficient

        self.rep = {self.operator_tup: self.coefficient}
    

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

    def __hash__(self) -> int:
        return hash(tuple(self.rep))

    def __eq__(self, other) -> bool:
        assert isinstance(other, SinglePauliString), 'SingleFermionicOperator can only be compared to SingleFermionicOperator'
        return hash(self) == hash(other)
    
    def __add__(self, other):
        
        if isinstance(other, SinglePauliString):
            if other.operator_tup == self.operator_tup:
                c = self.coefficient + other.coefficient
                return SinglePauliString(operator_tup=self.operator_tup, coefficient = c)        
            else:
                return PauliString(self, other)
        
        if isinstance(other, PauliString):
            return other + self
        
        if other == 0:
            return self

    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        c = - self.coefficient
        return SinglePauliString(operator_tup=self.operator_tup, coefficient=c)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        c = self.coefficient * other
        return SinglePauliString(operator_tup=self.operator_tup, coefficient=c)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return 1/other * self
    
    def __matmul__(self, other):
        assert isinstance(other, SinglePauliString) or isinstance(other, PauliString), 'Not SingleFermionicOperator or FermionicOperator'
        if isinstance(other, SinglePauliString):
            new_op = self.operator_tup + other.operator_tup
            new_c = self.coefficient * other.coefficient
            return SinglePauliString(operator_tup=new_op, coefficient=new_c)

        if isinstance(other, PauliString):
            return other.__rmatmul__(self)
    
    def __rmatmul__(self, other):
        return other @ self
        
    def __repr__(self) -> str:
        return str(self.rep)[1:-1]
    
    def __str__(self) -> str:
        return self.__repr__()



if __name__ == '__main__':

    sp1 = SinglePauliString(operator_tup = (('X', 'I', 'X', 'Z'), ('Y', 'Z', 'X', 'Z')), coefficient = -3.5j)
    sp2 = SinglePauliString(operator_tup = (('X', 'Y', 'I', 'X'),), coefficient = 0.5j)
    sp3 = SinglePauliString(operator_tup = (('Z', 'Y', 'I', 'Y'),), coefficient = 0.5j)
    

    sp4 = (sp1 + sp2) @ (sp3 + sp2)

    print(sp1)
    print(sp1.simplify())
    print(sp2)
    print(sp2.simplify())
    print(sp3)
    print(sp3.simplify())

    print('sp4:')
    print(sp4)
    print(sp4.simplify().rep)

    
    
    