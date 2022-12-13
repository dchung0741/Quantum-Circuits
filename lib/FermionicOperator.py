from functools import reduce
from itertools import product
from PauliString import PauliString, SinglePauliString, toPauliBasis


def jwEncodingMap(n: int):
    
    jw_dict = {}

    for i in range(n):
        cXi = tuple('Z' * i + 'X' + 'I' * (n - i - 1))
        cYi = tuple('Z' * i + 'Y' + 'I' * (n - i - 1))
        
        jw_dict[(i, 1)] = SinglePauliString(operator_tup= (cXi,), coefficient=0.5) - SinglePauliString(operator_tup= (cYi,), coefficient=0.5j)
        jw_dict[(i, 0)] = SinglePauliString(operator_tup= (cXi,), coefficient=0.5) + SinglePauliString(operator_tup= (cYi,), coefficient=0.5j)
    
    return jw_dict


class FermionicOperator:

    def __init__(self, *SingleFOs) -> None:
        
        """
        self.single_fermionic_operators: of SingleFermionicOperator
        """
        self.single_fermionic_operators = SingleFOs
        self.dict_adder = lambda d1, d2: {k: v for k in set(d1)|set(d2) if (v := d1.get(k, 0) + d2.get(k, 0)) != 0}
        
        single_fos = map(lambda fo: fo.rep, SingleFOs)
        self.fermionic_operator = reduce(self.dict_adder, single_fos)
        
        self.rep = set([SingleFermionicOperator( operator_tup=op_tup, coefficient=op_c) for op_tup, op_c in self.fermionic_operator.items()])
        if len(self.rep) == 0:
            self.rep = 0

    
    def get_dict(self):
        dict_form = { list(c.rep.keys())[0]: list(c.rep.values())[0] for c in self.rep}
        return dict_form


    def __len__(self):
        return len(self.rep)

    def __add__(self, other):
        assert isinstance(other, FermionicOperator) or isinstance(other, SingleFermionicOperator) or other == 0, '123'
        
        if isinstance(other, FermionicOperator):
            fo1 = tuple(self.rep)
            fo2 = tuple(other.rep)
            new_op = FermionicOperator(*(fo1 + fo2))

        elif isinstance(other, SingleFermionicOperator):
            fo1 = tuple(self.rep)
            fo2 = (other,)
            new_op = FermionicOperator(*(fo1 + fo2))

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
        return FermionicOperator(*new_rep)

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return  - self + other
    
    def __mul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        new_rep = map(lambda x: other * x, self.rep)
        return FermionicOperator(*new_rep)

    def __rmul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return self * other
    
    def __truediv__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return 1/other * self
    
    def __matmul__(self, other):
        assert isinstance(other, SingleFermionicOperator) or isinstance(other, FermionicOperator), 'Not SingleFermionicOperator or FermionicOperator'
        if isinstance(other, SingleFermionicOperator):
            new_rep = map(lambda x: x @ other, self.rep)
        
        elif isinstance(other, FermionicOperator):
            new_rep = [op1 @ op2 for op1, op2 in product(self.rep, other.rep)]
        
        return FermionicOperator(*new_rep)
    
    def __rmatmul__(self, other):
        assert isinstance(other, SingleFermionicOperator) or isinstance(other, FermionicOperator), 'Not SingleFermionicOperator or FermionicOperator'
        
        if isinstance(other, SingleFermionicOperator):
            new_rep = map(lambda x: other@x, self.rep)
            return FermionicOperator(*new_rep)
        
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


class SingleFermionicOperator:


    def __init__(self, operator_tup: tuple, coefficient: complex, jw_map: dict = None) -> None:
        
        """
        operator_tup = (position = int, dagger = 0, 1)
        """
        self.operator_tup = operator_tup
        self.coefficient = coefficient

        self.rep = {self.operator_tup: self.coefficient}
        
        self.jw_map = jw_map
    

    def jw_encoding(self):
        assert self.jw_map is not None, 'This method is not suitable for this operator'

        jw_op = [self.jw_map[op] for op in self.operator_tup]
        jw_op = reduce(lambda x, y: x @ y, jw_op)
        jw_op = jw_op.simplify() * self.coefficient
        
        return jw_op


    def __hash__(self) -> int:
        return hash(tuple(self.rep))

    def __eq__(self, other) -> bool:
        assert isinstance(other, SingleFermionicOperator), 'SingleFermionicOperator can only be compared to SingleFermionicOperator'
        return hash(self) == hash(other)
    
    def __add__(self, other):
        
        if isinstance(other, SingleFermionicOperator):
            if other.operator_tup == self.operator_tup:
                c = self.coefficient + other.coefficient
                return SingleFermionicOperator(operator_tup=self.operator_tup, coefficient = c)        
            else:
                return FermionicOperator(self, other)
        
        if isinstance(other, FermionicOperator):
            return other + self
        
        if other == 0:
            return self

    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        c = - self.coefficient
        return SingleFermionicOperator(operator_tup=self.operator_tup, coefficient=c)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        c = self.coefficient * other
        return SingleFermionicOperator(operator_tup=self.operator_tup, coefficient=c)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return 1/other * self
    
    def __matmul__(self, other):
        assert isinstance(other, SingleFermionicOperator) or isinstance(other, FermionicOperator), 'Not SingleFermionicOperator or FermionicOperator'
        if isinstance(other, SingleFermionicOperator):
            new_op = self.operator_tup + other.operator_tup
            new_c = self.coefficient * other.coefficient
            return SingleFermionicOperator(operator_tup=new_op, coefficient=new_c)

        if isinstance(other, FermionicOperator):
            return other.__rmatmul__(self)
    
    def __rmatmul__(self, other):
        return other @ self
        
    def __repr__(self) -> str:
        return str(self.rep)[1:-1]
    
    def __str__(self) -> str:
        return self.__repr__()


if __name__ == '__main__':

    sfo1 = SingleFermionicOperator(operator_tup = ((0, 0),), coefficient = -3.5j)
    sfo2 = SingleFermionicOperator(operator_tup = ((3, 0),), coefficient = 0.5j)
    sfo3 = SingleFermionicOperator(operator_tup = ((0, 1),), coefficient = 2)
    sfo4 = SingleFermionicOperator(operator_tup = ((1, 1),), coefficient = 2)

    fo1 = FermionicOperator(sfo1, sfo2, sfo3)
    fo2 = FermionicOperator(sfo1, sfo3, sfo4)
    
    
    print('fo1', fo1)
    print('fo2', fo2)
    print(sfo4 @ fo1)
    print(fo2 @ fo1)
    print(fo2.get_dict())

    

    print('----------- Jordan Wigner ------------------')
    
    n = 5
    jw = jwEncodingMap(n)
    for i, d in product(range(n), range(2)):
        print((i, d))
        print(jw[(i, d)])

    op1 = jw[(1, 1)] @ jw[(1, 0)]
    op2 = jw[(1, 0)] @ jw[(1, 1)]
    
    print(((op1 + op2).simplify()))
    sfo1 = SingleFermionicOperator(operator_tup = ((0, 0), (3, 1)), coefficient = -3.5j, jw_map=jw)
    print(sfo1.jw_encoding())