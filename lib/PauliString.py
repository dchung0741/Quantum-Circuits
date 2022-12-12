from functools import reduce
from itertools import product


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
            return new_rep
        
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

    def __init__(self, operator_tup: tuple, coefficient: complex) -> None:
        
        """
        operator_tup = (position = int, dagger = 0, 1)
        """
        self.operator_tup = operator_tup
        self.coefficient = coefficient

        self.rep = {self.operator_tup: self.coefficient}
    
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

    sfo1 = SinglePauliString(operator_tup = ('X',), coefficient = -3.5j)
    sfo2 = SinglePauliString(operator_tup = ('X',), coefficient = 0.5j)
    sfo3 = SinglePauliString(operator_tup = ('Y', 'Z'), coefficient = 2)
    sfo4 = SinglePauliString(operator_tup = ('X',), coefficient = 2)

    fo1 = PauliString(sfo1, sfo2, sfo3)
    fo2 = PauliString(sfo1, sfo3, sfo4)
    
    
    print('fo1', fo1)
    print('fo2', fo2)
    print(sfo4 @ fo1)
    print(fo2 @ fo1)