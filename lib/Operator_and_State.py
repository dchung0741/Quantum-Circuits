from functools import reduce
from itertools import product
from numpy import array, eye, kron
from math import prod



class SingleOperator:

    def __init__(self, operator_tup: tuple, coefficient: complex = 1, is_identity = False,
                 add_to_type = None,
                 act_on_type = None,
                 act_by_type = None,
                 ) -> None:
        
        """
        operator_tup = (position = int, dagger = 0, 1)
        """
        self.operator_tup = operator_tup
        self.coefficient = coefficient
        self.is_identity = is_identity
        # if is_identity:
        #     assert list(set(self.operator_tup))[0] == 'I'

        self.rep = {self.operator_tup: self.coefficient}


        self.add_to_type = Operator if add_to_type is None else add_to_type
        self.act_on_type = (SingleOperator, Operator, Ket, SingleKet) if act_on_type is None else act_on_type
        self.act_by_type = (SingleOperator, Operator) if act_by_type is None else act_by_type


    def __hash__(self) -> int:
        return hash(tuple(self.rep))

    def __eq__(self, other) -> bool:
        assert isinstance(other, SingleOperator), 'SingleFermionicOperator can only be compared to SingleFermionicOperator'
        return hash(self) == hash(other)
    

    # Basic Arithmetics
    def __add__(self, other):

        assert type(other) in (type(self), self.add_to_type) or other == 0, f'{(type(self))} cannot be added to {type(other)}'
        
        if isinstance(other, SingleOperator):
            if other.operator_tup == self.operator_tup:
                c = self.coefficient + other.coefficient
                is_identity = self.is_identity and other.is_identity
                return type(self)(operator_tup = self.operator_tup, coefficient = c, is_identity = is_identity)
            else:
                return self.add_to_type(self, other)
        
        if isinstance(other, self.add_to_type):
            return other + self
        
        if other == 0:
            return self


    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        c = - self.coefficient
        return type(self)(operator_tup=self.operator_tup, coefficient=c)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not multiplied by a numeric value'
        c = self.coefficient * other
        return type(self)(operator_tup=self.operator_tup, coefficient=c, is_identity = self.is_identity)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not divided by numeric value'
        return 1/other * self
    
    def __matmul__(self, other):
        assert type(other) in (*self.act_by_type, *self.act_on_type), f'{type(other)} cannot act on {type(self)}'
        
        if isinstance(other, SingleOperator):
            
            # c1*I @ c2*I
            if self.is_identity and other.is_identity:
                assert self.operator_tup == other.operator_tup, 'Identity not of the same dimension'
                new_op = self.operator_tup
                new_c = self.coefficient * other.coefficient
                return type(self)(operator_tup=new_op, coefficient=new_c, is_identity = True)

            # c1*I @ other
            elif self.is_identity and not other.is_identity:
                return self.coefficient * other
            
            # self @ c2*I
            elif not self.is_identity and other.is_identity:
                return self * other.coefficient

            # self @ other
            else:
                new_op = self.operator_tup + other.operator_tup
                new_c = self.coefficient * other.coefficient
                return type(self)(operator_tup=new_op, coefficient=new_c)

        if isinstance(other, Operator):
            if self.is_identity:
                return self.coefficient * other        
            else:
                return other.__rmatmul__(self)
        
        if isinstance(other, SingleKet):
            return other.__rmatmul__(self)
        
        if isinstance(other, Ket):
            return other.__rmatmul__(self)
        
    def __rmatmul__(self, other):
        return other @ self
        
    def __repr__(self) -> str:
        return str(self.rep)[1:-1]
    
    def __str__(self) -> str:
        return self.__repr__()


class Operator:

    def __init__(self, *SingleFOs, 
                 single_type = None,
                 act_on_type = None,
                 act_by_type = None,
                 ) -> None:
        
        self.single_fermionic_operators = SingleFOs
        self.dict_adder = lambda d1, d2: {k: v for k in set(d1)|set(d2) if (v := d1.get(k, 0) + d2.get(k, 0)) != 0}
        
        single_ops = map(lambda op: op.rep, SingleFOs)
        self.operator = reduce(self.dict_adder, single_ops)

        self.single_type = SingleOperator if single_type is None else single_type
        self.act_on_type = (SingleOperator, Operator, Ket, SingleKet) if act_on_type is None else act_on_type
        self.act_by_type = (SingleOperator, Operator) if act_by_type is None else act_by_type

        

        self.rep = set([self.single_type(operator_tup=op_tup, coefficient=op_c) for op_tup, op_c in self.operator.items()])
        if len(self.rep) == 0:
            self.rep = 0

    def get_dict(self):
        dict_form = { list(c.rep.keys())[0]: list(c.rep.values())[0] for c in self.rep}
        return dict_form
    

    def __hash__(self) -> int:
        return hash(tuple(self.rep))

    def __len__(self):
        return len(self.rep)

    def __add__(self, other):
        
        assert type(other) in (type(self), self.single_type) or other == 0, f'{type(self)} cannot be added to {type(other)}'
        
        if isinstance(other, Operator):
            fo1 = tuple(self.rep)
            fo2 = tuple(other.rep)
            new_op = type(self)(*(fo1 + fo2))

        elif isinstance(other, SingleOperator):
            fo1 = tuple(self.rep)
            fo2 = (other,)
            new_op = type(self)(*(fo1 + fo2))

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
        return type(self)(*new_rep)

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return  - self + other
    
    def __mul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        new_rep = map(lambda x: other * x, self.rep)
        return type(self)(*new_rep)

    def __rmul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return self * other
    
    def __truediv__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return 1/other * self
    
    def __matmul__(self, other):
        assert isinstance(other, self.act_on_type), f'{type(self)} cannot act on {type(other)}'
        
        if isinstance(other, SingleOperator):
            new_rep = map(lambda x: x @ other, self.rep)
            return type(self)(*new_rep)

        elif isinstance(other, Operator):
            new_rep = [op1 @ op2 for op1, op2 in product(self.rep, other.rep)]
            return type(self)(*new_rep)
        
        elif isinstance(other, (SingleKet, Ket)):
            return other.__rmatmul__(self)
    
    def __rmatmul__(self, other):
        assert isinstance(other, SingleOperator) or isinstance(other, Operator), f'{type(self)} cannot act on {type(other)}'
        
        if isinstance(other, SingleOperator):
            new_rep = map(lambda x: other @ x, self.rep)
            return type(self)(*new_rep)
        
        else:
            return other @ self
    

    def __repr__(self) -> str:
        if self.rep != 0:
            str_rep = ''
            for s in self.rep:
                str_rep += str(s)
                str_rep += '\n'

            return str_rep
        else:
            return '0'
    
    def __str__(self) -> str:
        return self.__repr__()


class SingleKet:
    
    def __init__(self, label_tup: tuple , coefficient: complex = 1, operator = None,
                 add_to_type = None,
                 act_by_type = None,) -> None:
        
        if operator is None:
            operator = SingleOperator(('I',), is_identity=True)

        assert isinstance(operator, SingleOperator) or operator is None, 'Operator must be an instance of SingleOperator or None'
        
        self.label_tup, self.coefficient, self.operator = self.redistribute_coefficient(label_tup=label_tup, coeff=coefficient, operator=operator)

        self.rep = {(self.operator, self.label_tup): self.coefficient}

        self.add_to_type = Ket if add_to_type is None else add_to_type
        self.act_by_type = (Operator, SingleOperator) if act_by_type is None else act_by_type


    def redistribute_coefficient(self, label_tup, coeff, operator):
        
        if isinstance(operator, SingleOperator):
            c_op = operator.coefficient
            c_in_ket = coeff
            c = c_op * c_in_ket
            
            op_tup = operator.operator_tup

            return label_tup, c, type(operator)(operator_tup=op_tup, coefficient=1, is_identity=operator.is_identity)
        
        else:
            return label_tup, coeff, operator



    def __hash__(self) -> int:
        return hash(tuple(self.rep.items()))

    def __eq__(self, other) -> bool:
        assert isinstance(other, SingleKet), f'{type(self)} can only be compared to {type(self)}.'
        return hash(self) == hash(other)
    

    # Basic Arithmetics
    def __add__(self, other):
        
        assert type(other) in (type(self), self.add_to_type) or other == 0, f'{(type(self))} cannot be added to {type(other)}'
        
        if isinstance(other, SingleKet):
            if other.label_tup == self.label_tup and self.operator == other.operator:
                c = self.coefficient + other.coefficient
                return type(self)(label_tup = self.label_tup, coefficient = c, operator = self.operator)
            else:
                return self.add_to_type(self, other)
        
        if isinstance(other, self.add_to_type):
            return other + self
        
        if other == 0:
            return self


    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        c = - self.coefficient
        return type(self)(label_tup=self.label_tup, coefficient=c, operator = self.operator)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not multiplied by a numeric value'
        c = self.coefficient * other
        return type(self)(label_tup=self.label_tup, coefficient=c, operator = self.operator)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not divided by numeric value'
        return 1/other * self

    
    def __rmatmul__(self, other):
        #type_assertion = reduce(lambda b1, b2: b1 or b2, [isinstance(other, t) for t in self.act_by_type])
        assert isinstance(other, self.act_by_type), f'{type(other)} cannot act on {type(self)}'

        new_operator = other @ self.operator

        if isinstance(new_operator, SingleOperator):
            return type(self)(label_tup= self.label_tup, coefficient=self.coefficient, operator=new_operator)
        
        if isinstance(new_operator, Operator):
            f_ket = lambda op: type(self)(label_tup= self.label_tup, coefficient=self.coefficient, operator=op)
            ket = map(f_ket, new_operator.rep)
            return self.add_to_type(*ket)
        
    def __repr__(self) -> str:
        ((op, label), coeff) = list(self.rep.items())[0]
        return f'{coeff} * {op.operator_tup} @ {label}'
    
    def __str__(self) -> str:
        return self.__repr__()



class Ket:
    
    def __init__(self, *SingleKets, 
                 single_type = None,
                 act_by_type = None,
                 ) -> None:
        
        self.single_kets = SingleKets
        self.dict_adder = lambda d1, d2: {k: v for k in set(d1)|set(d2) if (v := d1.get(k, 0) + d2.get(k, 0)) != 0}
        
        single_kets = map(lambda ket: ket.rep, SingleKets)
        self.ket = reduce(self.dict_adder, single_kets)

        self.single_type = SingleKet if single_type is None else single_type
        self.act_by_type = (SingleOperator, Operator) if act_by_type is None else act_by_type

        self.rep = set([self.single_type(label_tup=label_tup, coefficient=c, operator=op) for ((op, label_tup), c) in self.ket.items()])
        if len(self.rep) == 0:
            self.rep = 0

    def get_dict(self):
        dict_form = { list(c.rep.keys())[0]: list(c.rep.values())[0] for c in self.rep}
        return dict_form
    

    def __hash__(self) -> int:
        return hash(tuple(self.rep))

    def __len__(self):
        return len(self.rep)

    def __add__(self, other):
        
        assert type(other) in (type(self), self.single_type) or other == 0, f'{type(self)} cannot be added to {type(other)}'
        
        if isinstance(other, Ket):
            fo1 = tuple(self.rep)
            fo2 = tuple(other.rep)
            new_ket = type(self)(*(fo1 + fo2))

        elif isinstance(other, SingleKet):
            fo1 = tuple(self.rep)
            fo2 = (other,)
            new_ket = type(self)(*(fo1 + fo2))

        elif other == 0:
            new_ket = self
        
        else:
            pass
        
        """---------------------------------------------------------"""
        if len(new_ket.rep) == 0:
            return 0
        
        elif len(new_ket.rep) == 1:
            return list(new_ket.rep)[0]
        
        else:
            return new_ket


    def __radd__(self, other):
        return self + other

    def __neg__(self):
        new_rep = map(lambda x: -x, self.rep)
        return type(self)(*new_rep)

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return  - self + other
    
    def __mul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        new_rep = map(lambda x: other * x, self.rep)
        return type(self)(*new_rep)

    def __rmul__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return self * other
    
    def __truediv__(self, other):
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return 1/other * self
    

    def __rmatmul__(self, other):
        assert isinstance(other, SingleOperator) or isinstance(other, Operator), f'{type(self)} cannot act on {type(other)}'
        
        if isinstance(other, SingleOperator):
            new_rep = map(lambda x: other @ x, self.rep)
            return type(self)(*new_rep)
        
        else:
            new_rep = (sop @ sket for sop, sket in product(other.rep, self.rep))
            return type(self)(*new_rep)
    

    def __repr__(self) -> str:
        str_rep = ''
        for s in self.rep:
            str_rep += str(s)
            str_rep += '\n'

        return str_rep
    
    def __str__(self) -> str:
        return self.__repr__()



if __name__ == '__main__':
    
    sop1 = SingleOperator(operator_tup = ((0, 1),), coefficient = -3.5j)
    sop2 = SingleOperator(operator_tup = ((0, 0),), coefficient = 27j)
    sket1 = SingleKet((1, 0, 0, 0), 1j)
    sket2 = SingleKet((1, 1, 0, 0), 8)
    print((sop1 + sop2) @ (sket1 - 3j * sket2))
