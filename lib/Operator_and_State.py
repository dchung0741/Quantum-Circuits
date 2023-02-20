from functools import reduce
from itertools import product
from numpy import array, eye, kron
from math import prod
from typing_extensions import Self, TypeAlias
from typing import Union, Collection, Optional, Iterable, Iterator, Type, Callable, Any, cast, overload

Numerics: TypeAlias = Union[complex, float, int]
OperatorType: TypeAlias = Union["SingleOperator", "Operator"]
KetType: TypeAlias = Union["SingleKet", "Ket"]
LabelTuple: TypeAlias = tuple[Union[str, int], ...]

class SingleOperator:
    
    __slots__ = ('operator_tup', 'coefficient', 'is_identity', 'rep', 'add_to_type', 'act_on_type', 'act_by_type')

    def __new__(cls: type["SingleOperator"], 
                 operator_tup: tuple[tuple[Any], ...], coefficient: Optional[Numerics] = 1, is_identity: Optional[bool] = False,
                 add_to_type: Optional[type] = None,
                 act_on_type: Optional[tuple[type]] = None,
                 act_by_type: Optional[tuple[type]] = None) -> "SingleOperator":
        
        if coefficient == 0:
            return cast(Self, 0.)
        else:
            return super(SingleOperator, cls).__new__(cls)

    def __init__(self, operator_tup: tuple[tuple[Any], ...], coefficient: Numerics = 1, is_identity: bool = False,
                 add_to_type: Optional[type] = None,
                 act_on_type: Optional[tuple[type]] = None,
                 act_by_type: Optional[tuple[type]] = None,
                 ) -> None:
        
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

    def __eq__(self, other: "SingleOperator") -> bool: # type: ignore
        assert isinstance(other, SingleOperator), 'SingleFermionicOperator can only be compared to SingleFermionicOperator'
        return hash(self) == hash(other)
    
    # Basic Arithmetics
    
    # Addition  
    @overload
    def __add__(self, other: Numerics) -> "Self": ...
    @overload
    def __add__(self, other: OperatorType) -> Union[OperatorType, Numerics]: ...

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
        
        assert False

    @overload
    def __radd__(self, other: Numerics) -> "Self": ...
    @overload
    def __radd__(self, other: OperatorType) -> Union[OperatorType, Numerics]: ...
    
    def __radd__(self, other):
        return self + other
    
    # Negation
    def __neg__(self) -> "SingleOperator":
        c = - self.coefficient
        return type(self)(operator_tup=self.operator_tup, coefficient=c)
    
    # Subtraction
    @overload
    def __sub__(self, other: Numerics) -> "Self": ...
    @overload
    def __sub__(self, other: OperatorType) -> Union[OperatorType, Numerics]: ...
    
    def __sub__(self, other: Union[OperatorType, Numerics]) -> Union[OperatorType, Numerics]:
        return self + (-other)
    
    @overload
    def __rsub__(self, other: Numerics) -> "Self": ...
    @overload
    def __rsub__(self, other: OperatorType) -> Union[OperatorType, Numerics]: ...
    
    def __rsub__(self, other):
        return other + (-self)

    # Multiplication
    def __mul__(self, other: Numerics) -> Union[Self, Numerics]:
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not multiplied by a numeric value'
        c = self.coefficient * other
        return type(self)(operator_tup=self.operator_tup, coefficient=c, is_identity = self.is_identity)
    
    def __rmul__(self, other: Numerics) -> Union[Self, Numerics]:
        return self * other
    
    # Devision
    def __truediv__(self, other: Numerics) -> Union[Self, Numerics]:
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not divided by numeric value'
        return 1/other * self
    
    # Matrix Multiplication
    @overload
    def __matmul__(self, other: OperatorType) -> OperatorType: ...
    @overload
    def __matmul__(self, other: "SingleKet") -> "SingleKet": ...
    @overload
    def __matmul__(self, other: "Ket") -> "Ket": ...

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
        
    def __rmatmul__(self, other: OperatorType) -> OperatorType:
        return cast(OperatorType, other @ self)
    
    # Representation
    def __repr__(self) -> str:
        return str(self.rep)[1:-1]
    
    def __str__(self) -> str:
        return self.__repr__()


class Operator:

    __slots__ = ('rep', 'single_type', 'act_on_type', 'act_by_type')

    def __new__(cls: type["Operator"], *single_operators: SingleOperator, 
                 single_type: Optional[type] = None,
                 act_on_type: Optional[tuple[type]] = None,
                 act_by_type: Optional[tuple[type]] = None,) -> "Operator":
        
        dict_adder = lambda d1, d2: {k: v for k in set(d1)|set(d2) if (v := d1.get(k, 0) + d2.get(k, 0)) != 0}
        single_ops = map(lambda op: op.rep, single_operators)
        operator = reduce(dict_adder, single_ops)
        single_type = SingleOperator if single_type is None else single_type
        new_op_list: list[Type[SingleOperator]] = [single_type(operator_tup=op_tup, coefficient=op_c) for op_tup, op_c in operator.items()]

        if len(new_op_list) == 0:
            return cast(Operator, 0)
        
        elif len(new_op_list) == 1:
            return cast(Operator, new_op_list[0])
        
        else:
            # Solution provided in 
            # https://stackoverflow.com/questions/54358665/python-set-attributes-during-object-creation-in-new
            new_op = super().__new__(cls)
            object.__setattr__(new_op, 'rep', set(new_op_list))
            return new_op

    def __init__(self, *single_operators: SingleOperator, 
                 single_type: Optional[type] = None,
                 act_on_type: Optional[tuple[type]] = None,
                 act_by_type: Optional[tuple[type]] = None,
                 ) -> None:
        
        
        self.single_type = SingleOperator if single_type is None else single_type
        self.act_on_type = (SingleOperator, Operator, Ket, SingleKet) if act_on_type is None else act_on_type
        self.act_by_type = (SingleOperator, Operator) if act_by_type is None else act_by_type
        
        self.rep: set[SingleOperator]
    
    def __hash__(self) -> int:
        return hash(tuple(self.rep))

    def __len__(self) -> int:
        return len(self.rep)

    # Basic Arithmetic
    
    # Addition
    @overload
    def __add__(self, other: Numerics) -> Self: ...
    @overload
    def __add__(self, other: OperatorType) -> Union[OperatorType, Numerics]: ...

    def __add__(self, other):
        
        assert type(other) in (type(self), self.single_type) or other == 0, f'{type(self)} cannot be added to {type(other)}'
        
        if isinstance(other, Operator):
            op1 = tuple(self.rep)
            op2 = tuple(other.rep)
            new_op = type(self)(*(op1 + op2))

        elif isinstance(other, SingleOperator):
            op1 = tuple(self.rep)
            op2 = (other,)
            new_op = type(self)(*(op1 + op2))

        elif other == 0:
            new_op = self
        
        return new_op

    @overload
    def __radd__(self, other: Numerics) -> Self: ...
    @overload
    def __radd__(self, other: OperatorType) -> Union[OperatorType, Numerics]: ...

    def __radd__(self, other):
        return self + other

    # Negation
    def __neg__(self) -> Self:
        new_rep = map(lambda x: -x, self.rep)
        return type(self)(*new_rep)

    # Subtraction
    def __sub__(self, other: Union[OperatorType, Numerics]) -> Union[OperatorType, Numerics]:
        return self + (-other)
    
    def __rsub__(self, other: Union[OperatorType, Numerics]) -> Union[OperatorType, Numerics]:
        return  - self + other
    
    # Multiplication
    def __mul__(self, other: Numerics) -> Union[Self, Numerics]:
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        new_rep = cast(Iterator[SingleOperator], map(lambda x: other * x, self.rep))
        return type(self)(*new_rep)

    def __rmul__(self, other: Numerics) -> Union[Self, Numerics]:
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return self * other
    
    # Division
    def __truediv__(self, other: Numerics) -> Union[Self, Numerics]:
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return 1/other * self
    
    # matrix multiplication
    @overload
    def __matmul__(self, other: OperatorType) -> OperatorType: ...

    @overload
    def __matmul__(self, other: KetType) -> "Ket": ...

    def __matmul__(self, other):
        assert isinstance(other, self.act_on_type), f'{type(self)} cannot act on {type(other)}'
        
        if isinstance(other, SingleOperator):
            new_rep = cast(Iterator[SingleOperator], map(lambda x: x @ other, self.rep))
            return type(self)(*new_rep)

        elif isinstance(other, Operator):
            new_rep = cast(Iterator[SingleOperator], [op1 @ op2 for op1, op2 in product(self.rep, other.rep)])
            return type(self)(*new_rep)
        
        elif isinstance(other, (SingleKet, Ket)):
            return other.__rmatmul__(self)
    
    def __rmatmul__(self, other: OperatorType) -> OperatorType:
        assert isinstance(other, SingleOperator) or isinstance(other, Operator), f'{type(self)} cannot act on {type(other)}'
        
        if isinstance(other, SingleOperator):
            new_rep = cast(Iterator[SingleOperator], map(lambda x: other @ x, self.rep))
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

    __slots__ = ('label_tup', 'coefficient', 'operator', 'rep', 'add_to_type', 'act_on_type', 'act_by_type')

    def __new__(cls: type["SingleKet"], 
                label_tup: LabelTuple, 
                coefficient: Optional[Numerics] = 1, 
                operator: Optional[SingleOperator] = None,
                add_to_type: Optional[type["Ket"]] = None,
                act_by_type: Optional[tuple[OperatorType, ...]] = None) -> "SingleKet":
        
        if coefficient == 0:
            return cast(Self, 0.)
        else:
            return super(SingleKet, cls).__new__(cls)
    
    def __init__(self, 
                label_tup: LabelTuple, 
                coefficient: Numerics = 1, 
                operator: Optional[SingleOperator] = None,
                add_to_type: Optional[type["Ket"]] = None,
                act_by_type: Optional[tuple[OperatorType, ...]] = None,) -> None:
        
        if operator is None:
            operator = SingleOperator((('I',),), is_identity=True)

        assert isinstance(operator, SingleOperator) or operator is None, 'Operator must be an instance of SingleOperator or None'
        
        self.label_tup: LabelTuple 
        self.coefficient: Numerics 
        self.operator: SingleOperator
        
        self.label_tup, self.coefficient, self.operator = self.redistribute_coefficient(label_tup=label_tup, coeff=coefficient, operator=operator)

        self.rep: dict[tuple[SingleOperator, LabelTuple], Numerics] = {(self.operator, self.label_tup): self.coefficient}

        self.add_to_type: type[Ket] = cast(type, Ket) if add_to_type is None else add_to_type
        self.act_by_type: tuple[OperatorType, ...] = cast(tuple[OperatorType, ...], (Operator, SingleOperator)) if act_by_type is None else act_by_type

    def redistribute_coefficient(self, label_tup: LabelTuple, coeff: Numerics, operator: SingleOperator) -> tuple[LabelTuple, Numerics, SingleOperator]:
        
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

    def __eq__(self, other: "SingleKet") -> bool: # type: ignore
        assert isinstance(other, SingleKet), f'{type(self)} can only be compared to {type(self)}.'
        return hash(self) == hash(other)
    

    # Basic Arithmetics

    # Addition
    @overload
    def __add__(self, other: Numerics) -> "Self": ...
    
    @overload
    def __add__(self, other: KetType) -> Union[KetType, Numerics]: ...
    
    def __add__(self, other):
        
        assert type(other) in (type(self), self.add_to_type) or other == 0, f'{type(self)} cannot be added to {other}'
        
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

    @overload
    def __radd__(self, other: Numerics) -> "Self": ...
    
    @overload
    def __radd__(self, other: KetType) -> Union[KetType, Numerics]: ...

    def __radd__(self, other):
        return self + other

    # Negation    
    def __neg__(self) -> "Self":
        c = - self.coefficient
        return type(self)(label_tup=self.label_tup, coefficient=c, operator = self.operator)
    
    # Subtraction
    @overload
    def __sub__(self, other: Numerics) -> "Self": ...
    
    @overload
    def __sub__(self, other: KetType) -> Union[KetType, Numerics]: ...
    
    def __sub__(self, other):
        return self + (-other)
    
    @overload
    def __rsub__(self, other: Numerics) -> "Self": ...
    
    @overload
    def __rsub__(self, other: KetType) -> Union[KetType, Numerics]: ...
    
    def __rsub__(self, other):
        return other + (-self)

    # Multiplication
    def __mul__(self, other: Numerics) -> Union[Self, Numerics]:
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not multiplied by a numeric value'
        c: Numerics = self.coefficient * other
        return type(self)(label_tup=self.label_tup, coefficient=c, operator = self.operator)
    
    def __rmul__(self, other: Numerics) -> Union[Self, Numerics]:
        return self * other
    
    # Division
    def __truediv__(self, other: Numerics) -> Union[Self, Numerics]:
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not divided by numeric value'
        return 1/other * self
    
    # Matrix Multiplication
    @overload
    def __rmatmul__(self, other: SingleOperator) -> "Self": ...

    @overload
    def __rmatmul__(self, other: Operator) -> "Ket": ...

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

    __slot__ = ('rep', 'single_type', 'act_by_type')

    def __new__(cls: type["Ket"], *single_kets: SingleKet, 
                single_type: Optional[type] = None,
                act_by_type: Optional[tuple[type]] = None,) -> "Ket":
        
        dict_adder: Callable[[dict, dict], dict] = lambda d1, d2: {k: v for k in set(d1)|set(d2) if (v := d1.get(k, 0) + d2.get(k, 0)) != 0}
        single_kets_ = map(lambda op: op.rep, single_kets) 
        ket = reduce(dict_adder, single_kets_)
        single_type = SingleKet if single_type is None else single_type
        new_ket_list: list[Type[SingleKet]] = [single_type(label_tup=ket_tup, coefficient=ket_c, operator=ket_op) for (ket_op, ket_tup), ket_c in ket.items()]
        
        if len(new_ket_list) == 0:
            return cast(Ket, 0)
        
        elif len(new_ket_list) == 1:
            return cast(Ket, new_ket_list[0])
        
        else:
            # Solution provided in 
            # https://stackoverflow.com/questions/54358665/python-set-attributes-during-object-creation-in-new
            new_ket = super().__new__(cls)
            object.__setattr__(new_ket, 'rep', set(new_ket_list))
            return new_ket
    
    def __init__(self, *SingleKets: SingleKet, 
                 single_type: Optional[type] = None,
                 act_by_type: Optional[tuple[type]] = None,
                 ) -> None:
        

        self.single_type = SingleKet if single_type is None else single_type
        self.act_by_type = (SingleOperator, Operator) if act_by_type is None else act_by_type

        self.rep: set[SingleKet]

    def __hash__(self) -> int:
        return hash(tuple(self.rep))

    def __len__(self):
        return len(self.rep)

    # Arithmetics

    # Addition
    @overload
    def __add__(self, other: Numerics) -> "Self": ...
    
    @overload
    def __add__(self, other: KetType) -> Union[KetType, Numerics]: ...

    def __add__(self, other):
        
        assert type(other) in (type(self), self.single_type) or other == 0, f'{self} cannot be added to {other}'
        
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
        
        return new_ket

    @overload
    def __radd__(self, other: Numerics) -> "Self": ...
    
    @overload
    def __radd__(self, other: KetType) -> Union[KetType, Numerics]: ...

    def __radd__(self, other):
        return self + other

    # Negation
    def __neg__(self) -> "Self":
        new_rep = map(lambda x: -x, self.rep)
        return type(self)(*new_rep)

    # Subtraction
    @overload
    def __sub__(self, other: Numerics) -> "Self": ...
    
    @overload
    def __sub__(self, other: KetType) -> Union[KetType, Numerics]: ...

    def __sub__(self, other):
        return self + (-other)
    
    @overload
    def __rsub__(self, other: Numerics) -> "Self": ...
    
    @overload
    def __rsub__(self, other: KetType) -> Union[KetType, Numerics]: ...

    def __rsub__(self, other):
        return  - self + other
    

    # Multiplication
    def __mul__(self, other: Numerics) -> Union[Self, Numerics]:
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        new_rep = cast(Iterator[SingleKet], map(lambda x: other * x, self.rep))
        return type(self)(*new_rep)

    def __rmul__(self, other: Numerics) -> Union[Self, Numerics]:
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return self * other
    
    # Division
    def __truediv__(self, other: Numerics) -> Union[Self, Numerics]:
        assert isinstance(other, complex) or isinstance(other, float) or isinstance(other, int), 'Not numeric value'
        return 1/other * self
    
    # Matrix Multuplication
    @overload
    def __rmatmul__(self, other: SingleOperator) -> Self: ...

    @overload
    def __rmatmul__(self, other: Operator) -> "Ket": ...

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
    
    # sop1 = SingleOperator(operator_tup = ((0, 1),), coefficient = -3.5j)
    # sop2 = SingleOperator(operator_tup = ((0, 0),), coefficient = 27j)
    # sket1 = SingleKet((1, 0, 0, 0), 1j)
    # sket2 = SingleKet((1, 1, 0, 0), 8)
    # # print((sop1 + sop2) @ (sket1 - 3j * sket2))
    # # print(Operator(sop1, 4*sop2))
    # # print(SingleOperator(operator_tup = ((0, 1),), coefficient = 0))
    # print(Operator(sop1, sop2))

    
    # class TestSinglePauliString(SingleOperator):
    #     ...
    # class TestPauliString(Operator):

    #     def __new__(cls: type[Self], *single_operators: SingleOperator) -> Self:
    #         return super().__new__(cls, *single_operators, single_type=TestSinglePauliString, act_by_type= (int, float), act_on_type=(int, float))

    #     def __init__(self, *single_operators: SingleOperator) -> None:
    #         super().__init__(*single_operators, single_type=TestSinglePauliString, act_by_type= (int, float), act_on_type=(int, float))
    

    # print(type(list(TestPauliString(TestSinglePauliString(('X',)), TestSinglePauliString(('Y',))).rep)[0]))
    
    sket1 = SingleKet((1, 0, 0, 0), 1j)
    sket2 = SingleKet((1, 1, 0, 0), 8)

    ket = Ket(sket1, sket2)
    print(sket1 - sket1)