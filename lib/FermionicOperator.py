from .Operator_and_State import SingleOperator, Operator, SingleKet, Ket
from .PauliString import SinglePauliString, PauliString
from functools import reduce
from itertools import product
from numpy import array, eye, kron
from math import prod


def jwEncodingMap(n: int):
    
    jw_dict = {}

    for i in range(n):
        cXi = tuple('Z' * i + 'X' + 'I' * (n - i - 1))
        cYi = tuple('Z' * i + 'Y' + 'I' * (n - i - 1))
        
        jw_dict[(i, 1)] = SinglePauliString(operator_tup= (cXi,), coefficient=0.5) - SinglePauliString(operator_tup= (cYi,), coefficient=0.5j)
        jw_dict[(i, 0)] = SinglePauliString(operator_tup= (cXi,), coefficient=0.5) + SinglePauliString(operator_tup= (cYi,), coefficient=0.5j)
    
    jw_dict = jw_dict|{'I': SinglePauliString(operator_tup = (tuple('I'*n),), coefficient=1, is_identity=True)}
    return jw_dict


class SingleFermionicOperator(SingleOperator):

    def __init__(self, operator_tup: tuple, coefficient: complex = 1, jw_map: dict = None, is_identity = False) -> None:
        
        super().__init__(operator_tup, coefficient, is_identity=is_identity,
                         add_to_type = FermionicOperator, 
                         act_on_type = (FermionicOperator, SingleFermionicOperator, SingleFermionicState, FermionicState), 
                         act_by_type = (FermionicOperator, SingleFermionicOperator))

        self.jw_map = jw_map


    def jw_encoding(self, with_map = None):
        # assert self.jw_map is not None, 'This method is not suitable for this operator'

        jw_map = self.jw_map if with_map is None else with_map

        jw_op = [jw_map[op] for op in self.operator_tup]
        jw_op = reduce(lambda x, y: x @ y, jw_op)
        jw_op = jw_op.simplify() * self.coefficient
        
        return jw_op


    def simplify(self):
        pass


class FermionicOperator(Operator):

    def __init__(self, *SingleFOs, jw_map = None) -> None:
        super().__init__(*SingleFOs, 
                         single_type = SingleFermionicOperator, 
                         act_on_type = (FermionicOperator, SingleFermionicOperator, SingleFermionicState, FermionicState), 
                         act_by_type = (FermionicOperator, SingleFermionicOperator))
        
        self.jw_map = jw_map

    
    def jw_encoding(self, with_map = None):
        jw_map = self.jw_map if with_map is None else with_map
        res = [ sfo.jw_encoding(with_map = jw_map) for sfo in self.rep]
        return sum(res)
        
        
            
class SingleFermionicState(SingleKet):

    def __init__(self, label_tup: tuple, coefficient: complex, operator=None) -> None:
        if operator is None:
            operator = SingleFermionicOperator(('I',), is_identity=True)
        super().__init__(label_tup, coefficient, operator = operator, 
                         add_to_type = FermionicOperator, 
                         act_by_type = (FermionicOperator, SingleFermionicOperator))


    def evaluate(self):
        pass

class FermionicState(Ket):

    def __init__(self, *SingleFermionicStates) -> None:
        super().__init__(*SingleFermionicStates, 
                         single_type=SingleFermionicState, 
                         act_by_type = (FermionicOperator, SingleFermionicOperator))



if __name__ == '__main__':

    sfo1 = SingleFermionicOperator(operator_tup = ((0, 0),), coefficient = -3.5j)
    sfo2 = SingleFermionicOperator(operator_tup = ((3, 0),), coefficient = 0.5j)
    sfo3 = SingleFermionicOperator(operator_tup = ((0, 1),), coefficient = 2)
    sfo4 = SingleFermionicOperator(operator_tup = ((1, 1),), coefficient = 2)

    fo1 = FermionicOperator(sfo1, sfo2, sfo3)
    fo2 = FermionicOperator(sfo1, sfo3, sfo4)
    fo3 = sfo4 @ fo1 @ sfo2 + sfo1 @ sfo1 - 3 * fo1 @ fo2

    print(type(sfo4 @ fo1 @ sfo2))
    print(type(sfo1 @ sfo1))
    print(type(- 3 * fo1))
    print(type(- (3 * fo1) @ fo2))
    print(type(sfo1 @ sfo1 - 3 * fo1 @ fo2))
    
    print('fo1', fo1)
    print('fo2', fo2)
    print(fo3)
    print(fo2 @ fo1)
    print(fo2.get_dict())
    
    print(hash(fo3))

    print(type(fo3))

    
    print('----------- Jordan Wigner ------------------')
    
    n = 5
    jw = jwEncodingMap(n)
    for i, d in product(range(n), range(2)):
        print((i, d))
        print(jw[(i, d)])

    op1 = jw[(1, 1)] @ jw[(1, 0)]
    op2 = jw[(1, 0)] @ jw[(1, 1)]
    
    print(((op1 + op2).simplify()))
    # sfo1 = SingleFermionicOperator(operator_tup = ((0, 0), (3, 1)), coefficient = -3.5j, jw_map=jw)
    
    
    print('sfo1', sfo1.jw_encoding(with_map=jw))
    print('sfo2', sfo2.jw_encoding(with_map=jw))
    print('sfo3', sfo3.jw_encoding(with_map=jw))
    
    print(fo1.jw_encoding(with_map = jw))

    print('------------Identity test---------------')
    
    sfo_id = SingleFermionicOperator(('I',), coefficient=3, is_identity=True)
    print(sfo1)
    print(sfo_id @ sfo1)
    
    print(sfo_id.jw_encoding(with_map=jw))

    print('fo1', fo1)
    print('id @ fo1', (sfo_id @ fo1))
    print('fo1 @ id', (fo1 @ sfo_id))

    xxx = [i for i in fo1.rep]
    print(xxx)

    sfs1 = SingleFermionicState((1, 0, 0, 0), 1j)
    print(type(sfs1.operator))