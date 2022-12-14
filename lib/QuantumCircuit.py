from .PauliString import PauliString, SinglePauliString, toPauliBasis, SingleQuantumState
from typing import Dict
from numpy import array
from functools import reduce
from itertools import product

def generateQuantumGate(n: int, 
                        free_bits_dict: Dict[int, str] = {}, 
                        target_bits_dict: Dict[int, str] = {}, 
                        controlled_bits_dict: Dict[int, int] = {}, 
                        additional_operator_dict = None):
    
    identity = SinglePauliString((('I',)*n,), is_identity=1)
    additional_dict = {'P0': array([[1, 0], [0, 0]]), 'P1': array([[0, 0], [0, 1]])}
    additional_dict = additional_dict if additional_operator_dict is None else additional_dict|additional_operator_dict

    # free gate
    free_operator_tup = tuple([free_bits_dict[i] if i in free_bits_dict else 'I' for i in range(n)])  
    free_gate = SinglePauliString(operator_tup=(free_operator_tup,))

    # control bit dict
    
    controlled_bits_op_dict = {k: {0: 'P0', 1: 'P1'}[v] for k, v in controlled_bits_dict.items()}
    
    control_projector = tuple([controlled_bits_op_dict[i] if i in controlled_bits_op_dict else 'I' for i in range(n)])
    control_projector = SinglePauliString(operator_tup=(control_projector,))

    activated_operator = tuple([target_bits_dict[i] if i in target_bits_dict else 'I' for i in range(n)])
    activated_operator = SinglePauliString(operator_tup = (activated_operator,))

    control_projector_perp = identity - control_projector

    control_gate = control_projector_perp + activated_operator @ control_projector

    # construct gate
    gate = free_gate @ control_gate

    return gate.simplify(with_map=additional_dict)



def constructQuantumCircuit(*args):
    return reduce(lambda g1, g2: (g1 @ g2).simplify(), reversed(args))



if __name__ == '__main__':
    
    def print_title(f):

        def wrapper():
            func_name = f.__name__
            title_string = f'-------------------- {func_name} --------------------'
            print(title_string)
            f()
            print('-' * len(title_string))
        
        return wrapper
    
    # CNOT = generateQuantumGate(n = 2, free_bits_dict={}, target_bits_dict={1: 'X'}, controlled_bits_dict={0: 1})

    # print(CNOT)
    # print(CNOT.matrix_rep())

    from numpy import array, kron, pi, exp, sqrt, sin, cos, eye, zeros
    from functools import reduce

    X = array([[0, 1], [1, 0]])
    Y = array([[0, -1j], [1j, 0]])
    Z = array([[1, 0], [0, -1]])
    H = (X + Z)/sqrt(2)
    T = array([[1, 0], [0, exp(1j * pi/4)]])
    S = array([[1, 0], [0, 1j]])

    @print_title
    def Fredkin_from_Toffoli():
        Toffoli_c0c1_t2 = generateQuantumGate(n = 3, free_bits_dict={}, target_bits_dict={2: 'X'}, controlled_bits_dict={0: 1, 1: 1})
        Toffoli_c0c2_t1 = generateQuantumGate(n = 3, free_bits_dict={}, target_bits_dict={1: 'X'}, controlled_bits_dict={0: 1, 2: 1})
        Toffoli_c1c2_t0 = generateQuantumGate(n = 3, free_bits_dict={}, target_bits_dict={0: 'X'}, controlled_bits_dict={1: 1, 2: 1})

        print(Toffoli_c0c1_t2)
        print(Toffoli_c0c1_t2.matrix_rep())

        print(Toffoli_c0c1_t2.simplify())
        print(Toffoli_c0c2_t1.simplify())
        print(Toffoli_c0c1_t2.matrix_rep() @ Toffoli_c0c2_t1.matrix_rep() @ Toffoli_c0c1_t2.matrix_rep())

        Fredkin = Toffoli_c0c1_t2 @ Toffoli_c0c2_t1 @ Toffoli_c0c1_t2
        print(Fredkin)
        print(Fredkin.simplify())
        print(Fredkin.simplify().matrix_rep())
    
    

    @print_title
    def QuantumFourierTransform():
        G1 = generateQuantumGate(n = 3, free_bits_dict={0: 'H'}, target_bits_dict={}, controlled_bits_dict={}, additional_operator_dict={'H': H, 'S': S, 'T': T})
        G2 = generateQuantumGate(n = 3, free_bits_dict={}, target_bits_dict={0: 'S'}, controlled_bits_dict={1: 1}, additional_operator_dict={'H': H, 'S': S, 'T': T})
        G3 = generateQuantumGate(n = 3, free_bits_dict={}, target_bits_dict={0: 'T'}, controlled_bits_dict={2: 1}, additional_operator_dict={'H': H, 'S': S, 'T': T})
        G4 = generateQuantumGate(n = 3, free_bits_dict={1: 'H'}, target_bits_dict={}, controlled_bits_dict={}, additional_operator_dict={'H': H, 'S': S, 'T': T})
        G5 = generateQuantumGate(n = 3, free_bits_dict={}, target_bits_dict={1: 'S'}, controlled_bits_dict={2: 1}, additional_operator_dict={'H': H, 'S': S, 'T': T})
        G6 = generateQuantumGate(n = 3, free_bits_dict={2: 'H'}, target_bits_dict={}, controlled_bits_dict={}, additional_operator_dict={'H': H, 'S': S, 'T': T})
        
        swap = SinglePauliString(operator_tup=(('I', 'I', 'I'),), coefficient=0.5) 
        print(swap.simplify())

        swap = swap + SinglePauliString(operator_tup=(('X', 'I', 'X'),), coefficient=0.5) 
        swap = swap + SinglePauliString(operator_tup=(('Y', 'I', 'Y'),), coefficient=0.5) 
        swap = swap + SinglePauliString(operator_tup=(('Z', 'I', 'Z'),), coefficient=0.5) 
        
        # print(G1)
        # print(G2)
        # print(G3)
        # print(G4)
        # print(G5)
        # print(G6)
        
        FT = reduce(lambda x, y: x @ y, reversed([G1, G2, G3, G4, G5, G6]))
        FT = sqrt(8) * swap @ FT.simplify()
        print(FT)
        print(FT.simplify())
        print(FT.simplify().matrix_rep().round(3))
        print(FT.matrix_rep().round(3))


    @print_title
    def generate_entangle():

        g1 = generateQuantumGate(n = 2, free_bits_dict={0: 'H'}, target_bits_dict={}, controlled_bits_dict={}, additional_operator_dict={'H': H})
        g2 = generateQuantumGate(n = 2, free_bits_dict={}, target_bits_dict={1: 'X'}, controlled_bits_dict={0: 1}, additional_operator_dict={'H': H})
        circuit = reduce(lambda g1, g2: g1 @ g2, reversed([g1, g2]))
        initial_state = SingleQuantumState(label_tup=(0, 0))
        out_state = circuit @ initial_state
        print(circuit)
        print(out_state.evaluate())

    

    @print_title
    def CZ():

        circuit = generateQuantumGate(n = 2, free_bits_dict={}, target_bits_dict={1: 'Z'}, controlled_bits_dict={0: 1}, additional_operator_dict={})  
        initial_state_1 = SingleQuantumState(label_tup=(0, 0))
        out_state_1 = circuit @ initial_state_1

        initial_state_2 = SingleQuantumState(label_tup=(1, 1))
        out_state_2 = circuit @ initial_state_2

        initial_state_3 = SingleQuantumState(label_tup=(1, 0), coefficient=1/sqrt(2)) + SingleQuantumState(label_tup=(1, 1), coefficient=1/sqrt(2))
        out_state_3 = circuit @ initial_state_3
        print(circuit)
        
        print('out_state_1:\n', out_state_1.evaluate())
        print('out_state_2:\n', out_state_2.evaluate())
        print('out_state_3:\n', out_state_3.evaluate())
    

    @print_title
    def Clifford():
        CNOT = generateQuantumGate(n = 2, free_bits_dict={}, target_bits_dict={1: 'X'}, controlled_bits_dict={0: 1})
        XI = generateQuantumGate(n = 2, free_bits_dict={0: 'X', 1: 'I'}, target_bits_dict={}, controlled_bits_dict={})
        print((CNOT @ XI @ CNOT).simplify())
    # Fredkin_from_Toffoli()
    QuantumFourierTransform()
    # generate_entangle()
    # CZ()
    # Clifford()
    # Deutsch()
    # Bernstein_Vazirani()