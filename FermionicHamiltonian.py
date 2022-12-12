from HartreeFock import *
from FermionicOperator import*
from collections import Counter
from functools import reduce



class FermionicHamiltonian:

    def __init__(self, molecule_geometry: list, zeta_dict: dict) -> None:
        
        self.scf_result = HartreeFock(molecule_geometry=molecule_geometry, zeta_dict=zeta_dict)
        
        self.psi_AB_h = self.scf_result.psi_AB_h
        self.psi_AB_4_tensor = self.scf_result.psi_AB_4_tensor
        self.psi_RB_h = self.scf_result.psi_RB_h
        self.psi_RB_4_tensor = self.scf_result.psi_RB_4_tensor
        self.n = 2 * self.psi_AB_h.shape[0]

        self.fermionic_hamiltonian = self.get_fermionic_hamiltonian()
        self.jw_hamiltonian = self.get_jw_encoding()


    def get_fermionic_hamiltonian(self):
        res = 0
        for i in range(self.n):
            res += self.psi_AB_h[i//2, i//2].round(5) * SingleFermionicOperator(operator_tup=((i, 1), (i, 0)), coefficient=1)
        
        for i, j, k, l in product(*([range(self.n)]*4)):
            
            if (i % 2 == k % 2) and (j%2 == l%2):
                res += 0.5 * self.psi_AB_4_tensor[i//2, j//2, k//2, l//2].round(5) * SingleFermionicOperator(operator_tup=((i, 1), (j, 1), (k, 0), (l, 0)), coefficient=1)
            else:
                continue
        
        return res


    def get_jw_encoding(self):
        pass




if __name__ == '__main__':

    print('-------------------------------------------------------------------------------')
    H2_Hamiltonian = FermionicHamiltonian(molecule_geometry = [['H', [0, 0, 0]], ['H', [0, 0, 1.4]]], zeta_dict={'H': 1.24})
    print(H2_Hamiltonian.fermionic_hamiltonian)
    print(len(H2_Hamiltonian.fermionic_hamiltonian))
    print(H2_Hamiltonian.psi_AB_4_tensor.round(5)[0, 0, 0, 1])