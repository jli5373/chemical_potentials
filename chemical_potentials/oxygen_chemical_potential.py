import numpy as np
import matplotlib.pyplot as plt
import math
import general_functions as gf

display_constants=True
# Define constants
kb = 8.6173303e-5*1.6022e-19 # Boltzmann constant in J/K
h = 4.135667662e-15*1.6022e-19 # Planck constant in J*s
h_bar = h/(2*math.pi) # Reduced Planck constant in J*s
pi = math.pi # Pi
p_0 = 1.01325e5 # Pressure in Pa (1 atm)
N_avogadro = 6.022140857e23 # Avogadro's number
m = 2.6566962e-26*2 #mass of single O2 molecule in kg
sigma_sym = 2 # Symmetry number of O2 molecule (2 for homonuclear diatomic, 1 for heteronuclear diatomic)
I_spin = 3 # Electronic spin degeneracy of O2 molecule
B_0 = 0.18/1000*1.6022e-19 # Rotational constant in J
omega_0 = 4.739261388e13 # s^-1; tabulated at: 0.196 eV; corresponds to 4.739261388e13 Hz frequency


def oxygen_delta_mu_0(T,p=p_0):
    '''
    Calculates the delta chemical potential of oxygen in eV, given a temperature in K and pressure in Pa.
    Will return 0 if T is input as 0
    Inputs
    ------
        T: float
            Temperature in K
        p: float
            Pressure in Pa, default is 1 atm (1.01325e5 Pa)
    Returns
    -------
        mu_O: float
            Chemical potential of oxygen (1/2 O2) in eV
    '''
    if T == 0:
        return 0
    # Define partition function components
    q_trans=(2*pi*m/(h**2))**1.5*((kb*T)**2.5)/p #technically this is more like q_trans-pV term
    #q_rot = sum{J=0->infinite}[(2J+1)exp(-J(J+1)B_0/kb*T)]
    #after Euler-Maclaurin approx, and an approximation where we group complications due to indistinguishable particles into the sigma_sym factor, we get: (at relevant temperatures where rotational level spacing <<kb*T)
    #mu_rot=-kb*T*math.log(kb*T/(sigma_sym*B_0))
    #q_vib = sum{i=1->M,n=0->infinite}[exp(-(n+0.5)h_bar*omega_i/(kb*T))] #M is number of vibrational modes, omega_i for the molecule
    #evaluating
    #mu_vib=sum{i=1->M}[h_bar*omega_i/(kb*T)+kb*T*math.log(1-exp(-h_bar*omega_i/(kb*T)))]
    #mu_elec=math.log(I_spin) #I_spin is the electronic spin degeneracy of the molecule

    #delta_mu_O2 = -h*omega_0/(kb*T)
    #delta_mu_O2 = -kb*T*(math.log(q_trans) + math.log(kb*T/(sigma_sym*B_0)) - math.log(1-math.exp(-h*omega_0/(kb*T))) + math.log(I_spin) )# Oxygen chemical potential in J, single molecule
    delta_mu_O2 = -kb*T*(math.log(q_trans) + math.log(kb*T/(sigma_sym*B_0)) + math.log(I_spin) )# vibrations disabled - Oxygen chemical potential in J, single molecule
    mu_O = delta_mu_O2/2
    return mu_O/1.602176634e-19 # converted to eV

def delta_mu_oxygen_vibrational_component(T):
    '''
    Calculates the vibrational component of the delta chemical potential of oxygen in eV, given a temperature in K.
    Will return 0 if T is input as 0
    Inputs
    ------
        T: float
            Temperature in K
    Returns
    -------
        delta_mu_O_vib: float
            vibrational component of delta chemical potential of oxygen (1/2 O2) in eV
    '''
    if T == 0:
        return 0
    return kb*T*math.log(1-math.exp(-h*omega_0/(kb*T)))/2/1.602176634e-19 # converted to eV

def oxygen_mu_0(T,p=p_0,calc_type="None"):
    '''
    Calculates the chemical potential of oxygen in eV, given a temperature in K and pressure in Pa.
    Inputs
    ------
        T: float
            Temperature in K
        p: float
            Pressure in Pa, default is 1 atm (1.01325e5 Pa)
        calc_type: string
            Calculation type, default is None (sets no offset). Other options are "PBE", "SCAN", "R2SCAN"
    Returns
    -------
        mu_O: float
            Chemical potential of oxygen (1/2 O2) in eV
    '''
    if (calc_type=="None"):
        vasp_mu_O = 0.0
    elif(calc_type=="PBE"):
        vasp_mu_O = (-9.87432174+0)/2 #PBE O2 molecule (not corrected by 1.36eV/O2 [following Ceder paper])
    elif(calc_type=="SCAN"):
        vasp_mu_O = -5.950312485 #SCAN O2 molecule/2 (uncorrected, will need to validate for SCAN)
    elif(calc_type=="R2SCAN"):
        vasp_mu_O = 0.0 #unimplemented for R2SCAN
        print("R2SCAN not yet implemented")
    #print("VASP reference chemical potential is: ", )
    mu_O = vasp_mu_O + oxygen_delta_mu_0(T,p)
    return mu_O
