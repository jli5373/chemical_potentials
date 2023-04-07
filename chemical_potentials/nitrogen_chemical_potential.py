import numpy as np
import matplotlib.pyplot as plt
import math
from chemical_potentials import general_functions as gf

display_constants=True
# Define constants
kb = 8.6173303e-5*1.6022e-19 # Boltzmann constant in J/K
h = 4.135667662e-15*1.6022e-19 # Planck constant in J*s
h_bar = h/(2*math.pi) # Reduced Planck constant in J*s
pi = math.pi # Pi
p_0 = 1.01325e5 # Pressure in Pa (1 atm)
N_avogadro = 6.022140857e23 # Avogadro's number
m = 2.3258671e-26*2 #mass of single N2 molecule in kg
sigma_sym = 2 # Symmetry number of N2 molecule (2 for homonuclear diatomic, 1 for heteronuclear diatomic)
I_spin = 1 # Electronic spin degeneracy of N2 molecule NOTE: ISPIN=1 for N2
B_0 = 3.97212135e-23 # Rotational constant in J, JANAF tables list B_e = 1.99825 cm^-1 (N2)
omega_0 = 7.07265e13 # s^-1; tabulated at: 0.196 eV; corresponds to xx Hz frequency; JANAF tables list w_e = 2357.55 cm^-1 (N2)


def _nitrogen_delta_mu_0(T,p=p_0):
    '''
    Calculates the delta chemical potential of nitrogen in eV, given a temperature in K and pressure in Pa.
    Will return 0 if T is input as 0
    Inputs
    ------
        T: float
            Temperature in K
        p: float
            Pressure in Pa, default is 1 atm (1.01325e5 Pa)
    Returns
    -------
        mu_N: float
            Chemical potential of nitrogen (1/2 N2) in eV
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

    #delta_mu_N2 = -h*omega_0/(kb*T)
    delta_mu_N2 = -kb*T*(math.log(q_trans) + math.log(kb*T/(sigma_sym*B_0)) - math.log(1-math.exp(-h*omega_0/(kb*T))) + math.log(I_spin) )# nitrogen chemical potential in J, single molecule
    #delta_mu_N2 = -kb*T*(math.log(q_trans) + math.log(kb*T/(sigma_sym*B_0)) + math.log(I_spin) )# vibrations disabled - nitrogen chemical potential in J, single molecule
    mu_N = delta_mu_N2/2
    return mu_N/1.602176634e-19 # converted to eV

def _delta_mu_nitrogen_vibrational_component(T):
    '''
    Calculates the vibrational component of the delta chemical potential of nitrogen in eV, given a temperature in K.
    Will return 0 if T is input as 0
    Inputs
    ------
        T: float
            Temperature in K
    Returns
    -------
        delta_mu_O_vib: float
            vibrational component of delta chemical potential of nitrogen (1/2 N2) in eV
    '''
    if T == 0:
        return 0
    return kb*T*math.log(1-math.exp(-h*omega_0/(kb*T)))/2/1.602176634e-19 # converted to eV

def nitrogen_mu_0(T,p=p_0,offset=0):
    '''
    Calculates the chemical potential of nitrogen in eV, given a temperature in K and pressure in Pa.
    Inputs
    ------
        T: float
            Temperature in K
        p: float
            Pressure in Pa, default is 1 atm (1.01325e5 Pa)
        Offset: float
            Value to offset the chemical potential by, in eV. Represents the 0K correction to the chemical potential based on reference (VASP) calculations.
    Returns
    -------
        mu_N: float
            Chemical potential of nitrogen (1/2 N2) in eV
    '''
    #NOTE: For Jonny's calculations (ENCUT=590eV, k-point mech density=45), PBE offset = (-16.64101840+0)/2
    #NOTE: For Jonny's calculations (ENCUT=590eV, k-point mech density=45), SCAN offset = #TODO: Update SCAN offset value
    
    mu_N = offset + _nitrogen_delta_mu_0(T,p)
    return mu_N
