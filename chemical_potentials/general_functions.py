import numpy as np
import matplotlib.pyplot as plt
import math

N_avogadro = 6.022140857e23 # Avogadro's number

def load_janaf_csv(filename):
    '''
    Loads a JANAF .csv file and returns the data as a numpy array
    Inputs
    ------
        filename: string
            Name of the JANAF file to load
    Returns
    -------
        data: numpy array
            Data from the JANAF file
    '''
    print("Loading", filename)
    data = np.genfromtxt(filename, skip_header=2)
    data_dict = {}
    data_dict["T"] = data[:,0] #K
    data_dict["Cp"] = data[:,1] #J/mol*K
    data_dict["S"] = data[:,2] # J/mol*K
    data_dict["-[G-H(Tr)]/T"] = data[:,3] #J/mol*K
    data_dict["H-H(Tr)"] = data[:,4]*1000 # originally kJ/mol, converted to J/mol
    data_dict["delta-f_H"] = data[:,5]*1000 # originally kJ/mol, converted to J/mol Enthalpy of formation
    data_dict["delta-f G"] = data[:,6]*1000 # originally kJ/mol, converted to J/mol Gibbs free energy of formation
    data_dict["log_Kf"] = data[:,7]
    return data_dict

def gibbs_formation_energy_from_janaf(janaf_filename,T_min=0,T_max=9999):
    '''
        Retrieves the delta-f G (Gibbs formation energy) [kJ/mol] of the given JANAF file and returns it in eV/molecule.

        Inputs
        ------
            janaf_filename: string
                Name of the JANAF file to load
            T_min: float
                Minimum temperature in K, defaults to 0
            T_max: float
                Maximum temperature in K, defaults to 9999 (presumably you don't have data above 9999K)
        Returns
        -------
            delta_f_G: float
                Gibbs formation energy in eV/molecule
    '''
    janaf_data = load_janaf_csv(janaf_filename)
    # make sure lengths are all the same, print warning if not
    if len(janaf_data["T"]) != len(janaf_data["delta-f G"]):
        print("Warning: Lengths of data arrays are not equal!")
    # make sure T is in ascending order, print warning if not
    if not np.all(np.diff(janaf_data["T"]) > 0):
        print("Warning: Temperature array is not in ascending order!")
    # trim janaf data to relevant T range
    T = janaf_data["T"][(janaf_data["T"] >= T_min) & (janaf_data["T"] <= T_max)]
    delta_f_G = janaf_data["delta-f G"][(janaf_data["T"] >= T_min) & (janaf_data["T"] <= T_max)]
    delta_f_G = delta_f_G/N_avogadro/1.602176634e-19 # convert from J/mol to eV/molecule
    return delta_f_G, T

def formation_enthalpy_from_janaf(janaf_filename,T_min=0,T_max=9999):
    '''
        Retrieves the delta-f H (formation enthalpy) [kJ/mol] of the given JANAF file and returns it in eV/molecule.

        Inputs
        ------
            janaf_filename: string
                Name of the JANAF file to load
            T_min: float
                Minimum temperature in K, defaults to 0
            T_max: float
                Maximum temperature in K, defaults to 9999 (presumably you don't have data above 9999K)
        Returns
        -------
            delta_f_H: numpy array
                Formation enthalpy in eV/molecule (float)
            T: numpy array
    '''
    janaf_data = load_janaf_csv(janaf_filename)
    # make sure lengths are all the same, print warning if not
    if len(janaf_data["T"]) != len(janaf_data["delta-f_H"]):
        print("Warning: Lengths of data arrays are not equal!")
    # make sure T is in ascending order, print warning if not
    if not np.all(np.diff(janaf_data["T"]) > 0):
        print("Warning: Temperature array is not in ascending order!")
    # trim janaf data to relevant T range
    T = janaf_data["T"][(janaf_data["T"] >= T_min) & (janaf_data["T"] <= T_max)]
    delta_f_H = janaf_data["delta-f_H"][(janaf_data["T"] >= T_min) & (janaf_data["T"] <= T_max)]
    delta_f_H = delta_f_H/N_avogadro/1.602176634e-19 # convert from J/mol to eV/molecule
    return delta_f_H, T    

def calculate_mu_from_janaf(janaf_filename,T_min=0,T_max=9999):
    '''
        Takes janaf data and calculates chemical potential of the species for the specified T-range (ideally the relevant T-range)
        Executes (H-H(Tr)) - TS = G-H(Tr) and returns [G-H(Tr)]/N_avogadro

        Inputs
        ------
            janaf_filename: string
                Name of the JANAF file to load
            T_min: float
                Minimum temperature in K, defaults to 0
            T_max: float
                Maximum temperature in K, defaults to 9999 (presumably you don't have data above 9999K)
        Returns
        -------
            mu: numpy array
                Chemical potential in eV
            T: numpy array
                Temperature in K
    '''
    janaf_data = load_janaf_csv(janaf_filename)
    # make sure lengths are all the same, print warning if not
    if len(janaf_data["T"]) != len(janaf_data["Cp"]) or len(janaf_data["T"]) != len(janaf_data["S"]) or len(janaf_data["T"]) != len(janaf_data["-[G-H(Tr)]/T"]) or len(janaf_data["T"]) != len(janaf_data["H-H(Tr)"]):
        print("Warning: Lengths of data arrays are not equal!")
    # make sure T is in ascending order, print warning if not
    if not np.all(np.diff(janaf_data["T"]) > 0):
        print("Warning: Temperature array is not in ascending order!")
    # trim janaf data to relevant T range
    T = janaf_data["T"][(janaf_data["T"] >= T_min) & (janaf_data["T"] <= T_max)]
    Cp = janaf_data["Cp"][(janaf_data["T"] >= T_min) & (janaf_data["T"] <= T_max)]
    S = janaf_data["S"][(janaf_data["T"] >= T_min) & (janaf_data["T"] <= T_max)]
    delta_G = janaf_data["-[G-H(Tr)]/T"][(janaf_data["T"] >= T_min) & (janaf_data["T"] <= T_max)]
    H = janaf_data["H-H(Tr)"][(janaf_data["T"] >= T_min) & (janaf_data["T"] <= T_max)]
    # calculate mu
    mu = ((H - T*S)/N_avogadro)/1.602176634e-19 # eV
    return mu, T
    
def match_mu_T_data(loaded_data_1, loaded_data_2):
    '''
    Matches data loaded from two different files, and returns a list of the data with aligned temperature values.
    Inputs
    ------
        loaded_data_1: list or tuple
            Mu and T data loaded from first file
        loaded_data_2: list or tuple
            Mu and T data loaded from second file
    Returns
    -------
        synced_1: list
            Mu and T data from first file, with aligned temperature values
        synced_2: list
            Mu and T data from second file, with aligned temperature values
    '''
    T, idx1, idx2 = np.intersect1d(loaded_data_1[1], loaded_data_2[1], return_indices=True)
    synced_1 = [loaded_data_1[0][idx1], T]
    synced_2 = [loaded_data_2[0][idx2], T]
    # print change in data length
    print("Data length change: ", len(loaded_data_1[1])-len(T), len(loaded_data_2[1])-len(T))
    return synced_1, synced_2