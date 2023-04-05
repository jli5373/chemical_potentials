import numpy as np
import matplotlib.pyplot as plt
import math

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
I_spin = 3 # Electronic spin degeneracy of N2 molecule TODO: Find this value
B_0 = 3.97212135e-23 # Rotational constant in J, JANAF tables list B_e = 1.99825 cm^-1 (N2)
omega_0 = 7.07265e13 # s^-1; tabulated at: 0.196 eV; corresponds to xx Hz frequency; JANAF tables list w_e = 2357.55 cm^-1 (N2)

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
            mu_N: numpy array
                Chemical potential of nitrogen (1/2 N2) in eV
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
    
def nitrogen_delta_mu_0(T,p=p_0):
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

def delta_mu_nitrogen_vibrational_component(T):
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

def nitrogen_mu_0(T,p=p_0,calc_type="None"):
    '''
    Calculates the chemical potential of nitrogen in eV, given a temperature in K and pressure in Pa.
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
        mu_N: float
            Chemical potential of nitrogen (1/2 N2) in eV
    '''
    if (calc_type=="None"):
        vasp_mu_N = 0.0
    elif(calc_type=="PBE"):
        vasp_mu_N = (-16.64101840+0)/2 #PBE N2 molecule (not corrected by xeV/N2 [similar to Ceder paper])
    elif(calc_type=="SCAN"):
        vasp_mu_N = -0.0 #SCAN N2 molecule/2 (uncorrected, will need to validate for SCAN) TODO: Switch to N2
        print("Need to add N2 SCAN")
    elif(calc_type=="R2SCAN"):
        vasp_mu_N = 0.0 #unimplemented for R2SCAN
        print("R2SCAN not yet implemented")
    #print("VASP reference chemical potential is: ", )
    mu_N = vasp_mu_N + nitrogen_delta_mu_0(T,p)
    return mu_N

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

def main():
    if(display_constants):
        print("kb = ",kb," J/K\n","B_0 = ",B_0," J\n","omega_0 = ",omega_0," J\n","h = ",h," J*s\n","p_0 =",p_0, "Pa\n","m = ",m," kg\n","sigma_sym = ",sigma_sym,"\n ","I_spin = ",I_spin, "\n")

    # Load JANAF data
    _N2_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nitrides/N2-ref-023.txt",T_min=0,T_max=1000)
    N_janaf = (_N2_janaf[0]/2,_N2_janaf[1])
    Ti_ref_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Ti-alpha-002.txt",T_min=0,T_max=1000)
    TiN_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nitrides/TiN-cr-014.txt",T_min=0,T_max=1000)
    Al_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Al-ref-001.txt",T_min=0,T_max=1000)
    AlN_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nitrides/AlN-cr-071.txt",T_min=0,T_max=1000)
    # Cr undergoes phase transition at 311K
    Cr_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Cr-ref-001.txt",T_min=0,T_max=311)
    CrN_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nitrides/CrN-cr-008.txt",T_min=0,T_max=1000)
    V_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/V-ref-001.txt",T_min=0,T_max=1000)
    VN_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nitrides/VN-cr-017.txt",T_min=0,T_max=1000)
    Zr_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Zr-ref-001.txt",T_min=0,T_max=1000)
    ZrN_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nitrides/ZrN-cr-019.txt",T_min=0,T_max=1000)

    for i in [N_janaf,TiN_janaf,Ti_ref_janaf,AlN_janaf,CrN_janaf,VN_janaf,ZrN_janaf]:
        print("JANAF data length: ", i[1], i[0])
    
    # match T values between JANAF data
    TiN_janaf, Ti_ref_janaf = match_mu_T_data(TiN_janaf, Ti_ref_janaf)
    VN_janaf, AlN_janaf = match_mu_T_data(VN_janaf, AlN_janaf)
    CrN_janaf, AlN_janaf = match_mu_T_data(CrN_janaf, AlN_janaf)
    N_janaf, Ti_ref_janaf = match_mu_T_data(N_janaf, Ti_ref_janaf)
    N_janaf, AlN_janaf = match_mu_T_data(N_janaf, AlN_janaf)

    for i in [N_janaf,TiN_janaf,Ti_ref_janaf,AlN_janaf,CrN_janaf,VN_janaf,ZrN_janaf]:
        print("JANAF data length: ", i[1], i[0])

    
    # Vasp energies (eV)
    ### PBE ###
    TiN_fcc_vasp = -19.63752789/2 #PBE TiN fcc (per atom)
    Ti_vasp = -23.52120854/3 #PBE Ti hcp (per atom)
    AlN_wurtzite_vasp = -29.78281926/4 #PBE AlN wurtzite (per atom)
    Al_vasp = -14.98180931/4 #PBE Al fcc (per atom)
    CrN_hex_vasp = -18.91592839/2 #PBE CrN hexagonal (per atom)
    Cr_vasp = -19.02043759/2 #PBE Cr bcc (per atom)
    VN_hex_vasp = -19.68372627/2 #PBE VN hexagonal (per atom)
    V_vasp = -17.98143454/2 #PBE V bcc (per atom)
    ZrN_fcc_vasp = -20.37142413/2 #PBE ZrN fcc (per atom)
    Zr_vasp = -17.04196849/2 #PBE Zr bcc (per atom)

    # Print energies
    if(display_constants):
        print("TiN_fcc_vasp = ",TiN_fcc_vasp," eV/atom\n", "Ti_vasp = ",Ti_vasp," eV/atom\n", "AlN_wurtzite_vasp = ",AlN_wurtzite_vasp," eV/atom\n", "Al_vasp = ",Al_vasp," eV/atom\n", "CrN_hex_vasp = ",CrN_hex_vasp," eV/atom\n", "Cr_vasp = ",Cr_vasp," eV/atom\n", "VN_hex_vasp = ",VN_hex_vasp," eV/atom\n", "V_vasp = ",V_vasp," eV/atom\n", "ZrN_fcc_vasp = ",ZrN_fcc_vasp," eV/atom\n", "Zr_vasp = ",Zr_vasp," eV/atom\n")
    # N2 binding energy PBE
    N2_molecule_vasp = -16.64101840/2 #PBE N2 molecule (per atom) #TODO: Fix this energy value cause it was a bad relaxation
    N2_separated_vasp = -6.24872394/2 #PBE N2 separated (per atom)
    binding_energy_N2_vasp = N2_molecule_vasp - N2_separated_vasp #PBE binding energy of N2 (per atom)
    print("PBE binding energy of N2: ", binding_energy_N2_vasp, "eV")

    '''
    ### SCAN ###
    #TODO: Update SCAN energies
    TiN_fcc_vasp = /3 #PBE TiN fcc (per atom)
    Ti_vasp = /3 #PBE Ti hcp (per atom)
    AlN_wurtzite_vasp = /4 #PBE AlN wurtzite (per atom)
    Al_vasp = /4 #PBE Al fcc (per atom)
    CrN_hex_vasp = /2 #PBE CrN hexagonal (per atom)
    Cr_vasp = /2 #PBE Cr bcc (per atom)
    VN_hex_vasp = /2 #PBE VN hexagonal (per atom)
    V_vasp = /2 #PBE V bcc (per atom)
    ZrN_FCC_vasp = /2 #PBE ZrN fcc (per atom)
    Zr_vasp = /2 #PBE Zr bcc (per atom)
    # N2 binding energy PBE
    N2_molecule_vasp = /2 #PBE N2 molecule (per atom)
    N2_separated_vasp = /2 #PBE N2 separated (per atom)
    binding_energy_N2_vasp = N2_molecule_vasp - N2_separated_vasp #PBE binding energy of N2 (per atom)
    print("binding energy of N2: ", binding_energy_N2_vasp, "eV")
    '''

    # Define temperature range
    T = N_janaf[1] # Temperature in K
    # Calculate nitrogen chemical potential
    mu_N = np.zeros(len(T))
    delta_mu_N2_vib = np.zeros(len(T))
    for i in range(len(T)):
        mu_N[i] = nitrogen_mu_0(T[i],calc_type="PBE")
        delta_mu_N2_vib[i] = delta_mu_nitrogen_vibrational_component(T[i])

    # calculate dH = dH_vasp - dH_janaf
    # calculate dH_vasp = H(TiN)_vasp - H(Ti)_vasp - 0.5*mu(N2)_vasp <--- mu(N2)_vasp = E(N2)_vasp + delta_mu_N2 (or just grab from function)

    ######## START Loading JANAF formation enthalpies ########
    # TiN (fcc)
    dH_from_janaf_TiN_fcc = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nitrides/TiN-cr-014.txt",T_min=0,T_max=1000)
    #'''
    print('--------------------------\n',dH_from_janaf_TiN_fcc[0],type(dH_from_janaf_TiN_fcc[0]))
    # normalize to per N2 molecule
    print("normalizing TiN_fcc to per N2 molecule")
    for k,j in enumerate(dH_from_janaf_TiN_fcc[0]):
        dH_from_janaf_TiN_fcc[0][k] = j*2
    print('--------------------------\n',dH_from_janaf_TiN_fcc[0],type(dH_from_janaf_TiN_fcc[0]))
    #'''

    # AlN Wurtzite
    dH_from_janaf_AlN_wurtzite = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nitrides/AlN-cr-071.txt",T_min=0,T_max=1000)
    #'''
    print('--------------------------\n',dH_from_janaf_AlN_wurtzite[0],type(dH_from_janaf_AlN_wurtzite[0]))
    # normalize to per N2 molecule
    print("normalizing AlN_wurtzite to per N2 molecule")
    for k,j in enumerate(dH_from_janaf_AlN_wurtzite[0]):
        dH_from_janaf_AlN_wurtzite[0][k] = j*2
    print('--------------------------\n',dH_from_janaf_AlN_wurtzite[0],type(dH_from_janaf_AlN_wurtzite[0]))
    #'''
    
    # CrN (hexagonal)
    dH_from_janaf_CrN_hex = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nitrides/CrN-cr-008.txt",T_min=0,T_max=1000)
    #'''
    print('--------------------------\n',dH_from_janaf_CrN_hex[0],type(dH_from_janaf_CrN_hex[0]))
    # normalize to per N2 molecule
    print("normalizing CrN_hex to per N2 molecule")
    for k,j in enumerate(dH_from_janaf_CrN_hex[0]):
        dH_from_janaf_CrN_hex[0][k] = j*2
    print('--------------------------\n',dH_from_janaf_CrN_hex[0],type(dH_from_janaf_CrN_hex[0]))
    #'''

    # VN (hexagonal)
    dH_from_janaf_VN_hex = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nitrides/VN-cr-017.txt",T_min=0,T_max=1000)
    #'''
    print('--------------------------\n',dH_from_janaf_VN_hex[0],type(dH_from_janaf_VN_hex[0]))
    # normalize to per N2 molecule
    print("normalizing VN_hex to per N2 molecule")
    for k,j in enumerate(dH_from_janaf_VN_hex[0]):
        dH_from_janaf_VN_hex[0][k] = j*2
    print('--------------------------\n',dH_from_janaf_VN_hex[0],type(dH_from_janaf_VN_hex[0]))
    #'''

    # ZrN (fcc)
    dH_from_janaf_ZrN_fcc = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nitrides/ZrN-cr-019.txt",T_min=0,T_max=1000)
    #'''
    print('--------------------------\n',dH_from_janaf_ZrN_fcc[0],type(dH_from_janaf_ZrN_fcc[0]))
    # normalize to per N2 molecule
    print("normalizing ZrN_fcc to per N2 molecule")
    for k,j in enumerate(dH_from_janaf_ZrN_fcc[0]):
        dH_from_janaf_ZrN_fcc[0][k] = j*2
    print('--------------------------\n',dH_from_janaf_ZrN_fcc[0],type(dH_from_janaf_ZrN_fcc[0]))
    #'''
    ######## END Loading JANAF formation energies (Gibbs) ########

    # Length checks
    print("N: ", mu_N, len(mu_N), "dHf TiN: ", dH_from_janaf_TiN_fcc[0], len(dH_from_janaf_TiN_fcc[0]), "dHf AlN: ", dH_from_janaf_AlN_wurtzite[0], len(dH_from_janaf_AlN_wurtzite[0]), "dHf CrN: ", dH_from_janaf_CrN_hex[0], len(dH_from_janaf_CrN_hex[0]), "dHf VN: ", dH_from_janaf_VN_hex[0], len(dH_from_janaf_VN_hex[0]), "dHf ZrN: ", dH_from_janaf_ZrN_fcc[0], len(dH_from_janaf_ZrN_fcc[0]))
    
    
    ######## START Calculating dH = vasp - JANAF ########

    # TiN (fcc) normalized to per N2 molecule
    dH_vasp_TiN_fcc = (TiN_fcc_vasp*2 - Ti_vasp - mu_N)*2 # per N2 molecule
    dH_TiN_fcc = dH_vasp_TiN_fcc - dH_from_janaf_TiN_fcc[0]

    # AlN (wurtzite) normalized to per N2 molecule
    dH_vasp_AlN_wurtzite = (AlN_wurtzite_vasp*2 - Al_vasp - mu_N)*2 #per N2 molecule
    dH_AlN_wurtzite = dH_vasp_AlN_wurtzite - dH_from_janaf_AlN_wurtzite[0]

    # CrN (hexagonal) normalized to per N2 molecule
    dH_vasp_CrN_hex = (CrN_hex_vasp*2 - Cr_vasp - mu_N)*2 #per N2 molecule
    dH_CrN_hex = dH_vasp_CrN_hex - dH_from_janaf_CrN_hex[0]

    # VN (hexagonal) normalized to per N2 molecule
    dH_vasp_VN_hex = (VN_hex_vasp*2 - V_vasp - mu_N)*2 #per N2 molecule
    dH_VN_hex = dH_vasp_VN_hex - dH_from_janaf_VN_hex[0]

    # ZrN (fcc) normalized to per N2 molecule
    dH_vasp_ZrN_fcc = (ZrN_fcc_vasp*2 - Zr_vasp - mu_N)*2 #per N2 molecule
    dH_ZrN_fcc = dH_vasp_ZrN_fcc - dH_from_janaf_ZrN_fcc[0]

    ######## END Calculating dH = vasp - JANAF ########

    # Plot nitrogen chemical potential
    i = 3
    fig, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.plot(T,mu_N,label=r'Theoretical $\Delta$$\mu_N$')
    ax1.scatter(N_janaf[1],N_janaf[0],label="JANAF-N2/2")
    #ax1.scatter(TiN_janaf[1],TiN_janaf[0],label="JANAF-TiN")
    offset=mu_N-N_janaf[0]
    print("offset: ", offset)
    print("offset amount: ", N_janaf[0])
    print("mu_N: ", mu_N)
    #ax1.scatter(Ti_ref_janaf[1],Ti_ref_janaf[0],label="JANAF-Ti")
    #ax1.scatter(TiN_janaf[1],TiN_janaf[0],label="JANAF-TiN2-anatase")
    #ax1.scatter(AlN_janaf[1],AlN_janaf[0],label="JANAF-Nb")
    #ax1.scatter(CrN_janaf[1],CrN_janaf[0],label="JANAF-NbO-cr")
    #ax1.scatter(VN_janaf[1],VN_janaf[0],label="JANAF-NbN2-cr")
    #plot difference
    #ax1.plot(T,mu_N-offset[0],label=r'$\Delta$$\mu_N$ - $\Delta$$\mu_N$ offset @0K')
    #plot vibrational component of delta_mu_O
    #ax1.plot(T,delta_mu_N2_vib,label=r'$\Delta$$\mu_N$ vibrational component')
    #show offset amount
    #ax1.hlines(y=offset[0],xmin=T[0],xmax=T[-1],label=r'$\Delta$$\mu_N$ offset @0K',linestyles='dashed')
    #ax1.set_title(r'$\mu$ vs. Temperature')
    ax1.set_title(r'$\mu$ vs. Temperature; $I_{spin}$=%f' % I_spin)


    ax2.plot(T,dH_TiN_fcc,label=r'$TiN$ fcc',linestyle='dotted',color='blue')
    ax2.plot(T,dH_AlN_wurtzite,label=r'$AlN$ wurtzite',linestyle='dotted',color='yellow')
    ax2.plot(T,dH_CrN_hex,label=r'$CrN$ hexagonal',linestyle='dotted',color='green')
    ax2.plot(T,dH_VN_hex,label=r'$VN$ hexagonal',linestyle='dotted',color='red')
    ax2.plot(T,dH_ZrN_fcc,label=r'$ZrN$ fcc',linestyle='dotted',color='purple')
    print("Nitrogen offset is: ",offset[i]," eV at", T[i], "K")
    print("dH_TiN is: ",dH_TiN_fcc[i]," eV at ", T[i]," K")
    print("dH_AlN is: ",dH_AlN_wurtzite[i]," eV at ", T[i]," K")
    print("dH_CrN is: ",dH_CrN_hex[i]," eV at ", T[i]," K")
    print("dH_VN is: ",dH_VN_hex[i]," eV at ", T[i]," K")
    print("dH_ZrN is: ",dH_ZrN_fcc[i]," eV at ", T[i]," K")


    ax2.set_title(r'$\Delta\Delta H(T)$ = $\Delta H_{vasp}$ - $\Delta H_{janaf}$')
    plt.xlabel('Temperature (K)')
    ax1.set_ylabel(r'chemical potential (eV)')
    ax2.set_ylabel(r'$\Delta\Delta H$ (eV)')
    ax1.legend()
    ax2.legend()
    #plt.show()



    fig3, ax4 = plt.subplots()
    ax4.scatter(dH_from_janaf_TiN_fcc[0][i],dH_vasp_TiN_fcc[i],label="TiN-fcc",color='blue')
    ax4.scatter(dH_from_janaf_AlN_wurtzite[0][i],dH_vasp_AlN_wurtzite[i],label="AlN-wurtzite",marker='+',color='yellow')
    ax4.scatter(dH_from_janaf_CrN_hex[0][i],dH_vasp_CrN_hex[i],label="CrN-hexagonal",marker='*',color='green')
    ax4.scatter(dH_from_janaf_VN_hex[0][i],dH_vasp_VN_hex[i],label="VN-hexagonal",marker='^',color='red')
    ax4.scatter(dH_from_janaf_ZrN_fcc[0][i],dH_vasp_ZrN_fcc[i],label="ZrN-fcc",marker='s',color='purple')
    # plot vertical deviation from y=x line
    ax4.plot([dH_from_janaf_TiN_fcc[0][i],dH_from_janaf_TiN_fcc[0][i]],[dH_vasp_TiN_fcc[i],dH_from_janaf_TiN_fcc[0][i]],linestyle='dashed',color='blue')
    ax4.annotate('%f eV' % (dH_from_janaf_TiN_fcc[0][i]-dH_vasp_TiN_fcc[i]), xy=(dH_from_janaf_TiN_fcc[0][i],dH_vasp_TiN_fcc[i]), xytext=(dH_from_janaf_TiN_fcc[0][i],dH_vasp_TiN_fcc[i]))
    ax4.plot([dH_from_janaf_AlN_wurtzite[0][i],dH_from_janaf_AlN_wurtzite[0][i]],[dH_vasp_AlN_wurtzite[i],dH_from_janaf_AlN_wurtzite[0][i]],linestyle='dashed',color='yellow')
    ax4.annotate('%f eV' % (dH_from_janaf_AlN_wurtzite[0][i]-dH_vasp_AlN_wurtzite[i]), xy=(dH_from_janaf_AlN_wurtzite[0][i],dH_vasp_AlN_wurtzite[i]), xytext=(dH_from_janaf_AlN_wurtzite[0][i],dH_vasp_AlN_wurtzite[i]))
    ax4.plot([dH_from_janaf_CrN_hex[0][i],dH_from_janaf_CrN_hex[0][i]],[dH_vasp_CrN_hex[i],dH_from_janaf_CrN_hex[0][i]],linestyle='dashed',color='green')
    ax4.annotate('%f eV' % (dH_from_janaf_CrN_hex[0][i]-dH_vasp_CrN_hex[i]), xy=(dH_from_janaf_CrN_hex[0][i],dH_vasp_CrN_hex[i]), xytext=(dH_from_janaf_CrN_hex[0][i],dH_vasp_CrN_hex[i]))
    ax4.plot([dH_from_janaf_VN_hex[0][i],dH_from_janaf_VN_hex[0][i]],[dH_vasp_VN_hex[i],dH_from_janaf_VN_hex[0][i]],linestyle='dashed',color='red')
    ax4.annotate('%f eV' % (dH_from_janaf_VN_hex[0][i]-dH_vasp_VN_hex[i]), xy=(dH_from_janaf_VN_hex[0][i],dH_vasp_VN_hex[i]), xytext=(dH_from_janaf_VN_hex[0][i],dH_vasp_VN_hex[i]))
    ax4.plot([dH_from_janaf_ZrN_fcc[0][i],dH_from_janaf_ZrN_fcc[0][i]],[dH_vasp_ZrN_fcc[i],dH_from_janaf_ZrN_fcc[0][i]],linestyle='dashed',color='purple')
    ax4.annotate('%f eV' % (dH_from_janaf_ZrN_fcc[0][i]-dH_vasp_ZrN_fcc[i]), xy=(dH_from_janaf_ZrN_fcc[0][i],dH_vasp_ZrN_fcc[i]), xytext=(dH_from_janaf_ZrN_fcc[0][i],dH_vasp_ZrN_fcc[i]))

       
    #plot y=x line
    axis_lim=[-8,-2]
    ax4.plot(axis_lim,axis_lim,label='_nolegend_',linestyle='dotted')
    #ax4.set_xlim(axis_lim[0],axis_lim[1])
    #ax4.set_ylim(axis_lim[0],axis_lim[1])
    plt.xlabel("JANAF formation enthalpy (eV)")
    plt.ylabel("VASP formation enthalpy (eV), no offset")
    plt.title("Formation enthalpies (normalized per N2) at %f K" % T[i])
    ax4.legend()
    '''
    # plot offset vs oxidation state
    fig4, ax5 = plt.subplots()
    ax5.scatter(3,dH_TiN_fcc[i],label="TiN_fcc",marker='d',color='blue')
    ax5.scatter(3,dH_AlN_wurtzite[i],label="AlN_wurtzite",marker='x',color='yellow')
    ax5.scatter(3,dH_CrN_hex[i],label="CrN_hex",marker='s',color='green')
    ax5.scatter(3,dH_VN_hex[i],label="VN_hex",marker='o',color='red')
    ax5.scatter(3,dH_ZrN_fcc[i],label="ZrN_fcc",marker='^',color='purple')

    oxidation_states = [3,3,3,3,3]
    offsets = [dH_TiN_fcc[i],dH_AlN_wurtzite[i],dH_CrN_hex[i],dH_VN_hex[i],dH_ZrN_fcc[i]]
    '''
    '''
    import sklearn.linear_model
    linear_regression = sklearn.linear_model.LinearRegression()
    linear_regression.fit(np.array(oxidation_states).reshape(-1,1),np.array(offsets).reshape(-1,1))
    print(linear_regression.coef_)
    '''
    '''
    plt.title('Formation energy offsets vs oxidation state at %f K' % T[i])
    ax5.legend()
    plt.xlabel("Oxidation state")
    plt.ylabel("Offset (eV)")
    '''
    plt.show()

if __name__ == '__main__':
    main()