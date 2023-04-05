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
m = 2.6566962e-26*2 #mass of single O2 molecule in kg
sigma_sym = 2 # Symmetry number of O2 molecule (2 for homonuclear diatomic, 1 for heteronuclear diatomic)
I_spin = 3 # Electronic spin degeneracy of O2 molecule
B_0 = 0.18/1000*1.6022e-19 # Rotational constant in J
omega_0 = 4.739261388e13 # s^-1; tabulated at: 0.196 eV; corresponds to 4.739261388e13 Hz frequency

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
            mu_O: numpy array
                Chemical potential of oxygen (1/2 O2) in eV
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
    _O2_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/O-029.txt",T_min=0,T_max=1000)
    O_janaf = (_O2_janaf[0]/2,_O2_janaf[1])
    TiO2_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/TiO2-anatase-042.txt",T_min=0,T_max=1000)
    Ti_ref_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Ti-alpha-002.txt",T_min=0,T_max=1000)
    Nb_ref_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nb-ref-001.txt",T_min=0,T_max=1000)
    NbO_cr_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/NbO-cr-008.txt",T_min=0,T_max=1000)
    NbO2_cr_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/NbO2-cr-012.txt",T_min=0,T_max=1000)
    Nb2O5_cr_janaf = calculate_mu_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nb2O5-cr-016.txt",T_min=0,T_max=1000)

    # match T values between JANAF data
    TiO2_janaf, Ti_ref_janaf = match_mu_T_data(TiO2_janaf, Ti_ref_janaf)
    NbO2_cr_janaf, Nb_ref_janaf = match_mu_T_data(NbO2_cr_janaf, Nb_ref_janaf)
    NbO_cr_janaf, Nb_ref_janaf = match_mu_T_data(NbO_cr_janaf, Nb_ref_janaf)
    O_janaf, Ti_ref_janaf = match_mu_T_data(O_janaf, Ti_ref_janaf)
    O_janaf, Nb_ref_janaf = match_mu_T_data(O_janaf, Nb_ref_janaf)

    for i in [O_janaf,TiO2_janaf,Ti_ref_janaf,Nb_ref_janaf,NbO_cr_janaf,NbO2_cr_janaf]:
        print("JANAF data length: ", i[1], i[0])

    
    # Vasp energies (eV)
    ### PBE ###
    TiO2_vasp = -8.97807611083333 # TiO2 anatase PBE (per atom)
    TiO2_rutile_vasp = -8.94661720166667 # TiO2 rutile PBE (per atom)
    Ti_vasp = -7.83560428 # Ti hcp PBE
    Nb_bcc_vasp = -10.21615905 # Nb bcc PBE
    NbO_rs_vasp = -9.63023941 # NbO bcc PBE (per atom)
    NbO2_distorted_rutile_vasp = -9.28531345083333 # NbO2 distorted rutile PBE (per atom)
    Nb2O5_vasp = -9.16837372669643 # Nb2O5 PBE (per atom)
    Ti2O3_vasp = -9.010753039 # Ti2O3 PBE (per atom)
    TiO_vasp = -8.924520636 # Ti5O5 PBE (per atom)
    #TiO_vasp = -8.686619775 # fully filled TiO FCC PBE (per atom)
    MoO3_vasp = -269.53455837/32 # MoO3 PBE (per atom)
    WO3_vasp = -72.88082529/8 # WO3 PBE (per atom)
    Mo_vasp = -21.86550460/2 # Mo (cubic) PBE (per atom)
    W_vasp = -25.90719811/2 # W (cubic) PBE (per atom)

    '''
    ### SCAN ###
    TiO2_vasp = -7.10112399083333 # TiO2 anatase SCAN (per atom)
    TiO_vasp = -26.3972944895 # Ti5O5 fcc SCAN somehow higher E than HCP TiO???????
    Ti_vasp = -45.100338545 # Ti hcp SCAN
    Nb_bcc_vasp = -51.69843494 # Nb bcc SCAN
    NbO_rs_vasp = -36.7186747483333 # NbO bcc SCAN (per atom)
    NbO2_ideal_rutile_vasp = -18.74257971 # NbO2 ideal rutile SCAN (per atom)
    '''

    # Define temperature range
    T = O_janaf[1] # Temperature in K
    # Calculate oxygen chemical potential
    mu_O = np.zeros(len(T))
    delta_mu_O2_vib = np.zeros(len(T))
    for i in range(len(T)):
        mu_O[i] = oxygen_mu_0(T[i],calc_type="PBE")
        delta_mu_O2_vib[i] = delta_mu_oxygen_vibrational_component(T[i])

    # calculate dG = dG_vasp - dG_janaf
    # calculate dG_vasp = G(TiO2)_vasp - G(Ti)_vasp - mu(O2)_vasp <--- mu(O2)_vasp = E(O2)_vasp + delta_mu_O2 (or just grab from function)

    ######## START Loading JANAF formation energies (Gibbs) ########
    # TiO2 (anatase)
    dG_form_janaf_TiO2 = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/TiO2-anatase-042.txt",T_min=0,T_max=1000)
    dG_form_janaf_NbO2 = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/NbO2-cr-012.txt",T_min=0,T_max=1000)
    dG_form_janaf_NbO = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/NbO-cr-008.txt",T_min=0,T_max=1000) #unnormalized
    print('--------------------------\n',dG_form_janaf_NbO[0],type(dG_form_janaf_NbO[0]))
    #normalize to per O2
    for k,j in enumerate(dG_form_janaf_NbO[0]):
        dG_form_janaf_NbO[0][k] = j*2
    print('--------------------------\n',dG_form_janaf_NbO[0],type(dG_form_janaf_NbO[0]))
    dG_form_janaf_Ti = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Ti-alpha-002.txt",T_min=0,T_max=1000)
    dG_form_janaf_Nb = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nb-ref-001.txt",T_min=0,T_max=1000)
    dG_form_janaf_O = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/O-029.txt",T_min=0,T_max=1000)
    
    dG_form_janaf_Nb2O5 = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Nb2O5-cr-016.txt",T_min=0,T_max=1000) #unnormalized
    print('--------------------------\n',dG_form_janaf_Nb2O5[0],type(dG_form_janaf_Nb2O5[0]))
    #normalize to per O2
    for k,j in enumerate(dG_form_janaf_Nb2O5[0]):
        dG_form_janaf_Nb2O5[0][k] = j*2/5
    print('--------------------------\n',dG_form_janaf_Nb2O5[0],type(dG_form_janaf_Nb2O5[0]))
    
    dG_form_janaf_TiO2_rutile = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/TiO2-rutile-042.txt",T_min=0,T_max=1000)
    
    # Ti2O3 (hopefully corundum, labelled in JANAF as 'alpha', only up to 470K before phase transition to beta)
    dG_form_janaf_Ti2O3 = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/Ti2O3-cr-059.txt",T_min=0,T_max=400) #unnormalized
    #normalize to per O2
    for k,j in enumerate(dG_form_janaf_Ti2O3[0]):
        dG_form_janaf_Ti2O3[0][k] = j*2/3

    # TiO (alpha, unnormalized)
    dG_form_janaf_TiO = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/TiO-alpha-cr-018.txt",T_min=0,T_max=1000) #unnormalized
    #normalize to per O2
    for k,j in enumerate(dG_form_janaf_TiO[0]):
        dG_form_janaf_TiO[0][k] = j*2

    # MoO3 (cr, unnormalized)
    dG_form_janaf_MoO3 = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/MoO3-cr-014.txt",T_min=0,T_max=1000) #unnormalized
    #normalize to per O2
    for k,j in enumerate(dG_form_janaf_MoO3[0]):
        dG_form_janaf_MoO3[0][k] = j*2/3
    
    # WO3 (cr, unnormalized)
    dG_form_janaf_WO3 = formation_enthalpy_from_janaf("/home/jonnyli/Documents/McQuarrie/JANAF/WO3-cr-065.txt",T_min=0,T_max=1000) #unnormalized
    #normalize to per O2
    for k,j in enumerate(dG_form_janaf_WO3[0]):
        dG_form_janaf_WO3[0][k] = j*2/3

    ######## END Loading JANAF formation energies (Gibbs) ########

    # Length checks
    print("dGf TiO2: ", dG_form_janaf_TiO2[0], len(dG_form_janaf_TiO2[0]), "\ndGf_NbO2: ", dG_form_janaf_NbO2[0], len(dG_form_janaf_NbO2[0]), "\ndGf_NbO", dG_form_janaf_NbO[0], len(dG_form_janaf_NbO[0]), "\ndGf_Ti", dG_form_janaf_Ti[0], len(dG_form_janaf_Ti[0]), "\ndGf_Nb", dG_form_janaf_Nb[0], len(dG_form_janaf_Nb[0]), "\ndGf_O", dG_form_janaf_O[0], len(dG_form_janaf_O[0]), "\ndGf_TiO2_rutile", dG_form_janaf_TiO2_rutile[0], len(dG_form_janaf_TiO2_rutile[0]), "\ndGf_Ti2O3", dG_form_janaf_Ti2O3[0], len(dG_form_janaf_Ti2O3[0]))
    
    # Sync up length T across data
    dG_form_janaf_TiO2, dG_form_janaf_NbO = match_mu_T_data(dG_form_janaf_TiO2, dG_form_janaf_NbO)
    dG_form_janaf_TiO2_rutile, dG_form_janaf_TiO2 = match_mu_T_data(dG_form_janaf_TiO2_rutile, dG_form_janaf_TiO2)
    
    ######## START Calculating dG = vasp - JANAF ########
    
    # TiO2 (anatase)
    dG_vasp_TiO2 = TiO2_vasp*3 - Ti_vasp - (2*mu_O)
    dG_TiO2 = dG_vasp_TiO2 - dG_form_janaf_TiO2[0]

    # TiO2 (rutile)
    dG_vasp_TiO2_rutile = TiO2_rutile_vasp*3 - Ti_vasp - (2*mu_O)
    dG_TiO2_rutile = dG_vasp_TiO2_rutile - dG_form_janaf_TiO2_rutile[0]

    # NbO2 (rutile)
    # dG_janaf_NbO2 = NbO2_cr_janaf[0] - Nb_ref_janaf[0] - O_janaf[0]
    dG_vasp_NbO2 = NbO2_distorted_rutile_vasp*3 - Nb_bcc_vasp - (2*mu_O)
    dG_NbO2 = dG_vasp_NbO2 - dG_form_janaf_NbO2[0]

    # NbO (rs) normalized to per O2
    #dG_janaf_NbO = NbO_cr_janaf[0] - Nb_ref_janaf[0] - O_janaf[0]
    dG_vasp_NbO = (NbO_rs_vasp*2 - Nb_bcc_vasp - (mu_O))*2
    dG_NbO = dG_vasp_NbO - dG_form_janaf_NbO[0]
    
    # Nb2O5 normalized to per O2
    dG_vasp_Nb2O5 = (Nb2O5_vasp*7 - 2*Nb_bcc_vasp - (5*mu_O))*2/5
    dG_Nb2O5 = dG_vasp_Nb2O5 - dG_form_janaf_Nb2O5[0]

    # Ti2O3 (corundum) normalized to per O2
    # quick truncation of mu_O to match length of Ti2O3
    mu_O_short, dG_form_janaf_Ti2O3 = match_mu_T_data((mu_O,T), dG_form_janaf_Ti2O3)
    dG_vasp_Ti2O3 = (Ti2O3_vasp*5 - 2*Ti_vasp - (3*mu_O_short[0]))*2/3
    dG_Ti2O3 = dG_vasp_Ti2O3 - dG_form_janaf_Ti2O3[0]

    # TiO (alpha) normalized to per O2
    dG_vasp_TiO = (TiO_vasp*2 - Ti_vasp - (mu_O))*2
    dG_TiO = dG_vasp_TiO - dG_form_janaf_TiO[0]

    # MoO3 (cr) normalized to per O2
    dG_vasp_MoO3 = (MoO3_vasp*4 - Mo_vasp - (3*mu_O))*2/3
    dG_MoO3 = dG_vasp_MoO3 - dG_form_janaf_MoO3[0]

    # WO3 (cr) normalized to per O2
    dG_vasp_WO3 = (WO3_vasp*4 - W_vasp - (3*mu_O))*2/3
    dG_WO3 = dG_vasp_WO3 - dG_form_janaf_WO3[0]

    ######## END Calculating dG = vasp - JANAF ########

    # Plot oxygen chemical potential
    i = 3
    fig, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.plot(T,mu_O,label=r'Theoretical $\Delta$$\mu_O$')
    ax1.scatter(O_janaf[1],O_janaf[0],label="JANAF-O2/2")
    #ax1.scatter(TiO2_janaf[1],TiO2_janaf[0],label="JANAF-TiO2")
    offset=mu_O-O_janaf[0][0]
    print("mu_O: ", mu_O)
    print("offset: ", offset)
    #ax1.scatter(Ti_ref_janaf[1],Ti_ref_janaf[0],label="JANAF-Ti")
    #ax1.scatter(TiO2_janaf[1],TiO2_janaf[0],label="JANAF-TiO2-anatase")
    #ax1.scatter(Nb_ref_janaf[1],Nb_ref_janaf[0],label="JANAF-Nb")
    #ax1.scatter(NbO_cr_janaf[1],NbO_cr_janaf[0],label="JANAF-NbO-cr")
    #ax1.scatter(NbO2_cr_janaf[1],NbO2_cr_janaf[0],label="JANAF-NbO2-cr")
    #plot difference
    ax1.plot(T,mu_O-offset[0],label=r'$\Delta$$\mu_O$ - $\Delta$$\mu_O$ offset @0K')
    #plot vibrational component of delta_mu_O
    ax1.plot(T,delta_mu_O2_vib,label=r'$\Delta$$\mu_O$ vibrational component')
    ax1.hlines(y=offset[0],xmin=T[0],xmax=T[-1],label=r'$\Delta$$\mu_O$ offset @0K',linestyles='dashed')
    ax1.set_title(r'$\mu$ vs. Temperature')


    ax2.plot(T,dG_TiO2,label=r'$TiO_2$ anatase',linestyle='dotted',color='blue')
    ax2.plot(T,dG_NbO,label=r'$NbO$ bcc',linestyle='dotted',color='yellow')
    ax2.plot(T,dG_NbO2,label=r'$NbO_2$ rutile',linestyle='dotted',color='green')
    ax2.plot(T,dG_TiO2_rutile,label=r'$TiO_2$ rutile',linestyle='dotted',color='orange')
    ax2.plot(T,dG_Nb2O5,label=r'$Nb_2O_5$',linestyle='dotted',color='red')
    ax2.plot(mu_O_short[1],dG_Ti2O3,label=r'$Ti_2O_3$',linestyle='dotted',color='purple')
    ax2.plot(T,dG_TiO,label=r'$TiO$',linestyle='dotted',color='black')
    ax2.plot(T,dG_MoO3,label=r'$MoO_3$',linestyle='dotted',color='brown')
    ax2.plot(T,dG_WO3,label=r'$WO_3$',linestyle='dotted',color='pink')
    print("Oxygen offset is: ",offset[i]," eV at", T[i], "K")
    print("dG_TiO2 is: ",dG_TiO2[i]," eV at ", T[i]," K")
    print("dG_NbO is: ",dG_NbO[i]," eV at ", T[i]," K")
    print("dG_NbO2 is: ",dG_NbO2[i]," eV at ", T[i]," K")
    print("dG_TiO2_rutile is: ",dG_TiO2_rutile[i]," eV at ", T[i]," K")
    print("dG_Nb2O5 is: ",dG_Nb2O5[i]," eV at ", T[i]," K")
    print("dG_Ti2O3 is: ",dG_Ti2O3[i]," eV at ", T[i]," K")
    print("dG_TiO is: ",dG_TiO[i]," eV at ", T[i]," K")
    print("dG_MoO3 is: ",dG_MoO3[i]," eV at ", T[i]," K")
    print("dG_WO3 is: ",dG_WO3[i]," eV at ", T[i]," K")

    ax2.set_title(r'$\Delta\Delta H(T)$ = $\Delta H_{vasp}$ - $\Delta H_{janaf}$')
    plt.xlabel('Temperature (K)')
    ax1.set_ylabel(r'chemical potential (eV)')
    ax2.set_ylabel(r'$\Delta\Delta H$ (eV)')
    ax1.legend()
    ax2.legend()
    #plt.show()



    fig3, ax4 = plt.subplots()
    #plt.scatter(dG_janaf_NbO[0],dG_vasp_NbO[0],label="NbO")
    ax4.scatter(dG_form_janaf_TiO2[0][i],dG_vasp_TiO2[i],label="TiO2-anatase",color='blue')
    ax4.scatter(dG_form_janaf_NbO[0][i],dG_vasp_NbO[i],label="NbO",marker='+',color='yellow')
    ax4.scatter(dG_form_janaf_NbO2[0][i],dG_vasp_NbO2[i],label="NbO2-rutile",marker='*',color='green')
    ax4.scatter(dG_form_janaf_TiO2_rutile[0][i],dG_vasp_TiO2_rutile[i],label="TiO2-rutile",marker='x',color='orange')
    ax4.scatter(dG_form_janaf_Nb2O5[0][i],dG_vasp_Nb2O5[i],label="Nb2O5",marker='^',color='red')
    ax4.scatter(dG_form_janaf_Ti2O3[0][i],dG_vasp_Ti2O3[i],label="Ti2O3",marker='s',color='purple')
    ax4.scatter(dG_form_janaf_TiO[0][i],dG_vasp_TiO[i],label="Ti5O5",marker='d',color='black')
    ax4.scatter(dG_form_janaf_MoO3[0][i],dG_vasp_MoO3[i],label="MoO3",marker='p',color='brown')
    ax4.scatter(dG_form_janaf_WO3[0][i],dG_vasp_WO3[i],label="WO3",marker='h',color='pink')
    # plot vertical deviation from y=x line
    ax4.plot([dG_form_janaf_TiO2[0][i],dG_form_janaf_TiO2[0][i]],[dG_vasp_TiO2[i],dG_form_janaf_TiO2[0][i]],linestyle='dashed',color='blue')
    ax4.annotate('%f eV' % (dG_form_janaf_TiO2[0][i]-dG_vasp_TiO2[i]), xy=(dG_form_janaf_TiO2[0][i],dG_vasp_TiO2[i]), xytext=(dG_form_janaf_TiO2[0][i],dG_vasp_TiO2[i]))
    ax4.plot([dG_form_janaf_NbO[0][i],dG_form_janaf_NbO[0][i]],[dG_vasp_NbO[i],dG_form_janaf_NbO[0][i]],linestyle='dashed',color='yellow')
    ax4.annotate('%f eV' % (dG_form_janaf_NbO[0][i]-dG_vasp_NbO[i]), xy=(dG_form_janaf_NbO[0][i],dG_vasp_NbO[i]), xytext=(dG_form_janaf_NbO[0][i],dG_vasp_NbO[i]))
    ax4.plot([dG_form_janaf_NbO2[0][i],dG_form_janaf_NbO2[0][i]],[dG_vasp_NbO2[i],dG_form_janaf_NbO2[0][i]],linestyle='dashed',color='green')
    ax4.annotate('%f eV' % (dG_form_janaf_NbO2[0][i]-dG_vasp_NbO2[i]), xy=(dG_form_janaf_NbO2[0][i],dG_vasp_NbO2[i]), xytext=(dG_form_janaf_NbO2[0][i],dG_vasp_NbO2[i]))
    ax4.plot([dG_form_janaf_TiO2_rutile[0][i],dG_form_janaf_TiO2_rutile[0][i]],[dG_vasp_TiO2_rutile[i],dG_form_janaf_TiO2_rutile[0][i]],linestyle='dashed',color='orange')
    ax4.annotate('%f eV' % (dG_form_janaf_TiO2_rutile[0][i]-dG_vasp_TiO2_rutile[i]), xy=(dG_form_janaf_TiO2_rutile[0][i],dG_vasp_TiO2_rutile[i]), xytext=(dG_form_janaf_TiO2_rutile[0][i],dG_vasp_TiO2_rutile[i]))
    ax4.plot([dG_form_janaf_Nb2O5[0][i],dG_form_janaf_Nb2O5[0][i]],[dG_vasp_Nb2O5[i],dG_form_janaf_Nb2O5[0][i]],linestyle='dashed',color='red')
    ax4.annotate('%f eV' % (dG_form_janaf_Nb2O5[0][i]-dG_vasp_Nb2O5[i]), xy=(dG_form_janaf_Nb2O5[0][i],dG_vasp_Nb2O5[i]), xytext=(dG_form_janaf_Nb2O5[0][i],dG_vasp_Nb2O5[i]))
    ax4.plot([dG_form_janaf_Ti2O3[0][i],dG_form_janaf_Ti2O3[0][i]],[dG_vasp_Ti2O3[i],dG_form_janaf_Ti2O3[0][i]],linestyle='dashed',color='purple')
    ax4.annotate('%f eV' % (dG_form_janaf_Ti2O3[0][i]-dG_vasp_Ti2O3[i]), xy=(dG_form_janaf_Ti2O3[0][i],dG_vasp_Ti2O3[i]), xytext=(dG_form_janaf_Ti2O3[0][i],dG_vasp_Ti2O3[i]))
    ax4.plot([dG_form_janaf_TiO[0][i],dG_form_janaf_TiO[0][i]],[dG_vasp_TiO[i],dG_form_janaf_TiO[0][i]],linestyle='dashed',color='black')
    ax4.annotate('%f eV' % (dG_form_janaf_TiO[0][i]-dG_vasp_TiO[i]), xy=(dG_form_janaf_TiO[0][i],dG_vasp_TiO[i]), xytext=(dG_form_janaf_TiO[0][i],dG_vasp_TiO[i]))
    ax4.plot([dG_form_janaf_MoO3[0][i],dG_form_janaf_MoO3[0][i]],[dG_vasp_MoO3[i],dG_form_janaf_MoO3[0][i]],linestyle='dashed',color='brown')
    ax4.annotate('%f eV' % (dG_form_janaf_MoO3[0][i]-dG_vasp_MoO3[i]), xy=(dG_form_janaf_MoO3[0][i],dG_vasp_MoO3[i]), xytext=(dG_form_janaf_MoO3[0][i],dG_vasp_MoO3[i]))
    ax4.plot([dG_form_janaf_WO3[0][i],dG_form_janaf_WO3[0][i]],[dG_vasp_WO3[i],dG_form_janaf_WO3[0][i]],linestyle='dashed',color='pink')
    ax4.annotate('%f eV' % (dG_form_janaf_WO3[0][i]-dG_vasp_WO3[i]), xy=(dG_form_janaf_WO3[0][i],dG_vasp_WO3[i]), xytext=(dG_form_janaf_WO3[0][i],dG_vasp_WO3[i]))

       
    #plot y=x line
    axis_lim=[-11,-4.5]
    ax4.plot(axis_lim,axis_lim,label='_nolegend_',linestyle='dotted')
    #ax4.set_xlim(axis_lim[0],axis_lim[1])
    #ax4.set_ylim(axis_lim[0],axis_lim[1])
    plt.xlabel("JANAF formation enthalpy (eV)")
    plt.ylabel("VASP formation enthalpy (eV), no offset")
    plt.title("Formation enthalpies (normalized per O2) at %f K" % T[i])
    ax4.legend()

    # plot offset vs oxidation state
    fig4, ax5 = plt.subplots()
    ax5.scatter(2,dG_TiO[i],label="Ti5O5",marker='d',color='black')
    ax5.scatter(3,dG_Ti2O3[i],label="Ti2O3",marker='s',color='purple')
    ax5.scatter(4,dG_TiO2[i],label="TiO2 anatase",marker='o',color='blue')
    ax5.scatter(4,dG_TiO2_rutile[i],label="TiO2 rutile",marker='o',color='orange')
    ax5.scatter(4,dG_NbO2[i],label="NbO2 rutile",marker='^',color='cyan')
    ax5.scatter(5,dG_Nb2O5[i],label="Nb2O5",marker='v',color='red')
    ax5.scatter(2,dG_NbO[i],label="NbO",marker='d',color='gold')
    ax5.scatter(6,dG_MoO3[i],label="MoO3",marker='o',color='brown')
    ax5.scatter(6,dG_WO3[i],label="WO3",marker='x',color='green')
    oxidation_states = [2,3,4,4,4,5,2,6,6]
    offsets = [dG_TiO[i],dG_Ti2O3[i],dG_TiO2[i],dG_TiO2_rutile[i],dG_NbO2[i],dG_Nb2O5[i],dG_NbO[i],dG_MoO3[i],dG_WO3[i]]
    
    '''
    import sklearn.linear_model
    linear_regression = sklearn.linear_model.LinearRegression()
    linear_regression.fit(np.array(oxidation_states).reshape(-1,1),np.array(offsets).reshape(-1,1))
    print(linear_regression.coef_)
    '''
    plt.title('Formation enthalpy offsets vs oxidation state at %f K' % T[i])
    ax5.legend()
    plt.xlabel("Oxidation state")
    plt.ylabel("Offset (eV)")

    plt.show()

if __name__ == '__main__':
    main()