from scipy.optimize import curve_fit
import numpy as np
import math
import colorsys

def totalOccFromKobsKI(t,kobs,conc0E,conc0I,KI): #eq 1
    Y0=conc0E/(1+(KI/conc0I))
    Ymax=conc0E-Y0
    return Ymax*(1-np.e**(-kobs*t))+Y0 #returned as a fraction

def occFromKobs(t,kobs,conc0E): #eq 1, scales from initial conc of enzyme
    return conc0E*(1-np.e**(-kobs*t)) #returned as a fraction

def kobsFromConcI(KI,kinact,concI): #eq 2
    return (kinact*concI)/(KI+concI)

def occFromPop(sol):
    freeP = sol[:, 1]  # Free protein column
    bound_states = sol[:, 2:]  # Select all columns from index 2 onwards as bound state
    boundP = np.sum(bound_states, axis=1)  # Sum the bound states along the row axis
    occupancy = (boundP / (boundP + freeP))
    return occupancy

def finalStateOccFromPop(sol):
    freeP = sol[:, 1]  # Free protein column
    bound_states = sol[:, 2:]  # Select all columns from index 2 onwards as bound state
    boundP = np.sum(bound_states, axis=1)  # Sum the bound states along the row axis
    occupancy = (sol[:, -1] / (boundP + freeP))
    return occupancy

def kobsKIFromTotalOcc(t, occupancy, conc0E,conc0I): 
    # Create a lambda function to "fix" conc0E
    func = lambda t, kobs, KI: totalOccFromKobsKI(t, kobs, conc0E,conc0I,KI)
    popt, _ = curve_fit(func, t, occupancy, p0=[0.001,1000], bounds=(0, np.inf)) # Non-negative bounds
    kobs = popt[0]
    KI =  popt[1]
    return kobs,KI


def kobsFromOcc(t, occupancy, conc0E): 
    # Create a lambda function to "fix" conc0E
    func = lambda t, kobs: occFromKobs(t, kobs, conc0E)
    popt, _ = curve_fit(func, t, occupancy, p0=[0.001], bounds=(0, np.inf)) # Non-negative bounds
    kobs = popt[0]
    return kobs


def kinactKIFromkobs(concI, kobs):
    def model_func(I, KI, kinact):
        return (kinact * I) / (KI + I)
    popt, _ = curve_fit(model_func, concI, kobs, p0=[1000, 0.001], bounds=(0, np.inf))
    KI_fit, kinact_fit = popt
    return KI_fit, kinact_fit

def fit_occupancy(t,model,occ_fit_type=None):
    #TODO: change occ_fit_type to be CO=None, TO=None by default. State is specified with species name. Check if fourstate gets same results with TO
    #TODO: make it not reliant on deterministic results
    """
    occ_fit_type must be CO or TO for covalent or total occupancy, respectively.
    """
    protein_conc=model.species["E"]["conc"]
    ligand_conc=model.species["I"]["conc"]
    if occ_fit_type=="TO":
        occupancy=np.sum(model.traj_deterministic[:, 2:], axis=1)
        kobs,KINum = kobsKIFromTotalOcc(t,occupancy,protein_conc,ligand_conc)
        return kobs,KINum
    elif occ_fit_type=="CO":
        occupancy = model.traj_deterministic[:,model.species_order["EI"]]
        kobs = kobsFromOcc(t,occupancy,protein_conc)
        return kobs
    elif occ_fit_type is None:
        kobs = None
        return kobs
    else:
        raise ValueError("occ_fit_type must be CO or TO for covalent or total occupancy, respectively.")

def occ_from_kobs_complex(t, kobs, conc_0E, k_start, t_half, fraction_state):
    """
    Fitting function for covalent inhibition with a startup phase and a side reaction pathway.
    
    :param t: Time points
    :param conc_0E: Initial concentration of E
    :param k_start: Rate constant for the startup phase
    :param t_half: Time to half-maximum in the sigmoidal function
    :param kobs: Observed rate constant 
    :param fraction_state: Fraction of E that will eventually convert to the state
    :return: Concentration of state at each time point
    """
    return conc_0E * 1 / (1 + np.exp(-k_start * (t - t_half))) * (1 - np.exp(-kobs * t)) * fraction_state

def kobs_from_occ_complex(t, occupancy, conc0E): 
    """
    Function to fit k_start, t_half, kobs, and fraction_state based on occupancy and conc0E.

    :param t: Time points
    :param occupancy: Measured occupancy data
    :param conc0E: Initial concentration of E
    :return: Fitted values of kobs,k_start,t_half,fraction_state
    """
    func = lambda t, kobs, k_start, t_half, fraction_state: occ_from_kobs_complex(t, kobs, conc0E, k_start, t_half, fraction_state)
    initial_guesses = [0.01, min(t) + 0.1 * (max(t) - min(t)), 0.01, 0.999]  # k_start, t_half, kobs, fraction_state
    popt, _ = curve_fit(func, t, occupancy, p0=initial_guesses, bounds=(0, np.inf))  # Non-negative bounds
    kobs,k_start,t_half,fraction_state = popt
    return kobs,k_start,t_half,fraction_state 

def fit_occupancy_complex(t,model,occ_fit_type=None):
    #TODO: change occ_fit_type to be CO=None, TO=None by default. State is specified with species name. Check if fourstate gets same results with TO
    #TODO: make it not reliant on deterministic results
    """
    occ_fit_type must be CO or TO for covalent or total occupancy, respectively.
    """
    protein_conc=model.species["E"]["conc"]
    ligand_conc=model.species["I"]["conc"]
    if occ_fit_type=="TO":
        occupancy=np.sum(model.traj_deterministic[:, 2:], axis=1)
        kobs,KINum = kobsKIFromTotalOcc(t,occupancy,protein_conc,ligand_conc)
        return kobs,KINum
    elif occ_fit_type=="CO":
        occupancy = model.traj_deterministic[:,model.species_order["EI"]]
        kobs,k_start,t_half,fraction_state = kobs_from_occ_complex(t,occupancy,protein_conc)
        return kobs,k_start,t_half,fraction_state 
    elif occ_fit_type is None:
        kobs = None
        return kobs
    else:
        raise ValueError("occ_fit_type must be CO or TO for covalent or total occupancy, respectively.")

def make_int_rates(intOOM,Pf):
    """
    Provides rates between two states that are at rapid equilibrium and therefore approximated by the population distribution. 
    intOOM is the order of magnitude of the rates.
    Pf is the proportion of the forward state, ie kf/(kf+kb) or [B]/([A]+[B]) for A<-->B.
    """
    kf=Pf*10**(float(intOOM)+1)
    kb=10**(float(intOOM)+1)-kf
    return kf,kb

def round_sig(num, sig_figs, autoformat=True):
    """
    Round up using significant figures.
    autoformat = True (default) formats into scientific notation if the order of magnitude more than +/- 3.
    """
    if num == 0:
        return 0.0

    order_of_magnitude = math.floor(math.log10(abs(num)))
    shift = sig_figs - 1 - order_of_magnitude

    # Scale the number, round it, and scale it back
    scaled_num = num * (10 ** shift)
    rounded_scaled_num = math.floor(scaled_num + 0.5)
    result = rounded_scaled_num / (10 ** shift)

    # Handling scientific notation conditionally
    if autoformat and (order_of_magnitude >= 4 or order_of_magnitude <= -4):
        return format(result, '.{0}e'.format(sig_figs - 1))
    else:
        return result

def assign_colors_to_species(schemes, saturation_range=(0.5, 0.7), lightness_range=(0.3, 0.4), method=None, offset=0, overwrite_existing=False):
    """
    Assigns a distinct and aesthetically distributed color to each species in the provided dictionary of kinetic schemes. 
    Uses the golden ratio for even hue distribution.
    Method can be "GR" to use the golden ratio to generate the hues.
    Offset can be used to offset the hues.
    overwrite_existing: If True, overwrite existing colors; if False, assign colors only to species without colors.
    """
    #TODO: This only works with a dictionary of schemes. It should work on single schemes too. 
    unique_species = set()
    for scheme in schemes.values():
        unique_species.update(scheme["species"].keys())

    golden_ratio_conjugate = 0.618033988749895
    hue = 0

    n_colors = len(unique_species)
    color_mapping = {}
    hues = np.linspace(0, 1, n_colors, endpoint=False)

    for i, species in enumerate(unique_species):
        # Skip species with existing color unless overwriting
        if not overwrite_existing and "color" in scheme["species"][species]:
            continue

        if method == "GR":
            hue += golden_ratio_conjugate + offset
        else:
            hue = hues[i] + offset
        hue %= 1

        lightness = np.random.uniform(*lightness_range)
        saturation = np.random.uniform(*saturation_range)
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
        color_mapping[species] = hex_color

    for scheme in schemes.values():
        for species in scheme["species"].keys():
            if overwrite_existing or "color" not in scheme["species"][species]:
                scheme["species"][species]["color"] = color_mapping.get(species, scheme["species"][species].get("color"))

    return schemes
