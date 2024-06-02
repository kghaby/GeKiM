import gekim as gk
from gekim.fields.bio.enzyme.inhib import irrev as ii 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#TODO: add loop case
#TODO: add negative feedback case

# Test whether gekim NState systems yield the same output as hardcoded systems
# To add a new test case, add a new scheme to the schemes dictionary and make new hardcoded system class, then add it to compare. 
    # they need to share species with vani 3-state covlig systems, for simplicity

# User option to plot time course of NState systems (will still plot on mismatch)
PLOT=False

# Params
Kd = 10 #nM, koff/kon
koff = 0.0001 #~1/15 min
kon = koff/Kd
concI0,concE0=100,1
kinactf=0.001
kinactb=0
t = np.linspace(kon, 10000, 10000)

# Define gekim schemes
schemes={}
schemes["3S_vani"] = {
    "species": {
        "I": {"y0": concI0, "label": r"I"},
        "E": {"y0": concE0, "label": r"E"},
        "E_I": {"y0": 0, "label": r"E${\cdot}$I"},
        "EI": {"y0": 0, "label": r"E-I"},
    },
    "transitions": {
        "k1": {"k": kon, "source": ["E", "I"], "target": ["E_I"], "label": r"$k_{on}$"},
        "k2": {"k": koff, "source": ["E_I"], "target": ["E", "I"], "label": r"$k_{off}$"},
        "k3": {"k": kinactf, "source": ["E_I"], "target": ["EI"]}, #irrev step
        "k4": {"k": kinactb, "source": ["EI"], "target": ["E_I"]},
    },
}
schemes["3S_mod1.1"] = {
    "species": {
        "I": {"y0": concI0, "label": r"I"},
        "E": {"y0": concE0, "label": r"E"},
        "E_I": {"y0": 0, "label": r"E${\cdot}$I"},
        "EI": {"y0": 0, "label": r"E-I"},
    },
    "transitions": {
        "k1": {"k": kon, "source": ["E", "2.0I"], "target": ["E_I"], "label": r"$k_{on}$"},
        "k2": {"k": koff, "source": ["E_I"], "target": ["E", "2.0I"], "label": r"$k_{off}$"},
        "k3": {"k": kinactf, "source": ["3E_I"], "target": ["7EI"]}, #irrev step
        "k4": {"k": kinactb, "source": ["7EI"], "target": ["3E_I"]},
    },
}
schemes["3S_mod1.2"] = {
    "species": {
        "I": {"y0": concI0, "label": r"I"},
        "E": {"y0": concE0, "label": r"E"},
        "E_I": {"y0": 0, "label": r"E${\cdot}$I"},
        "EI": {"y0": 0, "label": r"E-I"},
    },
    "transitions": {
        "k1": {"k": kon, "source": ["E", "I", "I"], "target": ["E_I"], "label": r"$k_{on}$"},
        "k2": {"k": koff, "source": ["E_I"], "target": ["E", "I", "I"], "label": r"$k_{off}$"},
        "k3": {"k": kinactf, "source": ["E_I","E_I","E_I"], "target": ["7EI"]}, #irrev step
        "k4": {"k": kinactb, "source": ["7EI"], "target": ["E_I","E_I","E_I"]},
    },
}
schemes["3S_mod2.1"] = {
    "species": {
        "I": {"y0": concI0, "label": r"I"},
        "E": {"y0": concE0, "label": r"E"},
        "E_I": {"y0": 0, "label": r"E${\cdot}$I"},
        "EI": {"y0": 0, "label": r"E-I"},
    },
    "transitions": {
        "k1": {"k": kon, "source": ["2E", "I"], "target": ["E_I","E"], "label": r"$k_{on}$"},
        "k2": {"k": koff, "source": ["E_I","E"], "target": ["2E", "I"], "label": r"$k_{off}$"},
        "k3": {"k": kinactf, "source": ["E_I"], "target": ["EI"]}, #irrev step
        "k4": {"k": kinactb, "source": ["EI"], "target": ["E_I"]},
    },
}

schemes["3S_osc1"] = {
    "species": {
        "I": {"y0": concI0*1e-2, "label": r"I"},
        "E": {"y0": concE0, "label": r"E"},
        "E_I": {"y0": 0, "label": r"E${\cdot}$I"},
        "EI": {"y0": 0, "label": r"E-I"},
    },
    "transitions": {
        "k1": {"k": kon*8e1, "source": ["E", "0I"], "target": ["E_I"]},
        "k2": {"k": koff*5e2, "source": ["E_I"], "target": ["-E", "0I","E_I"]},
        "k3": {"k": kinactf*1, "source": ["E_I"], "target": ["EI"]},
        "k4": {"k": kinactf*1, "source": ["I"], "target": ["-1EI"]},
        "k5": {"k": kinactf*1, "source": ["EI"], "target": ["2E_I","I","E"]},
    },
    "dofit": False,
}

gk.utils.plotting.assign_colors_to_species(schemes,saturation_range=(0.5,0.8),lightness_range=(0.4,0.5),overwrite_existing=False,seed=1)

# Hardcoded 3S systems 
class ThreeStateVani():
    def __init__(self,Ki,koff,kinactf,kinactb,conc0Arr):
        self.Ki = Ki
        self.koff = koff
        self.kon = self.koff / Ki
        self.kinactf=kinactf
        self.kinactb=kinactb
        self.conc0Arr = conc0Arr
        self.sol=None

    @staticmethod
    def ode(t,concArr, params):
        kon,koff,kinactf,kinactb=params
        I, E, E_I, EI  = concArr
        dIdt = koff*E_I - kon*I*E
        dEdt = koff*E_I - kon*I*E 
        dE_Idt = kon*I*E - koff*E_I - kinactf*E_I + kinactb*EI
        dEIdt = kinactf*E_I - kinactb*EI
        return [dIdt, dEdt, dE_Idt, dEIdt]
        
    def solve(self, t):
        t_span = (t[0], t[-1])
        params = (self.kon, self.koff, self.kinactf, self.kinactb)
        solution = solve_ivp(self.ode, t_span, self.conc0Arr, t_eval=t, args=(params,), method='BDF', rtol=1e-6, atol=1e-8)
        return solution

class ThreeStateMod1dot1():
    def __init__(self,Ki,koff,kinactf,kinactb,conc0Arr):
        self.Ki = Ki
        self.koff = koff
        self.kon = self.koff / Ki
        self.kinactf=kinactf
        self.kinactb=kinactb
        self.conc0Arr = conc0Arr
        self.sol=None

    @staticmethod
    def ode(t,concArr, params):
        kon,koff,kinactf,kinactb=params
        I, E, E_I, EI  = concArr
        dIdt = 2*koff*E_I - 2*kon*(I**2)*E
        dEdt = koff*E_I - kon*(I**2)*E 
        dE_Idt = kon*(I**2)*E - koff*E_I - 3*kinactf*(E_I**3) + 3*kinactb*(EI**7)
        dEIdt = 7*kinactf*(E_I**3) - 7*kinactb*(EI**7)
        return [dIdt, dEdt, dE_Idt, dEIdt]
        
    def solve(self, t):
        t_span = (t[0], t[-1])
        params = (self.kon, self.koff, self.kinactf, self.kinactb)
        solution = solve_ivp(self.ode, t_span, self.conc0Arr, t_eval=t, args=(params,), method='BDF', rtol=1e-6, atol=1e-8)
        return solution

class ThreeStateMod1dot2():
    def __init__(self,Ki,koff,kinactf,kinactb,conc0Arr):
        self.Ki = Ki
        self.koff = koff
        self.kon = self.koff / Ki
        self.kinactf=kinactf
        self.kinactb=kinactb
        self.conc0Arr = conc0Arr
        self.sol=None

    @staticmethod
    def ode(t,concArr, params):
        kon,koff,kinactf,kinactb=params
        I, E, E_I, EI  = concArr
        dIdt = koff*E_I + koff*E_I - kon*I*I*E - kon*I*I*E
        dEdt = koff*E_I - kon*I*I*E 
        dE_Idt = kon*I*I*E - koff*E_I - kinactf*E_I*E_I*E_I - kinactf*E_I*E_I*E_I - kinactf*E_I*E_I*E_I + kinactb*(EI**7) + kinactb*(EI**7) + kinactb*(EI**7)
        dEIdt = 7*kinactf*E_I*E_I*E_I - 7*kinactb*(EI**7)
        return [dIdt, dEdt, dE_Idt, dEIdt]
        
    def solve(self, t):
        t_span = (t[0], t[-1])
        params = (self.kon, self.koff, self.kinactf, self.kinactb)
        solution = solve_ivp(self.ode, t_span, self.conc0Arr, t_eval=t, args=(params,), method='BDF', rtol=1e-6, atol=1e-8)
        return solution

class ThreeStateMod2dot1():
    def __init__(self,Ki,koff,kinactf,kinactb,conc0Arr):
        self.Ki = Ki
        self.koff = koff
        self.kon = self.koff / Ki
        self.kinactf=kinactf
        self.kinactb=kinactb
        self.conc0Arr = conc0Arr
        self.sol=None

    @staticmethod
    def ode(t,concArr, params):
        kon,koff,kinactf,kinactb=params
        I, E, E_I, EI  = concArr
        dIdt = koff*E_I*E - kon*I*(E**2) 
        dEdt = 2*koff*E_I*E - 2*kon*I*(E**2) + kon*I*(E**2) - koff*E_I*E
        dE_Idt = kon*I*(E**2) - koff*E_I*E - kinactf*E_I + kinactb*EI
        dEIdt = kinactf*E_I - kinactb*EI
        return [dIdt, dEdt, dE_Idt, dEIdt]
        
    def solve(self, t):
        t_span = (t[0], t[-1])
        params = (self.kon, self.koff, self.kinactf, self.kinactb)
        solution = solve_ivp(self.ode, t_span, self.conc0Arr, t_eval=t, args=(params,), method='BDF', rtol=1e-6, atol=1e-8)
        return solution
    
class ThreeStateOsc1():
    def __init__(self,Ki,koff,kinactf,kinactb,conc0Arr):
        self.Ki = Ki
        self.koff = koff
        self.kon = self.koff / Ki
        self.kinactf=kinactf
        self.kinactb=kinactb
        self.conc0Arr = conc0Arr
        self.conc0Arr[0] = self.conc0Arr[0]*1e-2 #I starts at 1
        self.sol=None

    @staticmethod
    def ode(t,concArr, params):
        kon,koff,kinactf,kinactb=params
        kon = kon*8e1
        koff = koff*5e2
        kinactf = kinactf*1
        
        I, E, E_I, EI  = concArr

        dIdt =  - 0 * kon * E * I**0\
                + 0 * koff * E_I\
                - kinactf * I\
                + kinactf * EI
        
        dEdt =  - kon * E * I**0\
                + -1 * koff * E_I\
                + kinactf * EI
        
        dE_Idt =  + kon * E * I**0\
                - koff * E_I\
                + koff * E_I\
                - kinactf * E_I\
                + 2 * kinactf * EI
        
        dEIdt =  + kinactf * E_I\
                + -1 * kinactf * I\
                - kinactf * EI
        
        return [dIdt, dEdt, dE_Idt, dEIdt]
        
    def solve(self, t):
        t_span = (t[0], t[-1])
        params = (self.kon, self.koff, self.kinactf, self.kinactb)
        solution = solve_ivp(self.ode, t_span, self.conc0Arr, t_eval=t, args=(params,), method='BDF', rtol=1e-6, atol=1e-8)
        return solution

# Old NState functions for making sure new versions produce the same results
def _ode_old(system, t, concentrations):
    """
    Compute the derivative of concentrations with respect to time.
    Can directly handle non-linear reactions (eg stoich coeff > 1)
    """ 
    ode_arr = np.zeros_like(concentrations)
    for tr in system.transitions.values():
        rate_constant = tr.k
        rate = rate_constant * np.prod([concentrations[system.species[sp_name].index] ** coeff for sp_name,coeff in tr.source])
        # Iterating through like this is beneficial because it captures stoichiometry that is evident in the list rather than coefficient 
            # (eg "source": ["E","E", "I"] is equal to "source": ["2E", "I"])
        for sp_name,coeff in tr.source:
            ode_arr[system.species[sp_name].index] -= coeff * rate
        for sp_name,coeff in tr.target:
            ode_arr[system.species[sp_name].index] += coeff * rate
    return ode_arr

def _solve_odes_old(system, t, method='BDF', rtol=1e-6, atol=1e-8, output_raw=False):
    """
    Solve the ODEs for the system.
    """
    conc0 = np.array([np.atleast_1d(sp.y0) for _, sp in system.species.items()])
    t_span = (t[0], t[-1])
    system.log_odes()

    try:
        solution = solve_ivp(
            fun=lambda t, conc: _ode_old(system, t, conc),
            t_span=t_span, y0=conc0, t_eval=t, method=method, rtol=rtol, atol=atol,
        )
        if not solution.success:
            raise RuntimeError("ODE solver failed: " + solution.message)

        system.t = t
        for name,data in system.species.items():
            data.simout["y"] = solution.y[data.index]

        system.logger.info("ODEs solved successfully.")
        if output_raw:
            return solution
        else:
            return
    except Exception as e:
        system.logger.error(f"Error in solving ODEs: {e}")
        raise

# Solve gekim systems
def solve_gekim_systems(schemes,plot_timecourse=False) -> None: 
    for name,scheme in schemes.items():
        system = gk.schemes.NState(scheme,quiet=True)
        #sol = _solve_odes_old(system,t,output_raw=True) #testing with old (nonvectorized) version
        system.set_simulator(gk.simulators.ODESolver)
        sol = system.simulator.simulate(t_eval=t,output_raw=True,atol=1e-8,rtol=1e-6)
        final_state = system.species["EI"].simout["y"]
        all_bound = system.sum_species_simout(blacklist=["E","I"])
        if scheme.get("dofit",True):
            fit_output = ii.kobs_uplim_fit_to_occ_final_wrt_t(
                t,final_state,nondefault_params={"Etot":{"fix":concE0}})
        else:
            fit_output = None
        schemes[name]["fit_output"] = fit_output
        schemes[name]["sol"] = sol
        
        if plot_timecourse:
            fig = plt.figure(figsize=(5, 3))

            # Time course of species 
            for species, sp_data in system.species.items():
                if species == "I":
                    continue
                plt.plot(t, sp_data.simout["y"], label=sp_data.label,color=sp_data.color)

            # All bound states
            all_bound = system.sum_species_simout(blacklist=["E","I"])
            plt.plot(t, all_bound,label='All Bound States',color="grey")

            # Fitted data
            plt.plot(t,  fit_output.y_fit,label=r"New Fit: $k_{\text{obs}}$ = "+
                    str(gk.utils.helpers.round_sig(fit_output.fitted_params["kobs"],3))+r" $\text{s}^{-1}$",ls='--', color="black")

            plt.xlabel('Time (s)')
            plt.ylabel('Concentration (nM)')
            plt.legend(frameon=False)
            plt.show()

# Solve hardcoded systems
def solve_hardcoded_system(system,t,concE0,dofit=True):
    sol = system.solve(t)
    final_state = sol.y[-1]
    if dofit:
        fit_output = ii.kobs_uplim_fit_to_occ_final_wrt_t(
            t,final_state,nondefault_params={"Etot":{"fix":concE0}})
    else: 
        fit_output = None
    system_dict = {
        "system": system,
        "sol": sol,
        "final_state": final_state,
        "fit_output": fit_output
    }
    return system_dict

def compare_systems(sys1_label,sys1_dict, sys2_label,sys2_dict, t, didfits=True) -> None:
    bad = False
    if np.allclose(sys1_dict["sol"].y, sys2_dict["sol"].y,rtol=1e-6):
        if didfits:
            if gk.utils.helpers.compare_dictionaries(sys1_dict["fit_output"].fitted_params,sys2_dict["fit_output"].fitted_params,rel_tol=1e-6,abs_tol=1e-8):
                print(f"GOOD\tsol and fit match: {sys1_label} and {sys2_label}","\n")
            else:
                print(f"BAD\tfit mismatch: {sys1_label} and {sys2_label}")
                print(f"{sys1_label}:\n",sys1_dict["fit_output"].fitted_params)
                print(f"{sys2_label}:\n",sys2_dict["fit_output"].fitted_params)
                print("\n")
                bad = True
        else:
            print(f"GOOD\tsol match: {sys1_label} and {sys2_label}","\n")
    else:
        print(f"BAD\tsol mismatch: {sys1_label} and {sys2_label}")
        print(f"{sys1_label}:\n",sys1_dict["sol"],"\n")
        print(f"{sys2_label}:\n",sys2_dict["sol"],"\n")
        print("\n")
        bad = True
    
    if bad:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        #axs[0].plot(t, sys1_dict["sol"][0], label="I")
        axs[0].plot(t, sys1_dict["sol"].y[1], label="E")
        axs[0].plot(t, sys1_dict["sol"].y[2], label="E_I")
        axs[0].plot(t, sys1_dict["sol"].y[3], label="EI")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Concentration (nM)")
        axs[0].set_title(sys1_label)
        axs[0].legend()

        #axs[1].plot(t, sys2_dict["sol"][0], label="I")
        axs[1].plot(t, sys2_dict["sol"].y[1], label="E")
        axs[1].plot(t, sys2_dict["sol"].y[2], label="E_I")
        axs[1].plot(t, sys2_dict["sol"].y[3], label="EI")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Concentration (nM)")
        axs[1].set_title(sys2_label)
        axs[1].legend()

        plt.tight_layout()
        plt.show()



def main() -> None:
    # Solve gekim systems
    solve_gekim_systems(schemes,plot_timecourse=PLOT)

    # Solve hardcoded systems
    conc0Arr = [concI0,concE0,0,0]
    hardcoded_vani = ThreeStateVani(Kd,koff,kinactf,kinactb,conc0Arr)
    vani_dict = solve_hardcoded_system(hardcoded_vani,t,concE0)
    hardcoded_mod1dot1 = ThreeStateMod1dot1(Kd,koff,kinactf,kinactb,conc0Arr)
    mod1dot1_dict = solve_hardcoded_system(hardcoded_mod1dot1,t,concE0)
    hardcoded_mod1dot2 = ThreeStateMod1dot2(Kd,koff,kinactf,kinactb,conc0Arr)
    mod1dot2_dict = solve_hardcoded_system(hardcoded_mod1dot2,t,concE0)
    hardcoded_mod2dot1 = ThreeStateMod2dot1(Kd,koff,kinactf,kinactb,conc0Arr)
    mod2dot1_dict = solve_hardcoded_system(hardcoded_mod2dot1,t,concE0)
    hardcoded_osc1 = ThreeStateOsc1(Kd,koff,kinactf,kinactb,conc0Arr)
    osc1_dict = solve_hardcoded_system(hardcoded_osc1,t,concE0,dofit=False)

    # Compare systems
    compare_systems("vani hardcoded", vani_dict, "vani gekim", schemes["3S_vani"], t)
    compare_systems("mod1.1 hardcoded", mod1dot1_dict, "mod1.1 gekim", schemes["3S_mod1.1"], t)
    compare_systems("mod1.2 hardcoded", mod1dot2_dict, "mod1.2 gekim", schemes["3S_mod1.2"], t)
    compare_systems("mod1.1 gekim", schemes["3S_mod1.1"], "mod1.2 gekim", schemes["3S_mod1.2"], t)
    compare_systems("mod2.1 hardcoded", mod2dot1_dict, "mod2.1 gekim", schemes["3S_mod2.1"], t)
    compare_systems("osc1 hardcoded", osc1_dict, "osc1 gekim", schemes["3S_osc1"], t,didfits=False)

if __name__ == "__main__":
    main()



    