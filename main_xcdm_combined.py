"""
main_xcdm_combined.py

Runs the XCDM MCMC twice using the two H0 priors:
    1) H0 = 68.0, sigmaH0 = 2.8
    2) H0 = 73.8, sigmaH0 = 2.4

Then plots both XCDM confidence contour sets on the same PNG.
The H0 = 73.8 prior case is plotted with dot-dash lines.
"""

# -----------------------------
# Importing Packages
# -----------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
from scipy.special import erf
from getdist import plots, MCSamples

# -----------------------------
# Data
# -----------------------------
data = pd.read_excel("RedshiftHubble.xlsx")                         # Saves excel as "data"
z = data.iloc[:, 0].to_numpy()                                      # Redshift (First Column of Excel)
Hobs = data.iloc[:, 1].to_numpy()                                   # Observed Hubble parameter (Second Column of Excel)
sigmaH = data.iloc[:, 2].to_numpy()                                 # 1-sigma uncertainty on Hobs (Third Column of Excel)


# -----------------------------
# XCDM model (Theoretical)
# -----------------------------
def E_xcdm(z, omega_m, wX):
    """Return the XCDM expansion function E(z)."""
    omega_x = 1.0 - omega_m                                         # Solving for Omega_x based on Omega_m
    E2 = omega_m * (1 + z)**3 + omega_x * (1 + z)**(3 * (1 + wX))   # Expansion Function (squared)

    # Reject parameter choices that make E(z) imaginary.
    if np.any(E2 <= 0):                                             # tests whether at least one element in the E2 array evaluates to true if E2<0
        return None                                                 # Return None if there's a negative number (preventing complex numbers)

    return np.sqrt(E2)                                              # return square root of E2 (for E(z))

# -----------------------------
# Statistics
# -----------------------------
def chi2_eff_xcdm(theta, H0, sigmaH0):
    """
    Effective chi-square after analytically marginalizing over H0
    with a Gaussian H0 prior.

    Measures how well one particular model fits our observed data
        Small Chi2 = model agrees with data
        Large Chi2 = Model disagrees with data
    """
    omega_m, wX = theta
    E = E_xcdm(z, omega_m, wX)                                      # Expansion Function

    if E is None:                                                   
        return np.inf                                               # E = Infinity if invalid E(z)

    ## Coefficients for chi2
    A = np.sum(E**2 / sigmaH**2) + 1 / sigmaH0**2                   # Collects terms mulitplying square of unknown true Hubble Constant
    B = np.sum(E * Hobs / sigmaH**2) + H0 / sigmaH0**2              # Measures overlap between: Model Shape E, Observed Values Hobs, Prior value H0
    C = np.sum(Hobs**2 / sigmaH**2) + H0**2 / sigmaH0**2            # Contains terms involving the observations and the H0 prior by themselves

    ## Argument of the error function from Eq. 18
    erf_argument = B / np.sqrt(2.0 * A)

    ## Full -2 ln(L_H), including normalization from Eq. 18
    chi2_marginalized = (C - B**2 / A) + (np.log(A * sigmaH0**2)) + (2 * np.log(2.0)) - 2.0 * np.log1p(erf(erf_argument))
    
    if not np.isfinite(chi2_marginalized):                          # Last check for valid value
        return np.inf
    
    return chi2_marginalized                                        # Effective Chi Square Equation


def log_prior_xcdm(theta):
    """Flat prior matching the plotted axis range.
    
    The model is only accepted when 
        0 < OmegaM < 0.55
        -2 < wX < -0.333
    """
    omega_m, wX = theta

    if 0.0 < omega_m < 0.55 and -2 < wX < -0.333:                   # If the parameters are within range, return a log prior of 0.0 (equal probability for all allowed values)
        return 0.0

    return -np.inf                                                  # If the parameters are outside range, return negative infinity (zero probability for those values)                    


def log_probability_xcdm(theta, H0, sigmaH0):
    """Log posterior, proportional to -0.5 chi2_eff for allowed parameters.
    
    Main Function that MCMC sampler uses
    """
    lp = log_prior_xcdm(theta)

    if not np.isfinite(lp):                                         # Checks whether lp is finite
        return -np.inf

    chi2 = chi2_eff_xcdm(theta, H0, sigmaH0)

    if not np.isfinite(chi2):                                       # Checks whether chi2 is finite
        return -np.inf

    return lp - 0.5 * chi2                                          # Log version for Gaussian Measurement errors


# -----------------------------
# MCMC runner
# -----------------------------
def run_xcdm_case(
    case_name,                                                      # name for the run
    H0,                                                             # Value for Hubble Constant
    sigmaH0,                                                        # Uncertainty in Hubble Constant
    initial_guess,                                                  # Starting guess for cosmology parameters
    ## Default Values
        nwalkers=1000,                                               # Number of walkers MCMC uses
        nsteps=10000,                                               # Number of steps Walkers take
        burnin=1000,                                                # Discard first ____ steps. Reduces influence of initial guess
        thin=10,                                                    # Keeps every ___ step. Reduces number of stored samples, reduces correlation between neighboring saved samples
        seed=12345,                                                 # Keep same "random". Changing seed gives different random run.
):
    ##Run one XCDM MCMC case and return samples + summary info
    print(f"\nRunning {case_name}")                                 # Prints name of the run
    print("-" * (8 + len(case_name)))                               # Prints divider

    np.random.seed(seed)                                            # Initializes Numpy's random-number generator. Makes random starting positions reproducible

    ndim = 2                                                        # Number of dimensions of parameter space. Here two bc OmegaM & wX
    pos = initial_guess + 1e-3 * np.random.randn(nwalkers, ndim)    # Creates the initial location of every walker

    ## MCMC proposal moves
    # Uses the ensemble's own spread to propose steps, which tracks curved degeneracies far more efficiently than StretchMove.
    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]

    ## MCMC SAMPLER
    sampler = emcee.EnsembleSampler(
        nwalkers,                                                   # Tells emcee how many walkers to use
        ndim,                                                       # How many parameters each walker carries
        log_probability_xcdm,                                       # Evaluates each proposed position
        args=(H0, sigmaH0),                                         # Fixed values that do not change
        moves=moves                                                 # Explicit proposal-move mixture
    )

    # Starts MCMC Calculation
    sampler.run_mcmc(pos, nsteps, progress=True)                    # Runs sampler for nsteps, starting from pos, and displays a progress bar (we love a good progress bar)

    # -----------------------------------------
    # Acceptance-fraction diagnostic
    # -----------------------------------------
    mean_acceptance_fraction = np.mean( sampler.acceptance_fraction)    # Calculates the mean acceptance fraction across all the initialized walkers
    minimum_acceptance_fraction = np.min(sampler.acceptance_fraction)   # Calculates the minimum acceptance fraction across all the initialized walkers
    maximum_acceptance_fraction = np.max(sampler.acceptance_fraction)   # Calculates the maximum acceptance fraction across all the initialized walkers

    ## Printing Acceptance Fraction Diagnostics
    print("\nAcceptance-fraction diagnostic")
    print("------------------------------")
    print(
        f"Mean acceptance fraction: "
        f"{mean_acceptance_fraction:.4f}"
    )
    print(
        f"Walker acceptance range: "
        f"{minimum_acceptance_fraction:.4f} to "
        f"{maximum_acceptance_fraction:.4f}"
    )

    # -----------------------------------------
    # Autocorrelation-time convergence check
    # -----------------------------------------
    post_burnin_steps = nsteps - burnin                             # Number of steps after burn-in

    print("\nAutocorrelation-time diagnostic")
    print("--------------------------------")

    try:                                                            # Try to calculate autocorrelation time. If fails, catch error and print warning message.
        autocorr_times = sampler.get_autocorr_time(                 # Calculates autocorrelation time for each parameter. Measures of how many steps it takes for chain to "forget" its previous state.
            discard=burnin, thin=1)

        parameter_names = [
            "Omega_m",
            "wX"
        ]

        all_parameters_pass = True                                  # Track if all parameters pass convergence test

        for parameter_name, tau in zip(
            parameter_names,
            autocorr_times
        ):
            required_steps = 50.0 * tau                             # Recommended minimum number of steps for convergence (50 times the autocorrelation time)
            chain_length_over_tau = post_burnin_steps / tau         # Ratio of post-burn-in chain length to autocorrelation time

            approximate_effective_samples = (
                nwalkers * post_burnin_steps / tau                  # Approximate number of independent samples in the chain
            )

            passes_convergence_test = (
                post_burnin_steps >= required_steps                 # Check if post-burn-in chain length is at least 50 times autocorrelation time
            )

            if not passes_convergence_test:
                all_parameters_pass = False                         # If any parameter fails, set convergence status to False (warning)

            ## Printing Diagnostics for each Parameter
            print(f"{parameter_name}:")
            print(
                f"  Autocorrelation time, tau:       "
                f"{tau:.2f} steps"
            )
            print(
                f"  Post-burn-in chain length:       "
                f"{post_burnin_steps}"
            )
            print(
                f"  Recommended minimum, 50*tau:    "
                f"{required_steps:.2f}"
            )
            print(
                f"  Chain length / tau:              "
                f"{chain_length_over_tau:.2f}"
            )
            print(
                f"  Approximate effective samples:  "
                f"{approximate_effective_samples:.0f}"
            )

            ## Printing Convergence Status for each parameter
            if passes_convergence_test:
                print(
                    "  Status: PASS — chain is at least "
                    "50 autocorrelation times long."
                )
            else:
                print(
                    "  Status: WARNING — chain is shorter "
                    "than 50 autocorrelation times."
                )

            print()

        ## Printing Overall Convergence Status for all parameters
        if all_parameters_pass:
            print("Overall convergence status: PASS")
        else:
            print("Overall convergence status: WARNING")
            print(
                "Increase nsteps before trusting the "
                "final contours."
            )

    except emcee.autocorr.AutocorrError as error:                   # Catching error if emcee fails to estimate the autocorrelation time and print warning message
        print(
            "WARNING: emcee could not reliably estimate "
            "the autocorrelation time."
        )
        print(
            "This usually means the chain is too short "
            "for a stable estimate."
        )
        print("Increase nsteps and run the sampler again.")
        print()
        print(error)            

    # -----------------------------------------
    # Trace plots for representative walkers
    # -----------------------------------------
    full_chain = sampler.get_chain()                                # Retrieves chain of samples from the sampler, including all walkers/steps

    parameter_labels = [
        r"$\Omega_{m0}$",
        r"$w_X$"
    ]

    number_of_walkers_to_plot = 20                                  # 20 walkers for visualization (increase to see more walkers and decrease for less)

    walker_indices = np.linspace(                                   # Selects evenly spaced walker indices to plot
        0,
        nwalkers - 1,
        number_of_walkers_to_plot,
        dtype=int
    )

    fig, axes = plt.subplots(                                       # Creates subplots for each parameter
        ndim,
        1,
        figsize=(10, 6),
        sharex=True
    )

    step_numbers = np.arange(nsteps)                                # Creates an array of step numbers from 0 to nsteps-1

    ## Plotting the traces for each parameter/walker (formatting, labeling, etc.)
    for parameter_index, axis in enumerate(axes):
        for walker_index in walker_indices:
            axis.plot(                                              # Plots trace of each walker for current parameter
                step_numbers,
                full_chain[
                    :,
                    walker_index,
                    parameter_index
                ],
                alpha=0.45,
                linewidth=0.7
            )

        axis.axvline(                                               # Draws vertical line to indicate burn-in period (important for convergence)
            burnin,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Burn-in ends at step {burnin}"
        )

        axis.set_ylabel(parameter_labels[parameter_index])          # Sets y-axis label for current parameter

        axis.legend(
            loc="upper right",
            fontsize=8
        )

        axis.grid(alpha=0.2)                                        # Grid lines for better visualization

    axes[-1].set_xlabel("Step number")                              # Sets x-axis label for the last subplot (step number)

    fig.suptitle(
        f"Walker trace plots: {case_name}",
        fontsize=12
    )

    plt.tight_layout()                                              # Adjusts spacing for better layout

    H0_filename = f"{H0:.1f}".replace(".", "p")
    trace_filename = f"plots/xcdm_trace_H0_{H0_filename}.png"  

    plt.savefig(
        trace_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved trace plot to: {trace_filename}")                 # Confirmation message

    # -----------------------------------------
    # Extract post-burn-in samples
    # -----------------------------------------
    samples = sampler.get_chain(                                    # Assembles post burn samples into a single array
        discard=burnin,
        thin=thin,
        flat=True
    )

    log_probs = sampler.get_log_prob(                               # Assembles post burn log probabilities into a single array
        discard=burnin,
        thin=thin,
        flat=True
    )

    # -----------------------------------------
    # Best-fit parameters
    # -----------------------------------------
    best_index = np.argmax(log_probs)                               # Finds index of the maximum log probability (best-fit parameters)
    best_params = samples[best_index]                               # Retrieves best-fit parameters corresponding to the maximum log probability

    min_chi2 = chi2_eff_xcdm(best_params, H0, sigmaH0)              # Calculates minimum chi-square value for the best-fit parameters  

    ## Printing these results
    print("\nBest-fit results")
    print("----------------")
    print(
        f"Best-fit Omega_m: "
        f"{best_params[0]:.5f}"
    )
    print(
        f"Best-fit wX:      "
        f"{best_params[1]:.5f}"
    )
    print(
        f"Minimum marginalized chi2_H:  "
        f"{min_chi2:.5f}"
    )

    # -----------------------------------------
    # Raw percentile summary
    # -----------------------------------------
    print("\nCentral values and sigma intervals")
    print("----------------------------------")

    labels = [
        "Omega_m",
        "wX"
    ]

    median_params = np.percentile(                                  # Defines median values for parameters
        samples,
        50,
        axis=0
    )

    for i, label in enumerate(labels):
        q16, q50, q84 = np.percentile(                              # Defines percent values for 16%, 50%, 84%
            samples[:, i],  
            [16, 50, 84]
        )

        q2p3, q97p7 = np.percentile(                                # Defines percent values for 2.275%, 97.725%
            samples[:, i],
            [2.275, 97.725]
        )

        q0p135, q99p865 = np.percentile(                            # Defines percent values for 0.135%, 99.865%
            samples[:, i],
            [0.135, 99.865]
        )

        ## Prints percent values (corresponds to 1-3 sigmas)
        print(f"{label}:")
        print(f"  Median  = {q50:.5f}")
        print(
            f"  1 sigma = {q50:.5f} "
            f"+{q84 - q50:.5f} "
            f"-{q50 - q16:.5f}"
        )
        print(
            f"  2 sigma = {q50:.5f} "
            f"+{q97p7 - q50:.5f} "
            f"-{q50 - q2p3:.5f}"
        )
        print(
            f"  3 sigma = {q50:.5f} "
            f"+{q99p865 - q50:.5f} "
            f"-{q50 - q0p135:.5f}"
        )
        print()

    # -----------------------------------------
    # GetDist samples
    # -----------------------------------------
    gd_samples = MCSamples(                                         # Main function for getDist
        samples=samples,                                           
        loglikes=-log_probs,
        names=[
            "omegam",
            "wX"
        ],
        labels=[
            r"\Omega_{m0}",
            r"w_X"
        ],
        label=case_name,
        ranges={
            "omegam": [0.0, 0.55],
            "wX": [-2.0, -0.333]
        }
    )

    gd_samples.updateSettings({                                     # Defines setting for our plot
        "smooth_scale_2D": 0.9,
        "num_bins_2D": 64,
        "contours": [
            0.682689492137,
            0.954499736104,
            0.997300203937
        ]
    })

    ## Prints stats we want
    print("\nGetDist marginalized statistics")
    print("-------------------------------")
    print(gd_samples.getMargeStats())

    return {
        "case_name": case_name,
        "H0": H0,
        "sigmaH0": sigmaH0,
        "samples": samples,
        "gd_samples": gd_samples,
        "best_params": best_params,
        "min_chi2": min_chi2,
        "median_params": median_params
    }


# -----------------------------
# Main script
# -----------------------------
def main():
    '''
    Function that actually puts everything together
    Runs the analysis twice, compares the two results on one graph, makrs the median values, saves graph
    '''
    os.makedirs("plots", exist_ok=True)                             # Creates a directory called "plots" if it doesn't already exist

    # H0 = 68.0, sigmaH0 = 2.8 case
    current_H0_case = run_xcdm_case(
        case_name=r"$H_0 = 68.0 \pm 2.8$",                          # Defines case name
        H0=68.0,                                                    # Value for Hubble Constant
        sigmaH0=2.8,                                                # Value for Uncertainty in Hubble Constant
        initial_guess=np.array([0.29, -1.04]),                       # Starting guess for cosmology parameters
        seed=12345,                                                 # Keep same "random". Changing seed gives different random run.
    )

    # H0 = 73.8, sigmaH0 = 2.4 case
    prior_H0_case = run_xcdm_case(
        case_name=r"$H_0 = 73.8 \pm 2.4$",                          # Defines case name
        H0=73.8,                                                    # Value for Hubble Constant
        sigmaH0=2.4,                                                # Value for Uncertainty in Hubble Constant
        initial_guess=np.array([0.26, -1.30]),                       # Starting guess for cosmology parameters (different from the first case, values taken from Farooq et al. 2013)
        seed=54321,                                                 # Keep same "random". Changing seed gives different random run.
    )

    # Creates plot object "g". Has 3 contour levels. Sets the legend font size.
    g = plots.get_single_plotter()
    g.settings.num_plot_contours = 3
    g.settings.legend_fontsize = 10

    ## Plots both MCMC runs on the same plot. The H0 = 73.8 prior case is plotted with dot-dash lines.
    g.plot_2d(
        [current_H0_case["gd_samples"], prior_H0_case["gd_samples"]],# Two Samples to plot. First is H0=68.0, second is H0=73.8
        "omegam",
        "wX",
        filled=False,
        colors=["blue", "red"],
        line_args=[
            {"lw": 1.0, "ls": "-"},                                  # H0 = 68.0 case: line width 2, line style normal solid
            {"lw": 1.0, "ls": "-"},                                 # H0 = 73.8 prior case: line width 2, line style dot-dash
        ],
        legend_labels=[                                              # Labels for the legend. The r before the string indicates that it's a raw string, useful for LaTeX formatting.
            r"$H_0 = 68.0 \pm 2.8$",
            r"$H_0 = 73.8 \pm 2.4$",
        ],
    )

    plt.xlabel(r"$\Omega_{m0}$")                                    # Label for x-axis
    plt.ylabel(r"$w_X$")                                            # Label for y-axis
    plt.xlim(0, 0.55)                                               # Domain for x-axis (Omega_m)
    plt.ylim(-2, -0.333)                                            # Range for y-axis (wX)

    # Median Point for Current H0 Case (H0 = 68.0)
    plt.scatter(
        current_H0_case["best_params"][0],                          # Omega_m best value for H0=68.0 case 
        current_H0_case["best_params"][1],                          # wX best value for H0=68.0 case
        color="blue",
        s=50,  
        marker="o",                                                 # Circle
        label=r"Best fit, $H_0 = 68.0 \pm 2.8$"
    )
    # Median Point for Prior H0 Case (H0 = 73.8)
    plt.scatter(
        prior_H0_case["best_params"][0],
        prior_H0_case["best_params"][1],
        color="red",
        s=50,
        marker="o",                                                 # Circle
        label=r"Best fit, $H_0 = 73.8 \pm 2.4$"
    )

    # Flat Universe Line 
    plt.plot([0, 0.55], [-1, -1], color="gray", ls=":", lw=0.9, label=r"Flat ΛCDM Line")                         # Plots the line for a flat universe
    
    # Accelerating Universe Line
    x = np.linspace(0, 0.55, 250)  # Create an array of 250 points between 0 and 0.55 for Omega_m
    y = -1 / (3 * (1 - x))  # Calculate the corresponding wx values for the accelerating/decelerating universe line
    plt.plot(x, y, color="gray", ls="--", lw=0.9, label=r"Accelerating/Decelerating Universe")  # Plots accelerating/decelerating universe line


    plt.legend(                                                     # Display legend on the plot.
        loc="lower left",                                           # Legend location
        fontsize=6.0                                                # Font size for the legend text
    )                                      

    output_path = "plots/xcdm_combined_contours.png"                # Output path for the saved plot
    plt.savefig(output_path, dpi=400, bbox_inches="tight")          # Saves the plot to the specified path with a resolution of 400 dpi and tight bounding box to minimize whitespace
    print(f"\nSaved combined plot to: {output_path}")               # Print the saved location

if __name__ == "__main__":                                          # Tells python to actually call main() when this script is ran directly
    main()
