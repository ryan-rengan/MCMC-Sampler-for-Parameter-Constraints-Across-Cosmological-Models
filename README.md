# Building a Python MCMC Sampler for Parameter Constraints Across Cosmological Models

This repository contains Python Markov Chain Monte Carlo (MCMC) analyses used to constrain cosmological parameters in the ΛCDM and XCDM cosmological models using observational measurements of the Hubble parameter, $H(z)$.

The project was developed during the 2026 Kansas State University Physics Research Experiences for Undergraduates program under the mentorship of Dr. Bharat Ratra.

---

## Project Overview

Observational cosmology uses measurements of the universe’s expansion rate to constrain theoretical cosmological models. In this project, observational $H(z)$ data are compared with predictions from two models:

- ΛCDM, which includes matter, spatial curvature, and a cosmological constant
- XCDM, which describes dark energy using a constant equation-of-state parameter $w_X$

The scripts use:

- [`emcee`](https://emcee.readthedocs.io/) for MCMC sampling
- [`GetDist`](https://getdist.readthedocs.io/) for marginalized statistics and confidence contours
- Gaussian priors on the Hubble constant $H_0$
- An analytically marginalized likelihood for $H_0$

Two Hubble constant priors are considered:

$H_0 = 68.0 \pm 2.8\ \mathrm{km\,s^{-1}\,Mpc^{-1}}$

and

$H_0 = 73.8 \pm 2.4\ \mathrm{km\,s^{-1}\,Mpc^{-1}}$

For each prior, the code:

1. Defines the cosmological expansion function.
2. Evaluates the marginalized likelihood.
3. Samples the cosmological parameter space.
4. Reports acceptance-fraction and autocorrelation diagnostics.
5. Calculates best-fit values and credible intervals.
6. Generates walker trace plots.
7. Generates combined confidence-contour plots.

---

## Cosmological Models

### ΛCDM

For the ΛCDM model, the dimensionless expansion function is

$E(z)=\frac{H(z)}{H_0} = \sqrt{ \Omega_{m0}(1+z)^3 +\Omega_{k0}(1+z)^2 +\Omega_{\Lambda}}$

where

$ \Omega_{k0}=1-\Omega_{m0}-\Omega_{\Lambda}$

The sampled parameters are:

- $\Omega_{m0}$: present-day matter density parameter
- $\Omega_{\Lambda}$: cosmological-constant density parameter

The priors used in the code are

$ 0 < \Omega_{m0} < 0.55 $

and

$0 < \Omega_{\Lambda} < 1.35$

### XCDM

For a spatially flat XCDM parametrization, the expansion function is

$E(z)=\sqrt{\Omega_{m0}(1+z)^3+(1-\Omega_{m0})(1+z)^{3(1+w_X)}}$

The sampled parameters are:

- $\Omega_{m0}$: present-day matter density parameter
- $w_X$: dark-energy equation-of-state parameter

The priors used in the code are

$0 < \Omega_{m0} < 0.55$

and

$-2 < w_X < -0.333$

The ΛCDM limit corresponds to

$w_X=-1.$

---

## Marginalized Likelihood

For a given cosmological model, the theoretical Hubble parameter is

$H_{\mathrm{th}}(z)=H_0E(z)$

The likelihood is analytically marginalized over $H_0$. The coefficients used in the marginalized chi-square expression are

$A=\sum_i \frac{E(z_i)^2}{\sigma_{H,i}^2}+\frac{1}{\sigma_{H_0}^2},$

$B=\sum_i \frac{E(z_i)H_{\mathrm{obs},i}}{\sigma_{H,i}^2}+\frac{\bar H_0}{\sigma_{H_0}^2},$

and

$C=\sum_i \frac{H_{\mathrm{obs},i}^2}{\sigma_{H,i}^2}+\frac{\bar H_0^2}{\sigma_{H_0}^2}.$

The marginalized statistic implemented in the scripts is

$\chi^2_{\mathrm{marg}}=C-\frac{B^2}{A}+\ln\left(A\sigma_{H_0}^2\right)+2\ln 2-2\ln\left[1+\mathrm{erf}\left(\frac{B}{\sqrt{2A}}\right)\right]$

---

## Repository Contents

```text
.
├── README.md
├── RedshiftHubble.xlsx
├── main_lcdm_combined.py
└── main_xcdm_combined.py
