#!/usr/bin/env python
"""For a very small cylinder telescope (whose beam transfer matrices we can
generate in ~30 minutes on 2 nodes of PI's Symmetry cluster), how much integration
time do we need to have the thermal noise fall below the cosmological 21cm
signal?
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyccl as ccl

import pumanoise as pn

NU21 = 1420.4 # MHz

Ddish = 5.6 # m


class MyTel(pn.RadioTelescope):
    """Specs approximating small test telescope.

    The precise specs are 2 20m-wide cylinders with 16 feeds each, 1.2m apart,
    with T_recv = 50K. I'll approximate this with a square-packed 6x6 array of dishes with
    physical area 24m^2 (~5.6m diameter), observing over half the sky with the
    PUMA values for omtcoupling, skycoupling, and optical efficiency.

    Parameters
    ----------
    C : ccl.Cosmology class
        CCL class defining the background cosmology.
    tint : float
        On-sky integration time, in years.
    """
    def __init__ (self,C,tint):
        pn.RadioTelescope.__init__(self,C,Nside=6, D=Ddish, tint=tint, fsky=0.5, effic=0.7,
                                Tampl=50., Tground=300., omtcoupling=0.9, skycoupling=0.9,
                                hexpack=False)

        self.numax = 800. # MHz
        self.numin = 700. # MHz
        self.channel_width = 4. # MHz

        self.zmax = NU21/self.numax - 1.
        self.zmin = NU21/self.numin - 1.
        self.zmean = NU21/np.mean([self.numax,self.numin]) - 1.


# Background cosmology
C = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, sigma8=0.8)

# Telescopes with different integration times
times = [5,10,100,200] # y
rt = {}
for t in times:
    rt[t] = MyTel(C,t)

# Mean redshift of survey, for computing power spectra
z = rt[times[0]].zmean

# kperp, kpar values for evaluating power spectrum
nk = 20
kperp = np.logspace(np.log10(0.005),-1,nk)
kpar = np.logspace(np.log10(0.005),-1,nk)
kperp_2d = np.outer(np.ones(nk),kperp)
kpar_2d = np.outer(kpar,np.ones(nk))

# Thermal noise power spectra for different integration times
p_noise = {}
for t in times:
    p_noise[t] = rt[t].PNoise(z,kperp)

# Approximate 21cm power spectrum
f = ccl.growth_rate(C,1/(1+z))
p_m = ccl.linear_matter_power(C,kperp,1/(1+z))
mu = 0.
p_21 = rt[times[0]].Tb(z)**2 * p_m * (rt[times[0]].bias(z) + f* mu**2)**2

# Plot signal and noise power spectra
plt.loglog(kperp/C["h"],p_21*C["h"]**3,color="black",label="21cm")
for t in times:
    plt.loglog(kperp/C["h"],p_noise[t]*C["h"]**3,"--",label="Thermal noise for t = %d years" % t)
plt.xlabel(r"$k_\perp\;[h\,{\rm Mpc}^{-1}]$")
plt.ylabel(r"$P(k_\perp,z_{\rm mean})\; [{\rm K}^2 h^{-3}\,{\rm Mpc}^3]$")
plt.legend()
plt.title(r"%d %.1fm dishes, $T_{\rm recv}=50\,{\rm K}$, $z_{\rm mean}=%.2f$"
            % (rt[times[0]].Nside**2,rt[times[0]].D,z))
plt.savefig("small_tel_test/ps_comparison_d%.1f.pdf" % rt[times[0]].D)
