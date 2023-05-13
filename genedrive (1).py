#%%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Python model observing the impact of the mass release of genetically-modified mosquitoes into the wild.
# Modified mosquitoes carry a gene drive that aggressively propogates the spread of an altered doublesex gene that disrupts female fertility and biting.

#params (simplified)

alpha = 33.0# rate at which eggs are laid by female mosquitoes
K_e = pow(10.0,7.0) # environmental carrying capacity of eggs
sig_egg = (1/3) # development rate of eggs
sig_larv = (1/3) # development rate of larvae
sig_pup = (1/3) # development rate of pupae
mu = 1/(29) # averaged natural mosquito mortality rate
mu_adult = 1/3 # 30% of adult moquitoes die per day or something idk
v_f = .5 # proportion of adult mosquitoes that are female
psi = .00002 # density-dependent larval mortality rate
t = np.linspace(0,800)

# initial conditions
E_uu0 = 10000.0
E_du0 = 10000.0
E_dd0 = 10000.0
L_uu0 = 10000.0
L_du0 = 10000.0
L_dd0 = 10000.0
P_uu0 = 10000.0
P_du0 = 10000.0
P_dd0 = 10000.0
Mf_uu0 = 10000.0
Mf_du0 = 10000.0
Mf_dd0 = 10000.0
Mm_uu0 = 10000.0
Mm_du0 = 10000.0
Mm_dd0 = 10000.0

def deriv(y, t, alpha, K_e, sig_egg, sig_larv, sig_pup, mu, v_f, psi):
    # Setting our vars
    E_uu,E_du,E_dd,L_uu,L_du,L_dd,P_uu,P_du,P_dd,Mf_uu,Mf_du,Mf_dd,Mm_uu,Mm_du,Mm_dd = y
    Mf = Mf_dd + Mf_du + Mf_uu ; Mm = Mm_dd + Mm_du + Mm_uu
    E = E_dd + E_du + E_uu ; P = P_dd + P_du + P_uu ; L = L_dd + L_du + L_uu

    # Probability functions, scaled appropriately
    q_f = ((Mf_uu + (0.05 * Mf_du))/Mf) # u
    q_m = ((Mm_uu + (0.05 * Mm_du))/Mm) # u
    p_f = ((Mf_dd + (0.95 * Mf_du))/Mf) # d
    p_m = ((Mm_dd + (0.95 * Mm_du))/Mm) # d
    
    # egg stage
    dE_uu = (alpha * q_f * q_m * (1.0 - max(E/K_e,0)) * (Mf_du + Mf_uu)) - ((sig_egg + mu) * E_uu)
    dE_du = (alpha * ((p_f * q_m) + (p_m * q_f)) * (1 - max(E/K_e,0)) * (Mf_du + Mf_uu))- ((sig_egg + mu) * E_du)
    dE_dd = (alpha * p_f * p_m * (1 - max(E/K_e,0)) * (Mf_du + Mf_uu)) - ((sig_egg + mu) * E_dd)

    # larval stage
    dL_uu = (sig_egg * E_uu) - ((sig_larv + (psi * L) + mu) * L_uu)
    dL_du = (sig_egg * E_du) - ((sig_larv + (psi * L) + mu) * L_du)
    dL_dd = (sig_egg * E_dd) - ((sig_larv + (psi * L) + mu) * L_dd)

    # pupal stage
    dP_uu = (sig_larv * L_uu) - ((sig_pup + mu) * P_uu)
    dP_du = (sig_larv * L_du) - ((sig_pup + mu) * P_du)
    dP_dd = (sig_larv * L_dd) - ((sig_pup + mu)* P_dd)

    # adult females
    dMf_uu = (v_f * sig_pup * P_uu) - (mu * Mf_uu)
    dMf_du = (v_f * sig_pup * P_du) - (mu * Mf_du)
    dMf_dd = (v_f * sig_pup * P_dd) - (mu * Mf_dd)

    # adult males
    dMm_uu = ((1 - v_f) * sig_pup * P_uu) - (mu_adult * Mm_uu)
    dMm_du = ((1 - v_f) * sig_pup * P_du) - (mu_adult * Mm_du)
    dMm_dd = ((1 - v_f) * sig_pup * P_dd) - (mu_adult * Mm_dd)

    return dE_uu,dE_du,dE_dd,dL_uu,dL_du,dL_dd,dP_uu,dP_du,dP_dd,dMf_uu,dMf_du,dMf_dd,dMm_uu,dMm_du,dMm_dd

y0 = E_uu0,E_du0,E_dd0,L_uu0,L_du0,L_dd0,P_uu0,P_du0,P_dd0,Mf_uu0,Mf_du0,Mf_dd0,Mm_uu0,Mm_du0,Mm_dd0

ret = odeint(deriv, y0, t, args=(alpha, K_e, sig_egg, sig_larv, sig_pup, mu, v_f, psi))

E_uu,E_du,E_dd,L_uu,L_du,L_dd,P_uu,P_du,P_dd,Mf_uu,Mf_du,Mf_dd,Mm_uu,Mm_du,Mm_dd = ret.T


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

ax.plot(t, Mm_uu, 'b', alpha=0.5, lw=2, label='Homozygous Unaltered Adult Mosquitoes')
ax.plot(t, Mm_du, 'r', alpha=0.5, lw=2, label='Heterozygous Altered Adult Mosquitoes')
ax.plot(t, Mm_dd, 'g', alpha=0.5, lw=2, label='Homozygous Altered Adult Mosquitoes')

ax.set_xlabel('Time /days')
ax.set_ylabel('# Male Mosquitoes')
ax.set_ylim(0,100000)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
# weekend notes ->
# annoying error that needs debugging
# extensions would include 
# - adding probabilities for sex ratio (we cant just change v_f I think)
# - Incorporating time it takes to find a mate for females
# - Incorporating and comparing periodic addition of altered mosquitoes vs just dumping them at t = 0

# we gotta fix this bug first though
# also put this all into LaTeX
# Make mabels thicker, clearer, consistent color codes


# %%
