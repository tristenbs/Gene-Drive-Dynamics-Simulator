#%%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Python model observing the impact of the mass release of genetically-modified mosquitoes into the wild.
# Modified mosquitoes carry a gene drive that biases the natural reproductive sex ratio towards males.

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
t = np.linspace(0,100)

# initial conditions
E_xu0 = 1000
E_xY0 = 1000
L_xu0 = 1000
L_xY0 = 1000
P_xu0 = 1000
P_xY0 = 1000
M_xY0 = 1000
M_xy0 = 1000
M_xx0 = 10000

def deriv(y,t, alpha, K_e, sig_egg, sig_larv, sig_pup, mu, v_f, psi):
    
    # setting our vars
    E_xu,E_xY,L_xu,L_xY,P_xu,P_xY,M_xx,M_xy,M_xY = y
    E = E_xu+E_xY
    L = L_xu+L_xY
    
    # egg stage
    dE_xu = (alpha * ((M_xy/(M_xY+M_xy + 1))) * (1.0 - max(E/K_e,0)) * (M_xx)) - ((sig_egg + mu) * E_xu)
    dE_xY = (alpha * (M_xY/(M_xy+M_xY + 1)) * (1 - max(E/K_e,0)) * (M_xx)) - ((sig_egg + mu) * E_xY)
    
    # larval stage
    dL_xu = (sig_egg * E_xu) - ((sig_larv + (psi * L) + mu) * L_xu)
    dL_xY = (sig_egg * E_xY) - ((sig_larv + (psi * L) + mu) * L_xY)
    
    # pupal stage
    dP_xu = (sig_larv * L_xu) - ((sig_pup + mu) * P_xu)
    dP_xY = (sig_larv * L_xY) - ((sig_pup + mu)* P_xY)
    
    # adult stage
    dM_xx = (sig_pup * P_xu * 0.5) - (mu_adult * M_xx)
    dM_xy = (sig_pup * P_xu * 0.5) - (mu_adult * M_xy)
    dM_xY = (sig_pup * P_xY) - (mu_adult * M_xY)
    
    return dE_xu,dE_xY,dL_xu,dL_xY,dP_xu,dP_xY,dM_xx,dM_xy,dM_xY


y0 = E_xu0,E_xY0,L_xu0,L_xY0,P_xu0,P_xY0,M_xx0,M_xy0,M_xY0

ret = odeint(deriv, y0, t, args=(alpha, K_e, sig_egg, sig_larv, sig_pup, mu, v_f, psi))
    
E_xu,E_xY,L_xu,L_xY,P_xu,P_xY,M_xx,M_xy,M_xY = ret.T

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

ax.plot(t, M_xx, 'b', alpha=0.5, lw=2, label='Mature Female Mosquitoes')

ax.plot(t, M_xY, 'r', alpha=0.5, lw=2, label='Mature Male Mosquitoes w/ gene drive')


ax.set_xlabel('Time /days')
ax.set_ylabel('# Mosquitoes')
ax.set_ylim(0,100000)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()



# %%
