import numpy as np
from .const import k_b, mu, m_p, Grav, pi, M_earth

def isolation_mass_loc(M_star, x, T, final_core_mass):
    from scipy.interpolate import interp1d
    
    c_s = np.sqrt(k_b * T / mu / m_p)
    v_K = np.sqrt(Grav * M_star / x)
    H_R = c_s / v_K
    isolation_mass = (H_R / 0.05)**3 * 20. * M_earth
    formation_loc = interp1d(isolation_mass, x, kind='cubic')(final_core_mass)
    
    return formation_loc
    

def accretion_rate(M_pl, M_star, loc, x, sigma_d, sigma_g, rho_s, res, a_0, T, alpha):
    a_1 = res[4]
    a_df, a_fr, a_dr = res[5], res[6], res[7]
    mask = np.array([adr < afr and adr < adf for adr,
                  afr, adf in zip(a_dr, a_fr, a_df)])
    fm = 0.75 * np.invert(mask) + 0.97 * mask
                  
    St_0 = pi / 2 * a_0 * rho_s / sigma_g
    St_1 = pi / 2 * a_1 * rho_s / sigma_g
    
    c_s = np.sqrt(k_b * T / mu / m_p)
    R_hill = np.power(M_pl / 3 / M_star, 1/3) * x
    Omega_K = np.sqrt(Grav * M_star / x**3)
    
    H_d0 = c_s / Omega_K * np.sqrt(alpha / (alpha + St_0))
    H_d1 = c_s / Omega_K * np.sqrt(alpha / (alpha + St_1))
    
    M_dot_2d_0 = 2 * np.power(np.minimum(St_0, 0.1)/0.1, 2/3) * R_hill**2 * Omega_K * sigma_d
    M_dot_2d_1 = 2 * np.power(np.minimum(St_1, 0.1)/0.1, 2/3) * R_hill**2 * Omega_K * sigma_d
    
    f_3d_0 = 1/2 * np.sqrt(pi/2) * np.power(St_0/0.1, 1/3) * R_hill / H_d0
    f_3d_0 = np.minimum(f_3d_0, 1)
    f_3d_1 = 1/2 * np.sqrt(pi/2) * np.power(St_1/0.1, 1/3) * R_hill / H_d1
    f_3d_1 = np.minimum(f_3d_1, 1)

    M_dot = M_dot_2d_0 * f_3d_0 * (1-fm) + M_dot_2d_1 * f_3d_1 * fm
    M_dot *= -1
    #accretion rate should be negative
    
    #need M_dot at location of planet, find nearest location in the array
    planet_pos_index = (np.absolute(x - loc)).argmin()
    vol_planet_pos = 0.5 * (x[planet_pos_index+1] - x[planet_pos_index-1])
    L = np.zeros_like(x)
    L[planet_pos_index] = M_dot[planet_pos_index] / (2 * pi * x[planet_pos_index] * 
                                            sigma_d[planet_pos_index] * vol_planet_pos)
    
    #K = np.zeros_like(x)
    #K[planet_pos_index] = M_dot[planet_pos_index] / (2 * pi * vol_planet_pos)
    
    return L, M_dot[planet_pos_index]
    