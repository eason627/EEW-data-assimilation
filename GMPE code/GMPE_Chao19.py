
"""
Python implementation of the Chao et al. (2020) Taiwan horizontal GMPE.

Reference
---------
Chao, S.-H., Chiou, B., Hsu, C.-C., and Lin, P.-S. (2020),
"A horizontal ground-motion model for crustal and subduction earthquakes
in Taiwan", Earthquake Spectra.
https://doi.org/10.1177/8755293019891711

Implementation and attribution
------------------------------
The original file credits Huang Yi-Sheng and records its creation as
Mon Mar 4 09:28:19 2024. Its implementation notes describe a translation
from MATLAB to Python. This is a local implementation, not an OpenQuake
wrapper. The historical module/function names (GMPE_Chao19 / Chao19H)
are retained; the model reference and coefficient file identify Chao 2020.

Coefficients and dependencies
----------------------------
GMPE_Chao20_model_coefficients.csv must remain alongside this module.
The table is resolved relative to __file__, independently of the caller's
working directory. Requires NumPy, pandas, SciPy and Matplotlib.

Inputs and units
----------------
Chao19H takes period T in seconds (T=0 for PGA), magnitude M, rupture
distance Rrup and rupture-top depth Ztor in km, Vs30 in m/s and Z10 in m.
Z10=-999 selects the empirical reference depth in this implementation.
Fault-style flags FRO/FSS/FNO select reverse/strike-slip/normal crustal
terms; Finter/Fintra select interface/intraslab terms. Fas and Fma flag
aftershocks and the Manila subduction term. FVS30 selects the Vs30 source
category (0: Kuo17, 1: KS17, 2: receiver function).

Returns and project usage
-------------------------
Chao19H returns (Sa, tau, phis2s, phiss), arrays shaped (sites, periods).
For PGA and spectral acceleration, Sa is in g, not gal; the other outputs
are between-event, between-site and single-station natural-log standard
deviations. Unit conversions and project-specific scale factors are
applied by the notebook, outside this module.

The publication notebook calls Chao19H for Taiwan events. It passes
hypocentral distance (SRhypo) to Rrup as its point-source approximation;
the model argument itself represents rupture distance. The file also
retains legacy Chao19H_DIPI directivity routines, which the publication
notebook does not call. These routines are separate from the azimuth
terms learned by the notebook's Kalman filter.
"""
#%%
import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

# Coefficient file lives next to this module; resolve it relative to __file__
# so the import works regardless of the caller's working directory.
_COEF_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "GMPE_Chao20_model_coefficients.csv")
import matplotlib.pyplot as plt

#%% 

#This is original GMPE function, and I just mofify from the m file to python code
def Chao19H(T,M,Rrup,Ztor,Vs30,Z10,FRO,FSS,FNO,Finter,Fintra,Fas,Fma,FVS30):
    
    # Input definition and size
    # T: period nt x 1
    # M: Magnitude nr x 1
    # Rrup: Distance in km unit nr x 1
    # Ztor: depth to top of rupture plane in km unit (-999 as default) nr x 1
    # Vs30: Vs30 in m/sec unit nr x 1
    # Z1.0: depth to Vs30 == 1000 m/s in m unit (-999 as default) nr x 1
    # FRO: flag for the R/RO fault of crustal earthquake (0/1) nr x 1
    # FSS: flag for the SS fault of crustal earthquake event (0/1) nr x 1
    # FNO: flag for the N/NO fault of crustal earthquake event (0/1) nr x 1
    # Finter: flag for the interface earthquake (0/1) nr x 1
    # Fintra: flag for the intraslab earthquake (0/1) nr x 1
    # Fas: flag for the aftershock (0/1) nr x 1
    # Fma: flag for the Manila Subduction (0/1) nr x 1
    # FVS30: flag for the source of Vs30 (0 for Kuo17 (measured), 1 for KS17
    # (inferred), 2 for Receiver Function (inferred)) 1 x 1
    #
    # Output
    # Sa: spectral acceleration in unit g nr x nt
    # tau: between event std. in ln(g) unit nr x nt
    # phis2s: between site std. in ln(g) unit nr x nt
    # phiss: single station std. in ln(g) unit nr x nt

    """Evaluate the base Chao et al. (2020) model.

    T=0 selects PGA. Distances/depth Ztor are in km, Vs30 in m/s,
    and Z10 in m (-999 uses the empirical reference). See the module
    notes for fault and site flags and the notebook's distance convention.

    Returns Sa, tau, phis2s, phiss with shape (sites, periods).
    For PGA/SA, Sa is in g; standard deviations are natural-log values.
    """
    period=np.array([0,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2,3,4,5,-1,-2])

    if type(M) == int or type(M) == float:
        len_M = 1
    else:
        len_M = len(M)
    if type(Rrup) == int or type(Rrup) == float:
        len_Rrup = 1
    else:
        len_Rrup = len(Rrup)
        
    NR = max([len_M , len_Rrup])
    
    
    if type(T) == int or type(T) == float:
        T = [T]
        NT = 1
    else:
        NT = len(T)
       
    
    Sa = np.zeros(shape=(NR,NT))
    tau =  np.zeros(shape=(NR,NT))
    phis2s =  np.zeros(shape=(NR,NT))
    phiss =  np.zeros(shape=(NR,NT))
        
    
    for I in range(NT):

        T_low = max(period[period <= T[I]])
        T_high = min(period[period >= T[I]])
        ip_low  = np.where( period == T_low)[0][0]
        ip_high = np.where( period == T_high)[0][0]
        if ip_low == ip_high :

            Sa[:, I], tau[:, I], phis2s[:, I], phiss[:, I] = Chao19H_sub(ip_low, M, Rrup,\
                                                                         Ztor, Vs30, Z10,\
                                                                         FRO, FSS, FNO, \
                                                                         Finter, Fintra, \
                                                                         Fas, Fma, FVS30)
        else:

            Sa_low, tau_low, phis2s_low, phiss_low = Chao19H_sub(ip_low, M, Rrup, Ztor, Vs30, Z10, FRO, FSS, FNO, Finter, Fintra, Fas, Fma, FVS30)
            Sa_high, tau_high, phis2s_high, phiss_high = Chao19H_sub(ip_high, M, Rrup, Ztor, Vs30, Z10, FRO, FSS, FNO, Finter, Fintra, Fas, Fma, FVS30)
            
            x = np.array([np.log(T_low), np.log(T_high)])
            Y_sa = np.array([np.log(Sa_low), np.log(Sa_high)])
            Y_tau = np.array([tau_low, tau_high])
            Y_phis2s = np.array([phis2s_low, phis2s_high])
            Y_phiss = np.array([phiss_low, phiss_high])
            Sa[:, I] = np.exp(interp1d(x,  Y_sa.T,axis=-1 ,kind='linear')(np.log(T[I])))
            tau[:, I] = interp1d(x, Y_tau.T,axis=-1, kind='linear')(np.log(T[I]))
            phis2s[:, I] = interp1d(x, Y_phis2s.T,axis=-1, kind='linear')(np.log(T[I]))
            phiss[:, I] = interp1d(x, Y_phiss.T,axis=-1, kind='linear')(np.log(T[I]))

            
    return Sa,tau,phis2s,phiss

def Chao19H_sub(iT, M, Rrup, Ztor, Vs30, Z10, FRO, FSS, FNO, Finter, Fintra, Fas, Fma, FVS30):
    
    
    GMPE_coefficients = pd.read_csv(_COEF_CSV,\
                                    index_col=["T"])
        
    par = GMPE_coefficients.loc[:,:"c30"].to_numpy()
    par1 = GMPE_coefficients.loc[:,"taucr1":"phis2s"].to_numpy()
    
    
    # Constants
    C4inter = 0.3
    C4intra = 0.2
    Mc = 7.1
    Mref = 6.5
    n = 2
    Vs30ref = 760
    Rrupref = 0

    Z10ref = np.exp(-4.08 / 2 * np.log((Vs30 ** 2 + 355.4 ** 2) / (1750 ** 2 + 355.4 ** 2)))
    Hcrref = 0
    Hinterref = 0
    Hintraref = 35

    # Compute Fcr and Fsb
    Fcr = (FRO + FSS + FNO)
    Fsb = Finter + Fintra
    
    # If Z10 is not provided (-999 default, scalar or vector), set it to Z10ref
    Z10 = np.where(np.asarray(Z10) == -999, Z10ref, Z10)
    #h=Fcr*10+Finter*10*np.exp(C4inter*(M-Mc)*np.heaviside(M-Mc))+Fintra*10*exp(C4intra*(M-Mc).*heaviside(M-Mc));
    # Calculate h
    h = (Fcr * 10 + Finter * 10 * np.exp(C4inter * (M - Mc) * (M > Mc)) + 
         Fintra * 10 * np.exp(C4intra * (M - Mc) * (M > Mc))) 
  
    # Assign FVS30 indicators
    if FVS30 == 0:
        Fvs30kuo17 = 1
        Fvs30ks17 = 0
        Fvs30rf = 0
    elif FVS30 == 1:
        Fvs30kuo17 = 0
        Fvs30ks17 = 1
        Fvs30rf = 0
    elif FVS30 == 2:
        Fvs30kuo17 = 0
        Fvs30ks17 = 0
        Fvs30rf = 1
    else:
        Fvs30kuo17 = 1
        Fvs30ks17 = 0
        Fvs30rf = 0
        print('FVS30 Input Error ! Use Vs30 from Kuo17')
    
    # Initialize Xt matrix
    if type(Rrup) == int or type(Rrup) == float:
        len_Rrup = 1
    else:
        len_Rrup = len(Rrup)
        
     
    NR = len_Rrup
    Xt = np.zeros((NR, 30))
    # Assign values to Xt columns
    Xt[:, 0] = FRO
    Xt[:, 1] = FSS
    Xt[:, 2] = FNO 
    Xt[:, 3] = Finter
    Xt[:, 4] = Fintra
    Xt[:, 5] = Fas
    Xt[:, 6] = Fma    
    Xt[:, 7] = (Fcr * (M - Mref))
    Xt[:, 8] = (Fsb * (M - Mref)) 
    Xt[:, 9] = (Fcr * (M - Mref)**2 - Fcr * (M - 7.6)**2 * (M > 7.6))
    Xt[:, 10] = (Fcr* (5 - M) * (M < 5))
    Xt[:, 11] = (Fsb * (5 - M) * (M < 5))
    Xt[:, 12] = (Fsb * (6 - M) * (M < 6))
    Xt[:, 13] = (Fcr * (Ztor - Hcrref))
    Xt[:, 14] = (Finter * (Ztor - Hinterref))
    Xt[:, 15] =( Fintra * (Ztor - Hintraref)) 
    Xt[:, 16] = Fcr * np.log((Rrup**n + h**n)**(1/n) / (Rrupref ** n + h**n)**(1/n));  
    Xt[:, 17] = (Fsb * np.log((Rrup**n + h**n)**(1/n) / (Rrupref**n + h**n)**(1/n)))
    Xt[:, 18] =  (Fcr * (M - Mref) * np.log((Rrup**n + h**n)**(1/n) / (Rrupref**n + h**n)**(1/n)))
    Xt[:, 19] = (Fsb * (np.minimum(M, Mc) - Mref) * np.log((Rrup**n + h**n)**(1/n) / (Rrupref**n + h**n)**(1/n)))
    Xt[:, 20] = (Fcr  * (Rrup - Rrupref))
    Xt[:, 21] =  (Fsb * (Rrup - Rrupref))
    Xt[:, 22] = 0
    Xt[:, 23] = np.log(1180 / Vs30ref) 
    Xt[:, 24] = 0  # log(Z10 / Z10ref)
    Xt[:, 25] = Fvs30kuo17
    Xt[:, 26] = Fvs30ks17
    Xt[:, 27] = Fvs30rf
    
    
    Xt[:, 28] = (Finter * (M - Mc) * (M > Mc)) 
    Xt[:, 29] =  (Fintra * (M - Mc) * (M > Mc))
    
    # Calculate Sa1180

    Sa1180 = np.exp(np.dot(Xt, par[iT]))

    # Change c value for PGV prediction (iT is 0-based here; PGV row index is 20, PGD is 21)
    if iT > 19:
        c = 2400
    else:
        c = 2.4

    Xt[:, 22] = (-1.5 * np.log((Vs30 / Vs30ref)) - np.log(Sa1180 + c) + \
                 np.log(Sa1180 + c * (Vs30 / Vs30ref) ** 1.5)) * (Vs30ref > Vs30)
    Xt[:, 23] = np.log(Vs30 / Vs30ref)
    Xt[:, 24] = np.log(Z10 / Z10ref)


    median = np.exp(np.dot(Xt, par[iT]))

    # Calculate tau, phiss, and phis2s

    fM = 0.5 * (np.minimum(6.5, np.maximum(4.5, M)) - 4.5)
    tau = Fcr* (par1[iT, 0] + (par1[iT, 1] - par1[iT, 0]) * fM) + \
          Fsb * (par1[iT, 2] + (par1[iT, 3] - par1[iT, 2]) * fM)
    phiss = Fcr* (par1[iT, 4] + (par1[iT, 5] - par1[iT, 4]) * fM) + \
            Fsb * (par1[iT, 6] + (par1[iT, 7] - par1[iT, 6]) * fM)
    phis2s = par1[iT,8] * np.ones(shape = (NR,))

    return median,tau,phis2s,phiss



#%%

#This GMPE function consider about the directivity term thich is developed by Chao et al.

def Chao19H_DIPI(T,M,Rrup,Ztor,Vs30,Z10,FRO,FSS,FNO,Finter,Fintra,Fas,Fma,FVS30,\
                 DI,deg,arfad,rv,e,phi,PI,Tp):
    
    # Input definition and size
    # T: period nt x 1
    # M: Magnitude nr x 1
    # Rrup: Distance in km unit nr x 1
    # Ztor: depth to top of rupture plane in km unit (-999 as default) nr x 1
    # Vs30: Vs30 in m/sec unit nr x 1
    # Z1.0: depth to Vs30 == 1000 m/s in m unit (-999 as default) nr x 1
    # FRO: flag for the R/RO fault of crustal earthquake (0/1) nr x 1
    # FSS: flag for the SS fault of crustal earthquake event (0/1) nr x 1
    # FNO: flag for the N/NO fault of crustal earthquake event (0/1) nr x 1
    # Finter: flag for the interface earthquake (0/1) nr x 1
    # Fintra: flag for the intraslab earthquake (0/1) nr x 1
    # Fas: flag for the aftershock (0/1) nr x 1
    # Fma: flag for the Manila Subduction (0/1) nr x 1
    # FVS30: flag for the source of Vs30 (0 for Kuo17 (measured), 1 for KS17
    # (inferred), 2 for Receiver Function (inferred)) 1 x 1
    # deg: site-to-hypocenter azimuth (deg)
    # arfad: site-to-hypocenter azimuth of the rupture directivity direction (deg)
    #% rv: velocity ratio between the rupture velocity and the shear wave velocity
    # e: coefficient of percent unilateral rupture
    # phi: deviation angle between primary and secondary rupture direction (deg)
    # PI: Pulse Indicator (0/1)
    # Tp: Pulse Period (-999 for Taiwan Default)
    #
    # Output
    # Sa: spectral acceleration in unit g nr x nt
    # tau: between event std. in ln(g) unit nr x nt
    # phis2s: between site std. in ln(g) unit nr x nt
    # phiss: single station std. in ln(g) unit nr x nt

    period=np.array([0,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2,3,4,5,-1,-2])

    if type(M) == int or type(M) == float:
        len_M = 1
    else:
        len_M = len(M)
    if type(Rrup) == int or type(Rrup) == float:
        len_Rrup = 1
    else:
        len_Rrup = len(Rrup)
        
    NR = max([len_M , len_Rrup])
    
    
    if type(T) == int or type(T) == float:
        T = [T]
        NT = 1
    else:
        NT = len(T)
       
    
    Sa = np.zeros(shape=(NR,NT))
    tau =  np.zeros(shape=(NR,NT))
    phis2s =  np.zeros(shape=(NR,NT))
    phiss =  np.zeros(shape=(NR,NT))
        
    
    for I in range(NT):

        T_low = max(period[period <= T[I]])
        T_high = min(period[period >= T[I]])
        ip_low  = np.where( period == T_low)[0][0]
        ip_high = np.where( period == T_high)[0][0]
        if ip_low == ip_high :

            Sa[:, I], tau[:, I], phis2s[:, I], phiss[:, I] = Chao19H_sub_DIPI(ip_low, M, Rrup,\
                                                                              Ztor, Vs30, Z10,\
                                                                              FRO, FSS, FNO, \
                                                                              Finter, Fintra, \
                                                                              Fas, Fma, FVS30,\
                                                                              DI,deg,arfad,rv,\
                                                                              e, phi , PI , Tp)
        else:

            Sa_low, tau_low, phis2s_low, phiss_low = Chao19H_sub_DIPI(ip_low, M, Rrup,\
                                                                      Ztor, Vs30, Z10,\
                                                                      FRO, FSS, FNO, \
                                                                      Finter, Fintra, \
                                                                      Fas, Fma, FVS30,\
                                                                      DI,deg,arfad,rv,\
                                                                      e, phi , PI , Tp)

            Sa_high, tau_high, phis2s_high, phiss_high = Chao19H_sub_DIPI(ip_high, M, Rrup,\
                                                                      Ztor, Vs30, Z10,\
                                                                      FRO, FSS, FNO, \
                                                                      Finter, Fintra, \
                                                                      Fas, Fma, FVS30,\
                                                                      DI,deg,arfad,rv,\
                                                                      e, phi , PI , Tp)
            x = np.array([np.log(T_low), np.log(T_high)])
            Y_sa = np.array([np.log(Sa_low), np.log(Sa_high)])
            Y_tau = np.array([tau_low, tau_high])
            Y_phis2s = np.array([phis2s_low, phis2s_high])
            Y_phiss = np.array([phiss_low, phiss_high])
            Sa[:, I] = np.exp(interp1d(x,  Y_sa.T,axis=-1 ,kind='linear')(np.log(T[I])))
            tau[:, I] = interp1d(x, Y_tau.T,axis=-1, kind='linear')(np.log(T[I]))
            phis2s[:, I] = interp1d(x, Y_phis2s.T,axis=-1, kind='linear')(np.log(T[I]))
            phiss[:, I] = interp1d(x, Y_phiss.T,axis=-1, kind='linear')(np.log(T[I]))

            
    return Sa,tau,phis2s,phiss

def Chao19H_sub_DIPI(iT, M, Rrup, Ztor, Vs30, Z10, FRO, FSS, FNO, Finter, Fintra, Fas, Fma, FVS30,\
                     DI,deg,arfad,rv,e,phi,PI,Tp):

    period=np.array([0,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2,3,4,5,-1,-2])
    
    GMPE_coefficients = pd.read_csv(_COEF_CSV,\
                                    index_col=["T"])
        
    par = GMPE_coefficients.loc[:,:"c30"].to_numpy()
    par1 = GMPE_coefficients.loc[:,"taucr1":"phis2s"].to_numpy()
    
    
    # Constants
    C4inter = 0.3
    C4intra = 0.2
    Mc = 7.1
    Mref = 6.5
    n = 2
    Vs30ref = 760
    Rrupref = 0

    Z10ref = np.exp(-4.08 / 2 * np.log((Vs30 ** 2 + 355.4 ** 2) / (1750 ** 2 + 355.4 ** 2)))

    Hcrref = 0
    Hinterref = 0
    Hintraref = 35

    # Compute Fcr and Fsb
    Fcr = (FRO + FSS + FNO)
    Fsb = Finter + Fintra
    
    # If Z10 is not provided (-999 default, scalar or vector), set it to Z10ref
    Z10 = np.where(np.asarray(Z10) == -999, Z10ref, Z10)
    #h=Fcr*10+Finter*10*np.exp(C4inter*(M-Mc)*np.heaviside(M-Mc))+Fintra*10*exp(C4intra*(M-Mc).*heaviside(M-Mc));
    # Calculate h
    h = (Fcr * 10 + Finter * 10 * np.exp(C4inter * (M - Mc) * (M > Mc)) + 
         Fintra * 10 * np.exp(C4intra * (M - Mc) * (M > Mc))) 
  
    # Assign FVS30 indicators
    if FVS30 == 0:
        Fvs30kuo17 = 1
        Fvs30ks17 = 0
        Fvs30rf = 0
    elif FVS30 == 1:
        Fvs30kuo17 = 0
        Fvs30ks17 = 1
        Fvs30rf = 0
    elif FVS30 == 2:
        Fvs30kuo17 = 0
        Fvs30ks17 = 0
        Fvs30rf = 1
    else:
        Fvs30kuo17 = 1
        Fvs30ks17 = 0
        Fvs30rf = 0
        print('FVS30 Input Error ! Use Vs30 from Kuo17')
    
    # Initialize Xt matrix
    if type(Rrup) == int or type(Rrup) == float:
        len_Rrup = 1
    else:
        len_Rrup = len(Rrup)
        
     
    NR = len_Rrup
    Xt = np.zeros((NR, 30))
    # Assign values to Xt columns
    Xt[:, 0] = FRO
    Xt[:, 1] = FSS
    Xt[:, 2] = FNO 
    Xt[:, 3] = Finter
    Xt[:, 4] = Fintra
    Xt[:, 5] = Fas
    Xt[:, 6] = Fma    
    Xt[:, 7] = (Fcr * (M - Mref))
    Xt[:, 8] = (Fsb * (M - Mref)) 
    Xt[:, 9] = (Fcr * (M - Mref)**2 - Fcr * (M - 7.6)**2 * (M > 7.6))
    Xt[:, 10] = (Fcr* (5 - M) * (M < 5))
    Xt[:, 11] = (Fsb * (5 - M) * (M < 5))
    Xt[:, 12] = (Fsb * (6 - M) * (M < 6))
    Xt[:, 13] = (Fcr * (Ztor - Hcrref))
    Xt[:, 14] = (Finter * (Ztor - Hinterref))
    Xt[:, 15] =( Fintra * (Ztor - Hintraref)) 
    Xt[:, 16] = Fcr * np.log((Rrup**n + h**n)**(1/n) / (Rrupref ** n + h**n)**(1/n));  
    Xt[:, 17] = (Fsb * np.log((Rrup**n + h**n)**(1/n) / (Rrupref**n + h**n)**(1/n)))
    Xt[:, 18] =  (Fcr * (M - Mref) * np.log((Rrup**n + h**n)**(1/n) / (Rrupref**n + h**n)**(1/n)))
    Xt[:, 19] = (Fsb * (np.minimum(M, Mc) - Mref) * np.log((Rrup**n + h**n)**(1/n) / (Rrupref**n + h**n)**(1/n)))
    Xt[:, 20] = (Fcr  * (Rrup - Rrupref))
    Xt[:, 21] =  (Fsb * (Rrup - Rrupref))
    Xt[:, 22] = 0
    Xt[:, 23] = np.log(1180 / Vs30ref) 
    Xt[:, 24] = 0  # log(Z10 / Z10ref)
    Xt[:, 25] = Fvs30kuo17
    Xt[:, 26] = Fvs30ks17
    Xt[:, 27] = Fvs30rf
    
    
    Xt[:, 28] = (Finter * (M - Mc) * (M > Mc)) 
    Xt[:, 29] =  (Fintra * (M - Mc) * (M > Mc))
    
    # Calculate Sa1180

    Sa1180 = np.exp(np.dot(Xt, par[iT]))

    # Change c value for PGV prediction (iT is 0-based here; PGV row index is 20, PGD is 21)
    if iT > 19:
        c = 2400
    else:
        c = 2.4
   
    Xt[:, 22] = (-1.5 * np.log((Vs30 / Vs30ref)) - np.log(Sa1180 + c) + \
                 np.log(Sa1180 + c * (Vs30 / Vs30ref) ** 1.5)) * (Vs30ref > Vs30)      
    Xt[:, 23] = np.log(Vs30 / Vs30ref)
    Xt[:, 24] = np.log(Z10 / Z10ref)
    
    # Directivity calculate

    if DI == 1:
        degt = np.arange(0, 360.1, 0.1)
        fd0 = np.mean(np.log(0.5 * ((1 + e)**2 / (1 - rv * np.cos((degt - arfad) / 180 * np.pi))**2 + 
                      (1 - e)**2 / (1 + rv * np.cos((degt - arfad - phi) / 180 * np.pi))**2)**0.5))
        fd = np.log(0.5 * ((1 + e)**2 / (1 - rv * np.cos((deg - arfad) / 180 * np.pi))**2 + 
                    (1 - e)**2 / (1 + rv * np.cos((deg - arfad - phi) / 180 * np.pi))**2)**0.5) - fd0
    else:
        fd = 0
    
    if DI == 1:
       if period[iT] == 0:
            a0 = 0.0851
       elif period[iT] == -1:
            a0 = 0.2334
       elif period[iT] == -2:
            a0 = 0.2434
       else:
            a0 = 0.0054
            a1 = 0.4478
            a2 = -2.3584
            a3 = 0.0816
    else:
        if period[iT] == 0:
            a0 = 0.2734
        elif period[iT] == -1:
            a0 = 0.5897
        elif period[iT] == -2:
            a0 = 0.5430
        else:
            a0 = 0.1059
            a1 = 0.7612
            a2 = -0.6443
            a3 = 0.1099
    if Tp == -999:
        Tp = np.exp(-6.7183 + 1.1382 * np.maximum(5.5, M))


    if period[iT] > 0:
        fp = a0 + a1 * np.exp(a2 * (np.log(period[iT] / Tp) - a3)**2)
    else:
        fp = a0

    median = np.exp(np.dot(Xt, par[iT]) + fd + fp * PI) 
    
    # Calculate tau, phiss, and phis2s

    fM = 0.5 * (np.minimum(6.5, np.maximum(4.5, M)) - 4.5)
    tau = Fcr* (par1[iT, 0] + (par1[iT, 1] - par1[iT, 0]) * fM) + \
          Fsb * (par1[iT, 2] + (par1[iT, 3] - par1[iT, 2]) * fM)
    phiss = Fcr* (par1[iT, 4] + (par1[iT, 5] - par1[iT, 4]) * fM) + \
            Fsb * (par1[iT, 6] + (par1[iT, 7] - par1[iT, 6]) * fM) 
    phis2s = par1[iT,8] * np.ones(shape = (NR,)) 

    return median,tau,phis2s,phiss


# This part is used to test the above code
"""
#Chao19H(T,M,Rrup,Ztor,Vs30,Z10,FRO,FSS,FNO,Finter,Fintra,Fas,Fma,FVS30)
# Define points
points = 100
# Create rup_distance array
Rrup = np.arange(0,200,1 )
HypDepth = 10
Ztor = HypDepth
Mag= 6

Vs30 = 760
FRO = 0
FSS = 1
FNO = 0 
Finter = 0
Fintra = 0 
Fas = 0 
Fma = 0
FVS30=0
# Call Chao19H function
iT = 0

#rupture directivity parameters
DI=1
rv=0.7
e=1
phi=0
PI=0
Tp=-999
rv=0.7
e=1
phi=0
PI=0
AzimuthD=340
MAzimuth = np.ones(shape=(len(Rrup),)) * 20

GMPE_M_4, tau_4, phis2s_4, phiss_4 = Chao19H_DIPI(iT, Mag , Rrup , \
                                             Ztor, Vs30 ,-999, \
                                             FRO,FSS,FNO,Finter,\
                                             Fintra,Fas,Fma,FVS30,\
                                             DI,MAzimuth,AzimuthD,rv,\
                                             e, phi , PI , Tp)

GMPE_M_5, tau_5, phis2s_5, phiss_5 = Chao19H(iT, Mag , Rrup , \
                                             Ztor, Vs30 ,-999, \
                                             FRO,FSS,FNO,Finter,\
                                             Fintra,Fas,Fma,FVS30)


fig , ax = plt.subplots(1,1,figsize=(15,9))
ax.loglog(Rrup,GMPE_M_4,'b-')
ax.loglog(Rrup,GMPE_M_5,'r-')
fig.savefig("test.png",dpi = 300)

"""