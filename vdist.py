# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:35:35 2024

@author: user
"""
   
import numpy as np

def vdist(lat1, lon1, lat2, lon2):
    
    if type(lat1) == float and type(lon1) == float :
        lat1 = np.array([lat1])
        lon1 = np.array([lon1])
    if type(lat2) == float and type(lon2) == float :
        lat2 = np.array([lat2])
        lon2 = np.array([lon2])
    # Reshape inputs
    keepsize = lat1.shape
    lat1 = lat1.ravel()
    lon1 = lon1.ravel()
    lat2 = lat2.ravel()
    lon2 = lon2.ravel()
    
    
    # Simple distance in degrees
    distance = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
    
    
    # Input check
    if np.any(np.abs(lat1) > 90) or np.any(np.abs(lat2) > 90):
        raise ValueError('Input latitudes must be between -90 and 90 degrees, inclusive.')

    # Supply WGS84 earth ellipsoid axis lengths in meters
    a = 6378137  # definitionally
    b = 6356752.31424518  # computed from WGS84 earth flattening coefficient

    # Preserve true input latitudes
    lat1tr = lat1.copy()
    lat2tr = lat2.copy()

    # Convert inputs in degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Correct for errors at exact poles by adjusting 0.6 millimeters
    kidx = np.abs(np.pi/2 - np.abs(lat1)) < 1e-10
    
    if np.any(kidx):
        lat1[kidx] = np.sign(lat1[kidx]) * (np.pi/2 - (1e-10))

    kidx = np.abs(np.pi/2 - np.abs(lat2)) < 1e-10
    if np.any(kidx):
        lat2[kidx] = np.sign(lat2[kidx]) * (np.pi/2 - (1e-10))

    f = (a - b) / a
    U1 = np.arctan((1 - f) * np.tan(lat1))   
    U2 = np.arctan((1 - f) * np.tan(lat2))
   
    lon1 = np.mod(lon1, 2*np.pi)
    lon2 = np.mod(lon2, 2*np.pi)
    L = np.abs(lon2 - lon1)
    
    kidx = L > np.pi
    if np.any(kidx):
        L[kidx] = 2*np.pi - L[kidx]

    lambda_val = L.copy()
    lambdaold = np.zeros_like(lat1)
    itercount = 0
    notdone = np.ones_like(lat1, dtype=bool)
    alpha = np.zeros_like(lat1)
    sigma = np.zeros_like(lat1)
    cos2sigmam = np.zeros_like(lat1)
    C = np.zeros_like(lat1)
    warninggiven = False
    sinsigma = np.ones_like(lat1, dtype=float)
    cossigma = np.ones_like(lat1, dtype=float)
    while np.any(notdone):
        itercount += 1
        if itercount > 50:
            if not warninggiven:
                warninggiven = True
                print('Essentially antipodal points encountered. Precision may be reduced slightly.')
            lambda_val[notdone] = np.pi
            break

        lambdaold[notdone] = lambda_val[notdone]
        
        sinsigma[notdone] = np.sqrt((np.cos(U2[notdone]) * np.sin(lambda_val[notdone])) ** 2 +
                           (np.cos(U1[notdone]) * np.sin(U2[notdone]) - np.sin(U1[notdone]) *
                            np.cos(U2[notdone]) * np.cos(lambda_val[notdone])) ** 2)

        cossigma[notdone] = np.sin(U1[notdone]) * np.sin(U2[notdone]) + \
                   np.cos(U1[notdone]) * np.cos(U2[notdone]) * np.cos(lambda_val[notdone])

        sinsigma[notdone] = np.real(sinsigma[notdone])
        cossigma[notdone] = np.real(cossigma[notdone])

        sigma[notdone] = np.arctan2(sinsigma[notdone], cossigma[notdone])

        alpha[notdone] = np.arcsin(np.cos(U1[notdone]) * np.cos(U2[notdone]) *
                                   np.sin(lambda_val[notdone]) / np.sin(sigma[notdone]))
        
        
        cos2sigmam[notdone] = np.cos(sigma[notdone]) - 2 * np.sin(U1[notdone]) * np.sin(U2[notdone]) /\
            np.cos(alpha[notdone]) ** 2

        C[notdone] = f / 16 * np.cos(alpha[notdone]) ** 2 * \
                     (4 + f * (4 - 3 * np.cos(alpha[notdone]) ** 2))
                     
        lambda_val[notdone] = L[notdone] + (1 - C[notdone]) * f * np.sin(alpha[notdone]) * \
                              (sigma[notdone] + C[notdone] * np.sin(sigma[notdone]) *
                               (cos2sigmam[notdone] + C[notdone] * np.cos(sigma[notdone]) *
                                (-1 + 2 * cos2sigmam[notdone] ** 2)))
                                        
        if np.any(lambda_val[notdone] > np.pi):
            if not warninggiven:
                warninggiven = True
                print('Essentially antipodal points encountered. Precision may be reduced slightly.')
            lambdaold[lambda_val > np.pi] = np.pi
            lambda_val[lambda_val > np.pi] = np.pi
            
        # Check convergence
        notdone = np.abs(lambda_val - lambdaold) > 1e-12

    u2 = np.cos(alpha) ** 2 * (a ** 2 - b ** 2) / b ** 2
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    deltasigma = B * np.sin(sigma) * (cos2sigmam + B / 4 * (np.cos(sigma) *
                    (-1 + 2 * cos2sigmam ** 2) - B / 6 * cos2sigmam *
                    (-3 + 4 * np.sin(sigma) ** 2) * (-3 + 4 * cos2sigmam ** 2)))

    result = b * A * (sigma - deltasigma).reshape(keepsize)
    
    lambda_val = np.abs(lambda_val);
    lambda_val = np.where(np.sign(np.sin(lon2 - lon1)) * np.sign(np.sin(lambda_val)) < 0, -lambda_val, lambda_val)
    a12 = np.arctan2(np.cos(U2) * np.sin(lambda_val), np.cos(U1) * np.sin(U2) - np.sin(U1) * np.cos(U2) * np.cos(lambda_val))
    a12 = np.where(a12 < 0, a12 + 2 * np.pi, a12)
    a12 = np.degrees(a12)
        
    a12[lat1 <= -90] = 0
    a12[lat1 >= 90] = 180
    
    a21 = np.arctan2(np.cos(U1) * np.sin(lambda_val), np.sin(U1) * np.cos(U2) - np.cos(U1) * np.sin(U2) * np.cos(lambda_val))
    a21 = np.where(a21 < 0, a21 + 2 * np.pi, a21)
    a21 = np.degrees(a21)
        
    a21[lat2 >= 90] = 180
    a21[lat2 <= -90] = 0
    
    
    return  result,a12,a21



