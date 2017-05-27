#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 21:30:15 2017

@author: jeremy
"""
import numpy as np
from scipy.optimize import curve_fit


def getGaussianFit(xVals, yVals, yErrors=None):
    "Fits a gaussian function to the given function, returns array with y values of fit"
    
    if yErrors == None:
        yErrors = np.full(yVals.size, 1)
    
    # gaussian function
    gauss = lambda x, A, mu, sigma: A * np.exp( -(x - mu)**2 / (2.0 * sigma**2) )

    # initial guesses for A, mu, sigma
    p0 = [1.0, 0.0, 1.0]

    params, cov_matrix = curve_fit(gauss, xVals, yVals, p0, yErrors)
    fitcurve = gauss(xVals, params[0], params[1], params[2]) # get the y values: arguments are A, mu, sigma

    return fitcurve


def getSAAAvgSpec(data, ID_x, ID_y):
    """
    Averages associated spectral values in an SAA, returns the averaged spectrum array
    """
    avgedspectra = []
    
    for spectralslice in data:
        SAAspecslice = spectralslice[np.ix_(ID_y, ID_x)] # get spectral values in this slice of the SAA
        avgedspectra.append(np.mean(SAAspecslice)) # get the average of all spec values in the SAA
    
    return avgedspectra


def getSTDDEVError(xVals, yVals, windowStart, windowStop):
    """
    Gets the standard deviation of the given y values between two points in the x values
    """
    
    windowYVals = yVals[(xVals >= windowStart) & (xVals <= windowStop)] # gets y values where start <= x <= stop
    stddev = np.std(windowYVals)
    return stddev
    