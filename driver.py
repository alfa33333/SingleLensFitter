import sys
import numpy as np
import SingleLensFitter as slf
import matplotlib.pyplot as plt
import os


# Basic necesary information to run the fitter.
data_source = "/home/alfa33333/Program/SingleLensFitter-master/bul-01/1998/bul-01/phot.dat"
#("/home/alfa33333/Program/SingleLensFitter-master/blg-001/2002/blg-001/phot.dat")
base = 15.702 #Base line for the flux.
folder = '' #path to the folder to save.
file_prefix = 'test' #prefix of the saved files.

def date(d1):#Obtains the date from the data
    return  d1[:,0]-2450000

def flux(d1):#Obtains the proper flux from the data
    return  pow(10.0*np.ones_like(d1[:,1]), (0.4*(base-d1[:,1])))

def error(d1):#Adjust the error  with the flux
    return  -0.4*np.log(10.)*flux(d1)*d1[:,2]

def main():

    #loading data and preparing it for the fitter
    d1 = np.loadtxt(data_source)
    data = {}
    eigencurve = {}
    indexmax = np.argmax(flux(d1))
    data['site1'] = (date(d1),flux(d1),error(d1))

    #Instance to the fitter
    fitter = slf.SingleLensFitter(data,[0.16,18,50.])
    fitter.plotprefix = folder+file_prefix

    #Running sampler
    fitter.Nested()

if __name__=='__main__':
    main()
