import os
#import mkl
#mkl.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import Read_Local_re as read
import Read_kmt as readkmt
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import SingleLensFitter as slf
from tlib import rmdpoint
from time import time
#


if __name__ == '__main__':
    #
    trange = (8220,8240)
    data = read.read_data('./Data/',trange)
    #data = readkmt.read_data('./Data/',trange)
    #rmdpoint(data,'KMTS37-I',8198.50797)
    fitter = slf.SingleLensFitter(data,[1.07056800e-01, 8.22955366e+03, 5.00686788e+00]) 


    folder = './output/'
    file_prefix = 'binsource-dynesty'
    fitter.add_binary_source(system='both')

    #Nested
    fitter.nlive = 1000
    fitter.tol = 0.1
    fitter.dynestyparallel = True
    #emcee
    fitter.nwalkers = 100
    fitter.nsteps = 200
    fitter.nsteps_production = 500 
    fitter.t0_limits = (8220, 8240)
    #plotting
    fitter.plotrange = (8228,8231)
    fitter.plotprefix = folder+file_prefix
    fitter.p = [1.07056800e-01, 8.22955366e+03, 5.00686788e+00, 6.98172906e-02,8.22947536e+03, 1.15723553e-01]
    t0 = time()
    #fitter.fit()
    #fitter.nestling() #beta nestle
    fitter.dnesty()
    t1 = time()
    print(t1-t0)