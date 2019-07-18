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
from tlib import *
from time import time
#


if __name__ == '__main__':
    #
    trange = (8200,8260)
    renormdic = readdict("./bindata/renormalise_values.dat")
    for i in renormdic.keys():
        print('source {} : [ {} {} ]'.format(i,renormdic[i][0],renormdic[i][1]))
    data = read.read_data('./Data/',trange,renorm=renormdic,minmag=False)
    #data = readkmt.read_data('./Data/',trange)
    #rmdpoint(data,'KMTS37-I',8198.50797)
    removedict = removesignal('./Data/filter', trange)
    print('Filtering data ...')
    for y in data.keys():
	    rmdpoint(data,y, removedict[y])
    
    fitter = slf.SingleLensFitter(data,[1.07056800e-01, 8.22955366e+03, 5.00686788e+00]) 
    


    folder = './output/single0677-emcee-renormv2/'
    file_prefix = 'single-source-dynesty'
    #fitter.add_binary_source(system='both')

    #Nested
    fitter.nlive = 500
    fitter.tol = 0.1
    fitter.dynestyparallel = True
    fitter.sampletype = 'unif'
    #emcee
    fitter.nwalkers = 100
    fitter.nsteps = 200
    fitter.nsteps_production = 500 
    #limits 
    fitter.t0t1extlim = True
    fitter.t0_limits = (8220, 8240)
    fitter.t02_limits = (8220,8240)
    #plotting
    fitter.plotrange = (8228,8231)
    fitter.plotprefix = folder+file_prefix
    #fitter.p = [1.07056800e-01, 8.22955366e+03, 5.00686788e+00, 6.98172906e-02,8.22947536e+03, 1.15723553e-01]
    t0 = time()
    #fitter.fit()
    #fitter.nestling() #beta nestle
    fitter.dnesty()
    t1 = time()
    print('Sampler time: {:f}'.format(t1-t0))
    #It run emcee for a short convergence
    fitter.plotprefix = folder+'singlesource-emcee'
    t0 = time()
    fitter.fitparallel()
    t1 = time()
    print('Sampler time: {:f}'.format(t1-t0))
    samples = np.load(fitter.plotprefix+'-samples-production.npy')
    lnp = np.load(fitter.plotprefix+'-lnp-production.npy')
    bestp = samples[ np.argmax(lnp) ]
    chi2 = fitter.chi2_calc(p=bestp)
    with open(folder+file_prefix+'minchi2.dat','w') as fid:
        fid.write('The minchi2 is {:f} \n'.format(chi2))
    fid.close()
    print('The minchi2 is {:f} \n'.format(chi2))
