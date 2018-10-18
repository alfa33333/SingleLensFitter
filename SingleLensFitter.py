import numpy as np
from scipy import linalg
from scipy import optimize

import emcee
import george
from george import kernels

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from pylab import subplots_adjust

import corner
from astropy.stats import mad_std

from scipy.interpolate import interp1d
from pymultinest import solve 
import dynesty
from dynesty import NestedSampler
from dynesty.utils import resample_equal

import nestle

class SingleLensFitter():

	"""Class definition for single microlens event fitter."""

	def __init__(self,data,initial_parameters,eigen_lightcurves=None,reference_source=None):

		"""Initialise the SingleLensFitter.

		inputs:

			data:           A dictionary with each key being a data set name string and each 
						value being a tuple of (date, flux, flux_err) each being numpy
						arrays.

			initial_parameters: A numpy array of starting guess values for u_0, t_0, t_E.

			eigen_lightcurves:      If defined, this should be a dictionary with the same keys as data,
						and each value being an n x m numpy array of n lightcurves each 
						with m data values corresponding to the dates of the input data.
						These lightcurves (detrend vectors) are fitted linearly to the data
						at the same time as the magnification and flux parameters. An
						eigen_lightucrves dictionary entry can be defined for any subset 
						of data sources, and different data sources can have different
						numbers of eigenlightcurves.

		"""

		self.data = data
		self.eigen_lightcurves = eigen_lightcurves
		self.initial_parameters = initial_parameters
		self.p = initial_parameters

		if reference_source is None:

			self.reference_source = self.data.keys()[0]

			print 'Using',self.reference_source,'as reference.'

		else:

			if reference_source in self.data:

				self.reference_source = reference_source

			else:

				self.reference_source = self.data.keys()[0]

				print 'Warning:',reference_source,'is not a valid data source.'
				print 'Using',self.reference_source,'as reference.'


		self.parameter_labels = [r"$u_0$",r"$t_0$",r"$t_E$"]
		self.ndim = 3

		self.use_finite_source = False

		self.marginalise_linear_parameters = True
		self.fit_blended = True

		self.u0_limits = (0.0,1.3)
		self.tE_limits = (0.5,200)
		self.t0_limits = None
		self.lrho_limits = (-6,0)

		self.use_gaussian_process_model = False
		self.GP_default_params = (1.0,-2.0)
		self.ln_a_limits = (-5,15)
		self.ln_tau_limits = (-5.5, 7.0)
		self.ln_a = {}
		self.ln_tau = {}

		self.use_mixture_model = False
		self.mixture_default_params = (0.0001,1.0e8,0.0)
		self.P_b_limits = (0.0,0.05)
		self.V_b_limits = (0.0,1.0e12)
		self.Y_b_limits = (-1.e5,1.e5)
		self.P_b = {}
		self.V_b = {}
		self.Y_b = {}

		self.use_finite_source = False

		self.nwalkers = 50
		self.nsteps = 50
		self.nsteps_production = 500
		self.thresh_std = 0.1
		self.thresh_mean = 0.05
		self.max_burnin_iterations = 200
		self.emcee_optimize_first = True
		self.emcee_lnp_convergence_threshold = 0.5
		self.emcee_mean_convergence_threshold = 0.1
		self.emcee_std_convergence_threshold = 0.1

		self.plotprefix = 'single_lens_fit'
		self.samples = None
		self.n_plot_samples = 30
		self.make_plots = True

		self.plot_colours = ['#FF0000', '#000000', '#008000', '#800080', '#FFA500', '#A52A2A', '#ff9999', '#999999', \
					'#4c804c', '#804d80', '#ffdb98', '#a56262', '#CCFF66', '#CC9900', '#9966FF', '#FF3366']

		return


	def add_finite_source(self,lrho=None):

		from mpmath import ellipe
		
		self.use_finite_source = True
		self.finite_source_index = self.ndim

		if lrho is not None:
			self.lrho = lrho
		else:
			self.lrho = -3.0

		self.p = np.hstack((self.p,self.lrho))
		self.parameter_labels.append(r"$log_{10} rho$")
		self.ndim += 1

		lz = np.arange(-5,10,0.01)
		fsz = 10.0**lz
		ell = np.zeros_like(fsz)
		rsz = np.arcsin(1.0/fsz)
		p2 = np.pi/2.0
		for zi in range(len(lz)):
			if fsz[zi] < 1.0:
				ell[zi] = ellipe(p2,fsz[zi])
			else:
				ell[zi] = ellipe(rsz[zi],fsz[zi])
		self._ellipe_interpolator = interp1d(fsz,ell)

	def add_mixture_model(self):

		self.use_mixture_model = True
		self.mixture_index = self.ndim

		for site in self.data.keys():

			self.p = np.hstack((self.p, self.mixture_default_params))
			self.ndim += 3
			self.parameter_labels.append(site+'_P_b')
			self.parameter_labels.append(site+'_V_b')
			self.parameter_labels.append(site+'_Y_b')



	def add_gaussian_process_model(self):

		self.use_gaussian_process_model = True
		self.gaussian_process_index = self.ndim

		for site in self.data.keys():

			self.p = np.hstack((self.p, self.GP_default_params))
			self.ndim += 2
			self.parameter_labels.append(site+'_ln_a')
			self.parameter_labels.append(site+'_ln_tau')



	def lnprior_GP(self,GP_params):
		params = ['ln_a','ln_tau']
		for i, p in enumerate(params):
			prange = eval('self.'+p+'_limits')
			if prange:
				if GP_params[i] < prange[0] or GP_params[i] > prange[1]:
					return -np.inf
		return 0.0

	

	def lnprior_mixture(self,mixture_params):
		params = ['P_b','V_b','Y_b']
		for i, p in enumerate(params):
			prange = eval('self.'+p+'_limits')
			if prange:
				if mixture_params[i] < prange[0] or mixture_params[i] > prange[1]:
					return -np.inf
		return 0.0

	

	def lnprior_ulens(self):
		params = ['u0','t0','tE']
		for p in params:
			prange = eval('self.'+p+'_limits')
			if prange:
				param = eval('self.'+p)
				if param < prange[0] or param > prange[1]:
					return -np.inf
		return 0.0


	def lnprior_lrho(self,lrho):
		if lrho < self.lrho_limits[0] or lrho > self.lrho_limits[1]:
			return -np.inf
		return 0.0


	def lnprob(self,p):

		self.p = p
		self.u0 = p[0]
		self.t0 = p[1]
		self.tE = p[2]

		lp = self.lnprior_ulens()

		if self.use_finite_source:
			lp += self.lnprior_lrho(p[self.finite_source_index])

		if self.use_mixture_model:

			pi = self.mixture_index
			for data_set_name in self.data.keys():

				self.P_b[data_set_name] = p[pi]
				self.V_b[data_set_name] = p[pi+1]
				self.Y_b[data_set_name] = p[pi+2]
				lp += self.lnprior_mixture(p[pi:pi+3])
				pi += 3

		if self.use_gaussian_process_model:

			pi = self.gaussian_process_index
			for data_set_name in self.data.keys():

				self.ln_a[data_set_name] = p[pi]
				self.ln_tau[data_set_name] = p[pi+1]
				lp += self.lnprior_GP(p[pi:pi+2])
				pi += 2

		if np.isfinite(lp):
			lp += self.lnlikelihood()
		else:
			return -np.inf

		return lp

	def neglnprob(self,p):
		return -self.lnprob(p)


	def lnlikelihood(self):

		lnprob = 0.0

		for data_set_name in self.data.keys():

			t, y, yerr = self.data[data_set_name]
			mag = self.magnification(t)

			if self.use_gaussian_process_model:
				a = np.exp(self.ln_a[data_set_name])
				tau = np.exp(self.ln_tau[data_set_name])
				gp = george.GP(a * kernels.ExpKernel(tau))
				gp.compute(t, yerr)
				self.cov = gp.get_matrix(t)
				result, lp = self.linear_fit(data_set_name,mag)
				model = self.compute_lightcurve(data_set_name,t)
				lnprob = gp.lnlikelihood(y-model)

			else:
				result, lp = self.linear_fit(data_set_name,mag)
				lnprob += lp

		return lnprob


	def linear_fit(self,data_key,mag):

		_, y, yerr = self.data[data_key]

		if self.use_gaussian_process_model:
			C = self.cov
		else:
			C = np.diag(yerr**2)

		C_inv = linalg.inv(C)

		if self.fit_blended:
			A = np.vstack((np.ones_like(mag),mag))
			n_params = 2
		else:
			A = (mag-1.0).reshape(1,len(mag))
			n_params = 1

		if self.eigen_lightcurves is not None:
			if data_key in self.eigen_lightcurves:
				eigs = self.eigen_lightcurves[data_key]
				for i in range(eigs.shape[0]):
					A = np.vstack((A,eigs[i,:]))
					n_params += 1

		S = np.dot(A,np.dot(C_inv,A.T)).reshape(n_params,n_params)
		b = np.dot(A, np.dot(C_inv,y).T)

		if self.marginalise_linear_parameters:

			try:
				M = linalg.inv(S)
			except linalg.LinAlgError:
				return (0,0), -np.inf

			g = np.dot(M,b)
			D = y - np.dot(A.T,g)
			chi2 = np.dot(D.T,np.dot(C_inv,D))

			if self.use_mixture_model:

				detM = linalg.det(M)
				lnprob = np.log( 2*np.pi*np.sum( (1.0 - self.P_b[data_key]) * \
						np.exp(-D**2/(2.0*yerr**2)) / \
						np.sqrt(detM) + \
						self.P_b[data_key]*np.exp(-(y-self.Y_b[data_key])**2 / \
						2*(self.V_b[data_key]+yerr**2)) / \
						np.sqrt(2*np.pi*(self.V_b[data_key]+yerr**2)) ))
			else:

				lnprob = np.log(2*np.pi) - 0.5*chi2 - 0.5*np.log(linalg.det(M))

			return g, lnprob

		else:

			try:
				a = linalg.solve(S,b)
			except linalg.LinAlgError:
				return (0,0), -np.inf
			D = y - np.dot(A.T,a)
			chi2 = np.dot(D.T,np.dot(C_inv,D))

			if self.use_mixture_model:

				lnprob = np.sum( np.log( (1.0 - self.P_b[data_key])*np.exp(-D**2/(2.0*yerr**2)) / \
						np.sqrt(2*np.pi*yerr**2) + \
						self.P_b[data_key]*np.exp(-(y-self.Y_b[data_key])**2 / \
						2*(self.V_b[data_key]+yerr**2)) / \
						np.sqrt(2*np.pi*(self.V_b[data_key]+yerr**2)) ))

			else:

				lnprob = -np.log(np.sum(np.sqrt(2*np.pi*yerr**2))) - 0.5*chi2

			return a, lnprob


	def magnification(self,t,p=None):

		if p is None:
			p = self.p

		u0, t0, tE = p[:3]

		if self.use_finite_source:
			lrho = p[self.finite_source_index]

		tau = (t-t0)/tE

		u = np.sqrt(u0**2+tau**2)

		A = (u**2 + 2.0)/(u*np.sqrt(u**2+4.0))

		if self.use_finite_source:

			rho = 10.0**lrho


			#
			#  This is from Lee et al. (2009) ApJ, 695, 200 - there seems to be a bug somewhere
			#

			# n = 10

			# for q in range(len(t)):

			# 	if u[q] <= rho:

			# 		k = np.arange(1,n)
			# 		theta = np.pi*k/n
			# 		u2 = u[q]*np.cos(theta) + np.sqrt(rho**2 - u[q]**2 * np.sin(theta)**2)
			# 		f1 = u2 * np.sqrt(u2**2 + 4.0)

			# 		k = np.arange(1,n+1)
			# 		theta = np.pi*(2.0*k - 1.0)/(2.0*n)
			# 		u2 = u[q]*np.cos(theta) + np.sqrt(rho**2 - u[q]**2 * np.sin(theta)**2)
			# 		f2 = u2 * np.sqrt(u2**2 + 4.0)


			# 		A[q] = (1.0/(2.0*n*rho**2)) * ( ((u[q]+rho)/3.0) * np.sqrt((u[q]+rho)**2+4.0) - \
			# 										((u[q]-rho)/3.0) * np.sqrt((u[q]-rho)**2+4.0) + \
			# 										(2.0/3.0) * np.sum(f1) + (4.0/3.0) * np.sum(f2) )

			# 	else:

			# 		k = np.arange(1,n/2)
			# 		theta = 2.0*k*np.arcsin(rho/u[q])/n
			# 		if u[q] <= np.arcsin(rho/u[q]):
			# 			u1 = u[q] * np.cos(theta) - np.sqrt(rho**2 - u[q]**2 * np.sin(theta)**2)
			# 			u2 = u[q] * np.cos(theta) + np.sqrt(rho**2 - u[q]**2 * np.sin(theta)**2)
			# 			f1 = u2 * np.sqrt(u2**2 + 4.0) - u1 * np.sqrt(u1**2 + 4.0)
			# 		else:
			# 			f1 = 0.0

			# 		k = np.arange(1,n/2+1)
			# 		theta = (2.0*k-1.0)*np.arcsin(rho/u[q])/n
			# 		if u[q] <= np.arcsin(rho/u[q]):
			# 			u1 = u[q] * np.cos(theta) - np.sqrt(rho**2 - u[q]**2 * np.sin(theta)**2)
			# 			u2 = u[q] * np.cos(theta) + np.sqrt(rho**2 - u[q]**2 * np.sin(theta)**2)
			# 			f2 = u2 * np.sqrt(u2**2 + 4.0) - u1 * np.sqrt(u1**2 + 4.0)
			# 		else:
			# 			f2 = 0.0

			# 		A[q] = (np.arcsin(rho/u[q])/(np.pi*n*rho**2)) * \
			# 						(   ((u[q]+rho)/3.0) * np.sqrt((u[q]+rho)**2+4.0) - \
			# 							((u[q]-rho)/3.0) * np.sqrt((u[q]-rho)**2+4.0) + \
			# 							(2.0/3.0) * np.sum(f1) + (4.0/3.0) * np.sum(f2) )



			#
			#  This is from Gould (1994) ApJ, 421, L71
			#

			z = np.abs(u)/rho

			fs_points = np.where(z < 10.0)[0]

			print 'fs_points', fs_points
			print 'z[fs_points]', z[fs_points]

			B0 = 4 * z[fs_points] * self._ellipe_interpolator(z[fs_points]) / np.pi
			A[fs_points] *= B0

		return A



	def emcee_has_converged(self,sampler,n_steps=100):

		# Emcee convergence testing is not easy. The method we will adopt
		# is to test whether the parameter means and standard deviations, and ln_p, have 
		# stabilised, comparing the last n steps, with the previous n steps.

		#std_threshold = 0.01
		#mean_threshold = 0.01

		n_test = sampler.chain.shape[0]*n_steps

		lnp = sampler.lnprobability.T.ravel()
		if len(lnp) < 2*n_test:
			return False

		converged = True

		steps = sampler.chain.shape[1]

		with open(self.plotprefix+'_lnp','a') as fid:

			fid.write("After %d steps, parameter means, standard deviations, convergence metrics and ln_P:\n"%steps)

			for k in range(sampler.chain.shape[2]):

				samples = sampler.chain[:,:,k].T.ravel()
				mean2 = np.mean(samples[-2*n_test:-n_test])
				mean1 = np.mean(samples[-n_test:])
				std2 = np.std(samples[-2*n_test:-n_test:])
				std1 = np.std(samples[-n_test:])

				delta_param = np.abs(mean1 - mean2)/std1
				delta_std = np.abs(std2-std1)/std2

				fid.write("%g %g %g %g\n"%(mean1,std1,delta_param,delta_std))

				if  delta_param > self.emcee_mean_convergence_threshold:
					converged = False

				if  delta_std > self.emcee_std_convergence_threshold:
					converged = False

			lnp_delta = np.mean(lnp[-n_test:]) - np.mean(lnp[-2*n_test:-n_test])

			if lnp_delta > self.emcee_lnp_convergence_threshold:
				converged = False

			fid.write("delta lnp: %10.4f %d\n"%(lnp_delta,converged))

		return converged


	def lnprior_nest(self,ucube):

		cube = np.copy(ucube)
		#prior for hypercube for the main parameters, flat priors.
		cube[0]= cube[0]*(self.u0_limits[1] - self.u0_limits[0]) + self.u0_limits[0]
		cube[1]= cube[1]*(self.data[self.data.keys()[0]][0][-1]-self.data[self.data.keys()[0]][0][0])+ self.data[self.data.keys()[0]][0][0]
		cube[2]= cube[2]*(self.tE_limits[1] - self.tE_limits[0]) + self.tE_limits[0]

		return cube

	def lnlike_nest(self,cube):#Likelihood function to evaluate for nested. 
		lnprob = 0.0

		for data_set_name in self.data.keys():

			t, y, yerr = self.data[data_set_name]
			mag = self.magnification(t,p = cube)

			if self.use_gaussian_process_model:
				a = np.exp(self.ln_a[data_set_name])
				tau = np.exp(self.ln_tau[data_set_name])
				gp = george.GP(a * kernels.ExpKernel(tau))
				gp.compute(t, yerr)
				self.cov = gp.get_matrix(t)
				result, lp = self.linear_fit(data_set_name,mag)
				model = self.compute_lightcurve(data_set_name,t)
				lnprob = gp.lnlikelihood(y-model)

			else:
				result, lp = self.linear_fit(data_set_name,mag)
				lnprob += lp

		return lnprob 

	def Nested(self):# Nested sampling main calling method.
		if self.p is None:
			raise Exception('Error in SingleLensFitter.fit(): No initial_parameters found.')
			return None
		#prints the boundary to be used for the prior
		t0lim,_,_ = self.data[self.data.keys()[0]]
		print('Prior boundaries:')
		print('u_0 = [%f , %f]'% self.u0_limits)
		print('t_0 = [%f , %f]'% (t0lim[0],t0lim[-1]))
		print('t_E = [%f , %f]'% self.tE_limits)

		#basic parameters to run the sampler
		#This parameters will be moved to the class.
		nlive = 600 # livepoints.
		ndim = self.ndim #number of dimensions
		tol = 0.01 #stopping criterion
		print('')
		#run the algorithm
		result =solve(LogLikelihood=self.lnlike_nest,Prior=self.lnprior_nest,
					n_dims=ndim,outputfiles_basename=self.plotprefix,
					n_live_points=nlive,
					evidence_tolerance=tol,
					verbose=True)
		u0line = result['samples'].T
		print('')
		print('evidence:%(logZ).f +- %(logZerr).1f' % result )
		print('')
		print('parameter values:')
		for name, col in zip(self.parameter_labels,u0line):
			print('%15s : %.3f +- %.3f' %(name,col.mean(),col.std()))

		return

	def dnesty(self):#Nested sampling using Dynesty
		if self.p is None:
			raise Exception('Error in SingleLensFitter.fit(): No initial_parameters found.')
			return None
		#prints the boundary to be used for the prior
		t0lim,_,_ = self.data[self.data.keys()[0]]
		print('Prior boundaries:')
		print('u_0 = [%f , %f]'% self.u0_limits)
		print('t_0 = [%f , %f]'% (t0lim[0],t0lim[-1]))
		print('t_E = [%f , %f]'% self.tE_limits)

		#basic parameters to run the sampler
		print('')
		print('dynesty version: {}'.format(dynesty.__version__))
		#This parameters will be moved to the class.
		nlive = 400 # livepoints.
		bound = 'multi'
		ndim = self.ndim #number of dimensions
		sample = 'rwalk'
		tol = 1.0 #stopping criterion
		print('')
		#Create sampler
		sampler = NestedSampler(self.lnlike_nest,self.lnprior_nest,ndim,
				bound=bound,sample=sample,nlive=nlive)

		#run sampler
		sampler.run_nested(dlogz=tol, print_progress=True) # Output the progress bar

		res = sampler.results # get results dictionary from sampler

		logZdynesty = res.logz[-1]        # value of logZ
		logZerrdynesty = res.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

		# draw posterior samples
		weights = np.exp(res['logwt'] - res['logz'][-1])
		samples_dynesty = resample_equal(res.samples, weights)

		#print results

		output = (logZdynesty, logZerrdynesty)
		print()
		print('evidence: %.3f +- %.3f' % (logZdynesty, logZerrdynesty) )
		print()
		print(res.summary())
		print()
		print('parameter values:')
		#for name, col in zip(parameters, samples_dynesty.transpose()):
		#	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
		print('%15s : %.3f +- %.3f' % (self.parameter_labels[0],np.mean(samples_dynesty[:,0]),np.std(samples_dynesty[:,0])) )
		print('%15s : %.3f +- %.3f' % (self.parameter_labels[1],np.mean(samples_dynesty[:,1]),np.std(samples_dynesty[:,1])) )
		print('%15s : %.3f +- %.3f' % (self.parameter_labels[2],np.mean(samples_dynesty[:,2]),np.std(samples_dynesty[:,2])) )



		return

	def nestling(self):
		# Run nested sampling.
		#rstate = np.random.RandomState(1)
		result = nestle.sample(self.lnlike_nest, self.lnprior_nest,
				3, npoints=400,method= 'multi',
				callback = nestle.print_progress)
		print(result.summary())

		nweights = result.weights/np.max(result.weights)
		# re-scale weights to have a maximum of one
		nweights = result.weights/np.max(result.weights)
'
		# get the probability of keeping a sample from the weights
		keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

		# get the posterior samples
		samples_nestle = result.samples[keepidx,:]

		print np.mean(samples_nestle[:,0])
		print np.mean(samples_nestle[:,1])
		print np.mean(samples_nestle[:,2])
		return

	def fit(self):
		
		if self.p is None:
			raise Exception('Error in SingleLensFitter.fit(): No initial_parameters found.')
			return None

		print 'Initial parameters:', self.p
		print 'ln Prob = ',self.lnprob(self.p)

		ndim = self.ndim

		if self.emcee_optimize_first:

			print 'Optimising...'

			optimize.minimize(self.neglnprob,self.p,method='Nelder-Mead')

			print 'Optimized parameters:', self.p
			print 'ln Prob = ',self.lnprob(self.p)

		print ndim, len(self.p), self.nwalkers

		self.state = [self.p + 1e-8 * np.random.randn(ndim) \
						for i in xrange(self.nwalkers)]

		sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.lnprob)

		print("Running burn-in...")

		iteration = 0
		converged = False
		steps = 0

		self.count = 0

		print 'ndim, walkers, nsteps, max_iterations:', ndim, self.nwalkers, self.nsteps, self.max_burnin_iterations

		while not converged and iteration < self.max_burnin_iterations:

			self.state, lnp , _ =sampler.run_mcmc(self.state,self.nsteps,progress =True)

			iteration += 1
			print 'iteration', iteration, 'completed'

			kmax = np.argmax(sampler.flatlnprobability)
			self.p = sampler.flatchain[kmax,:]

			np.save(self.plotprefix+'-state-burnin',np.asarray(self.state))

			if self.make_plots:

				self.plot_chain(sampler,suffix='-burnin.png')

				ind = 3

				if self.use_finite_source:
					ind += 1

				npar = 0
				if self.use_mixture_model:
					npar += 3
				if self.use_gaussian_process_model:
					npar += 2

				if npar > 0:

					for data_set_name in self.data.keys():
					
						self.plot_chain(sampler,index=range(ind,npar+ind),  \
								suffix='-burnin-'+data_set_name+'.png', \
								labels=self.parameter_labels[ind:ind+npar])
						ind += npar

				self.plot_combined_lightcurves()

			converged = self.emcee_has_converged(sampler,n_steps=self.nsteps)

		print("Running production...")

		sampler.reset()

		self.state, lnp, _ = sampler.run_mcmc(self.state,self.nsteps_production,progress =True)

		self.samples = sampler.flatchain

		params = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
								zip(*np.percentile(self.samples[:,:4], \
								[16, 50, 84], axis=0)))

		self.p = np.asarray(params)[:,0]

		self.u0 = self.p[0]
		self.t0 = self.p[1]
		self.tE = self.p[2]

		if self.use_finite_source:
			self.rho = self.p[self.finite_source_index]

		if self.make_plots:

			self.plot_chain(sampler,suffix='-final.png')
			ind = 3

			if self.use_finite_source:
				ind += 1

			if npar > 0:

				for data_set_name in self.data.keys():
				
					self.plot_chain(sampler,index=range(ind,npar+ind),  \
							suffix='-final-'+data_set_name+'.png', \
							labels=self.parameter_labels[ind:ind+npar])
					ind += npar

			self.plot_lightcurves()
			self.plot_chain_corner()

		print 'Results:'
		print 'u0', params[0]
		print 't0', params[1]
		print 'tE', params[2]
		if self.use_finite_source:
			print 'rho', params[self.finite_source_index]

		with open(self.plotprefix+'.fit_results','w') as fid:
			fid.write('u0 %f %f %f\n'%(params[0][0],params[0][1],params[0][2]))
			fid.write('t0 %f %f %f\n'%(params[1][0],params[1][1],params[1][2]))
			fid.write('tE %f %f %f\n'%(params[2][0],params[2][1],params[2][2]))
			if self.use_finite_source:
				pi = self.finite_source_index
				fid.write('rho %f %f %f\n'%(params[pi][0],params[pi][1],params[pi][2]))


		np.save(self.plotprefix+'-state-production',np.asarray(self.state))
		np.save(self.plotprefix+'-min_chi2-production',np.asarray(sampler.flatchain[np.argmax(sampler.flatlnprobability)]))

		return


	def plot_chain(self,s,index=None,plot_lnprob=True,suffix='',labels=None):

		if index is None:
			index = range(self.ndim)

		if labels is None:
			labels = self.parameter_labels

		ndim = len(index)

		plt.figure(figsize=(8,11))
		
		subplots_adjust(hspace=0.0001)

		for i in range(ndim):

			if i == 0:
				plt.subplot(ndim+plot_lnprob,1,i+1)
				ax1 = plt.gca()
			else:
				plt.subplot(ndim+plot_lnprob,1,i+1,sharex=ax1)

			plt.plot(s.chain[:,:,index[i]].T, '-', color='k', alpha=0.3)

			if labels:
				plt.ylabel(labels[i])

			ax = plt.gca()

			if i < ndim-1+plot_lnprob:
				plt.setp(ax.get_xticklabels(), visible=False)
				ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
				ax.locator_params(axis='y',nbins=4)

		if plot_lnprob:
			plt.subplot(ndim+plot_lnprob,1,ndim+plot_lnprob,sharex=ax1)
			plt.plot(s.lnprobability.T, '-', color='r', alpha=0.3)
			plt.ylabel(r"$ln P$")
			ax = plt.gca()
			ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
			ax.locator_params(axis='y',nbins=4)

		plt.savefig(self.plotprefix+suffix)
		plt.close()



	def compute_lightcurve(self,data_key, x, params=None):

		t, _, _ = self.data[data_key]
		coeffs, _ = self.linear_fit(data_key,self.magnification(t,params))

		if self.fit_blended:
			fx = coeffs[0]+coeffs[1]*self.magnification(x,params)
		else:
			fx = coeffs[0]*(self.magnification(x,params)-1)

		return fx


	def plot_combined_lightcurves(self,t_range=None,y_range=None):

		plt.figure(figsize=(8,11))
	
		colour = iter(plt.cm.jet(np.linspace(0,1,len(self.data))))
	
		if t_range is None:
			t_min = self.p[1]-2*self.p[2]
			t_max = self.p[1]+2*self.p[2]
		else:
			t_min, t_max = t_range

		n_data = len(self.data)

		gs = gridspec.GridSpec(2,1,height_ratios=(3,1))

		# Main lightcurve plot

		ax0 = plt.subplot(gs[0])

		a0 = {}
		a1 = {}

		for site in self.data.keys():

			t, y, yerr = self.data[site]
			mag = self.magnification(t)
			a, lnprob = self.linear_fit(site, mag)
			a0[site] = a[1]
			a1[site] = a[0]

		for k, site in enumerate(self.data.keys()):

			scaled_dflux = a0[self.reference_source]*((self.data[site][1] - a1[site])/a0[site]) + a1[self.reference_source]
			scaled_dflux_err = a0[self.reference_source]*((self.data[site][1] + self.data[site][2] - a1[site])/a0[site]) + \
								a1[self.reference_source] - scaled_dflux

			data_merge = 25 - 2.5*np.log10(scaled_dflux)
			sigs_merge = np.abs(25 - 2.5*np.log10(scaled_dflux+scaled_dflux_err) - data_merge)

			ax0.errorbar(self.data[site][0], data_merge, sigs_merge, fmt='.', ms=2, mec=self.plot_colours[k], \
				c=self.plot_colours[k], label=site)

		# Plot the model

		t_plot = np.linspace(t_min,t_max,10001)

		A = self.magnification(t_plot)

		ax0.plot(t_plot,25-2.5*np.log10(a0[self.reference_source]*A+a1[self.reference_source]),'b-')
		ax0.invert_yaxis()
		plt.ylabel(r'$I_{'+self.reference_source+r'}$')

		ax0.grid()

		plt.legend()
		plt.xlabel('HJD-2450000')

		if y_range is not None:
			plt.ylim(y_range)

		if t_range is not None:
			plt.xlim(t_range)

		xlim = ax0.get_xlim()

		# Residuals plot

		ax1 = plt.subplot(gs[1],sharex=ax0)

		for k, site in enumerate(self.data.keys()):
	
			A = self.magnification(self.data[site][0])

			scaled_dflux = a0[self.reference_source]*((self.data[site][1] - a1[site])/a0[site]) + a1[self.reference_source]
			scaled_dflux_err = a0[self.reference_source]*((self.data[site][1] + self.data[site][2] - a1[site])/a0[site]) + \
								a1[self.reference_source] - scaled_dflux

			scaled_model = 25 -2.5*np.log10(a0[self.reference_source]*A + a1[self.reference_source])
			data_merge = 25 - 2.5*np.log10(scaled_dflux) 
			sigs_merge = np.abs(25 - 2.5*np.log10(scaled_dflux+scaled_dflux_err) - data_merge)

			data_merge -= scaled_model

			ax1.errorbar(self.data[site][0], data_merge, sigs_merge, fmt='.', ms=2, mec=self.plot_colours[k], \
				c=self.plot_colours[k], label=site)

		ax1.set_xlim(xlim)
		ax1.grid()

		if y_range is not None:
			ymean = np.mean(y_range)
			y_range = (y_range[0]-ymean,y_range[1]-ymean)
			plt.ylim(y_range)

		plt.xlabel('HJD-2450000')
		ax1.invert_yaxis()
		plt.ylabel(r'$\Delta I_{'+self.reference_source+r'}$')

		plt.tight_layout()

		plt.savefig(self.plotprefix+'-combined-lightcurve')
		plt.close()




	def plot_lightcurves(self):

		plt.figure(figsize=(8,11))
	
		colour = iter(plt.cm.jet(np.linspace(0,1,len(self.data))))

		xmin = self.initial_parameters[1]-2*self.initial_parameters[2]
		xmax = self.initial_parameters[1]+2*self.initial_parameters[2]

		n_data = len(self.data)
		for i, data_set_name in enumerate(self.data.keys()):

			t, y, yerr = self.data[data_set_name]
			#c=next(colour)
			c = 'r'

			if i == 0:
				plt.subplot(n_data,1,i+1)
				ax1 = plt.gca()
			else:
				plt.subplot(n_data,1,i+1,sharex=ax1)

			y_cond = y
			if self.eigen_lightcurves is not None:
				if data_set_name in self.eigen_lightcurves:
					coeffs, _ = self.linear_fit(data_set_name,self.magnification(t))
					ci = 1
					if self.fit_blended:
						ci = 2
					eigs = self.eigen_lightcurves[data_set_name]
					for j in range(eigs.shape[0]):
						y_cond -= coeffs[ci+j]*eigs[j,:]

			plt.errorbar(t, y_cond, yerr=yerr, fmt=".", color=c, capsize=0)
			ax=plt.gca()
			ax.set_xlim(xmin,xmax)
			plt.xlabel(r"$\Delta t (d)$")
			plt.ylabel(data_set_name+r"  $\Delta F$")
	
			x = np.linspace(xmin,xmax, 3000)

			if not(self.use_gaussian_process_model):
				plt.plot(x, self.compute_lightcurve(data_set_name,x),color="k")
				ylim = ax.get_ylim()
		
				# Plot posterior samples.
			for s in self.samples[np.random.randint(len(self.samples), size=self.n_plot_samples)]:

				if self.use_gaussian_process_model:

					# Set up the GP for this sample.
					a, tau = np.exp(s[3+2*i:3+2*i+2])
					gp = george.GP(a * kernels.ExpKernel(tau))
					gp.compute(t, yerr)
					self.cov = gp.get_matrix(t)
					modelt = self.compute_lightcurve(data_set_name,t,params=s)
					modelx = self.compute_lightcurve(data_set_name,x,params=s)

					# Compute the prediction conditioned on the observations
					# and plot it.
					m = gp.sample_conditional(y - modelt,x) + modelx
					plt.plot(x, m, color="#4682b4", alpha=0.3)

				else:

					plt.plot(x, self.compute_lightcurve(data_set_name,x,params=s), \
							color="#4682b4",alpha=0.3)

			if not(self.use_gaussian_process_model):
				ax.set_ylim(ylim)

		plt.savefig(self.plotprefix+'-lc.png')

		plt.close()

 
	def plot_chain_corner(self):

		if self.use_finite_source:

			figure = corner.corner(self.samples[:,:4],
						labels=[r"$u_0$",r"$t_0$",r"$t_E$",r"$\rho$"],
						quantiles=[0.16, 0.5, 0.84],
						truths=(self.u0,self.t0,self.tE,self.rho),
						show_titles=True, title_args={"fontsize": 12})

		else:

			figure = corner.corner(self.samples[:,:3],
						labels=[r"$u_0$",r"$t_0$",r"$t_E$"],
						quantiles=[0.16, 0.5, 0.84],
						truths=(self.u0,self.t0,self.tE),
						show_titles=True, title_args={"fontsize": 12})

		figure.savefig(self.plotprefix+'-pdist.png')

