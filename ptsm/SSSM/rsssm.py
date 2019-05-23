import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from utils import randgen, normalize_exp
#from hmm import continuous_observation_HMM as cHMM
from .adaptedCHMM import adaptedCHMM as cHMM
from ..LDS import LDS
from .sssm import SSSM

class robustSSSM(SSSM):
	def __init__(self,D,pi_d,A,B,pi_m,pi_s,K,S,O,E_h,E_o,T=0):
		self.D = D #Dirichlet priors - Dirichlet distributions are column-wise 
		self.M_mean = self.D/np.sum(self.D,axis=0) #Expectation of transition matrix of HMM
		self.pi_d = pi_d #prior state distributions for first discrete hidden variable
		self.K = K #Number of discrete states
		self.A = A #Dict that contains transition matrices for continuous hidden variables. Key values must be ascending ordered integers that start from 0
		self.B = B #Dict that contains emission matrices. Key values must be ascending ordered integers that start from 0
		self.pi_m = pi_m #Dict with prior state means for the first continuous hidden variables for each state
		self.pi_s = pi_s #Dict with prior state covariances for the first continuous hidden variables for each state
		self.E_h = E_h #Dict with covariances for states noise
		self.E_o = E_o #Dict with covariances for observation noise
		self.S = S #Dimensionality of states
		self.O = O #Dimensionality of observations
		self.T = T #Number of time slices
		super(robustSSSM,self).__init__(self.M_mean,pi_d,A,B,pi_m,pi_s,K,S,O,E_h,E_o,T)

	#Parameter estimation, Bayesian inference for states and transition matrix of HMM
	def VEM(self,y,A_init={0:np.nan},B_init={0:np.nan},pi_m_init={0:np.nan},pi_s_init={0:np.nan},pi_d_init=np.nan,E_h_init={0:np.nan},E_o_init={0:np.nan},estimate=['M','A','B','pi_m','pi_s','pi_d','E_h','E_o'],num_iterations_VEM=20,num_iterations_SVI=40):
		if 'M' in estimate:
			estimate_list = []
			for element in estimate: 
				if element != 'M':
					estimate_list.append(element)
		else:
			self.M, self.A, self.B, self.pi_m, self.pi_s, self.E_h, self.E_o, h_list, H_list, gamma, log_alpha,log_beta_postdict,prob_matrix = super(robustSSSM,self).EM(y,pi_d_init=pi_d_init,
					A_init=A_init,B_init=B_init,pi_m_init=pi_m_init,pi_s_init=pi_s_init,E_h_init=E_h_init,E_o_init=E_o_init,estimate=estimate_list,num_iterations_EM=num_iterations_VEM,num_iterations_SVI=num_iterations_SVI)
			return self.M, self.A, self.B, self.pi_m, self.pi_s, self.E_h, self.E_o, h_list, H_list, gamma
		for it in range(num_iterations_VEM):
			#_,_,_,_,_,_,gamma,log_alpha,log_beta_postdict,prob_matrix = super(robustSSSM,self).structured_vi_for_em(y,num_iterations=num_iterations_SVI)
			if it == 0:
				_, self.A, self.B, self.pi_m, self.pi_s, self.E_h, self.E_o, h_list, H_list, gamma, log_alpha,log_beta_postdict,prob_matrix = super(robustSSSM,self).EM(y,pi_d_init=pi_d_init,
					A_init=A_init,B_init=B_init,pi_m_init=pi_m_init,pi_s_init=pi_s_init,E_h_init=E_h_init,E_o_init=E_o_init,estimate=estimate_list,num_iterations_EM=1,num_iterations_SVI=num_iterations_SVI)
			else:
				_, self.A, self.B, self.pi_m, self.pi_s, self.E_h, self.E_o, h_list, H_list, gamma, log_alpha,log_beta_postdict,prob_matrix = super(robustSSSM,self).EM(y,pi_d_init=self.pi_d,
					A_init=self.A,B_init=self.B,pi_m_init=self.pi_m,pi_s_init=self.pi_s,E_h_init=self.E_h,E_o_init=self.E_o,estimate=estimate_list,num_iterations_EM=1,num_iterations_SVI=num_iterations_SVI)
			M_count = np.zeros((self.K,self.K))
			for t in range(self.T):
				if t != 0:
					prob_vector = prob_matrix[:,t]
					log_M_new = np.log(self.M)+prob_vector.reshape(self.K,1)+log_alpha[:,t-1].reshape(1,self.K)+log_beta_postdict[:,t]
					mx = np.max(log_M_new)
					log_M_new = log_M_new - mx
					M_new = np.exp(log_M_new)
					M_new = M_new/M_new.sum()
					M_count = M_count + M_new
			M_estimated = M_count/(np.sum(gamma[:,0:-1],axis=1).reshape(self.K,1))
			self.D += M_estimated
			self.M_mean = self.D/np.sum(self.D,axis=0)
			super(robustSSSM,self).change_M(self.M_mean)
		h_list, H_list, gamma = super(robustSSSM,self).structured_vi(y)
		return self.D, self.A, self.B, self.pi_m, self.pi_s, self.E_h, self.E_o, h_list, H_list, gamma

	#method to modify dirichlet parameters - required for SPLDS
	def change_D(self,new_D):
		self.D = new_D
