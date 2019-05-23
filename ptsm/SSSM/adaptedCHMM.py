#Override some of the continuous HMM methods for SSSM
#Letters and notations differs from the ones in SSSM

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from utils import randgen
from ..HMM import continuous_observation_HMM as cHMM

class adaptedCHMM(cHMM):
	def __init__(self,A,pi,S,S_h,S_o,transLDSdict,emisLDSdict,pi_mLDSdict,pi_sLDSdict,E_h_dict,E_o_dict,h_list,H_list,g_list,G_list,K=0):
		self.A = A #Transition matrix
		#self.C = C #Matrix of means(O-by-S dimensional)
		#self.R = R #Observation noise matrix(O-by-O dimensional)
		self.pi = pi #prior state distributions for first hidden variable
		self.S = S #Number of possible states
		#self.O = O #Dimensionality of observations
		self.S_h = S_h #Dimensionality of continuous latent variables
		self.S_o = S_o #Dimensionality of observations
		self.K = K #Number of time slices
		self.transLDSdict = transLDSdict #Dict of transition matrices of LDS
		self.emisLDSdict = emisLDSdict #Dict of emission matrices of LDS
		self.pi_mLDSdict = pi_mLDSdict #Dict with prior state means for the first continuous hidden variables for each state
		self.pi_sLDSdict = pi_sLDSdict #Dict with prior state covariances for the first continuous hidden variables for each state
		self.E_h_dict = E_h_dict #Dict of state transition noise covariance matrices of LDS
		self.E_o_dict = E_o_dict #Dict of emission noise covariance matrices of LDS
		self.h_list = h_list #List of means for approximate posterior marginals q(x_t)
		self.H_list = H_list #List of covariances for approximate posterior marginals q(x_t)
		self.g_list = g_list #List of means for approximate posterior joints q(x_t_plus_1,x_t)
		self.G_list = G_list #List of covariances for approximate posterior joints q(x_t_plus_1,x_t)
	#Inference Forward pass
	def forward_pass(self,y):
		log_alpha = np.zeros((self.S,self.K))
		log_alpha_predict = np.zeros((self.S,self.K))
		for k in range(self.K):
			h_t, H_t = self.h_list[k], self.H_list[k]
			if k == 0:
				log_alpha_predict[:,k] = np.log(self.pi)
				log_alpha[:,k] = self.state_update_z1(y[k],log_alpha_predict[:,k],self.transLDSdict,self.emisLDSdict,self.pi_mLDSdict,self.pi_sLDSdict,self.E_o_dict,self.S,self.S_h,self.S_o,h_t,H_t)
			else:
				h_t_old, H_t_old = self.h_list[k-1], self.H_list[k-1]
				g_t_t_old,G_t_t_old = self.g_list[k-1], self.G_list[k-1]
				log_alpha_predict[:,k] = self.state_predict(self.A,log_alpha[:,k-1])
				log_alpha[:,k] = self.state_update(y[k],log_alpha_predict[:,k],self.transLDSdict,self.emisLDSdict,self.E_h_dict,self.E_o_dict,self.S,self.S_h,self.S_o,h_t,h_t_old,H_t,H_t_old,g_t_t_old,G_t_t_old)
		return log_alpha, log_alpha_predict

	#Inference Backward pass
	def backward_pass(self,y):
		log_beta = np.ones((self.S,self.K))
		log_beta_postdict = np.zeros((self.S,self.K))
		for k in range(self.K):
			t = self.K - 1 - k
			h_t, H_t = self.h_list[t], self.H_list[t]
			if t == (self.K - 1):
				h_t_old, H_t_old = self.h_list[t-1], self.H_list[t-1]
				g_t_t_old,G_t_t_old = self.g_list[t-1], self.G_list[t-1]
				log_beta[:,t] = self.state_update(y[t],log_beta_postdict[:,t],self.transLDSdict,self.emisLDSdict,self.E_h_dict,self.E_o_dict,self.S,self.S_h,self.S_o,h_t,h_t_old,H_t,H_t_old,g_t_t_old,G_t_t_old)
			elif t > 0:
				h_t_old, H_t_old = self.h_list[t-1], self.H_list[t-1]
				g_t_t_old,G_t_t_old = self.g_list[t-1], self.G_list[t-1]
				log_beta_postdict[:,t] = self.state_postdict(self.A,log_beta[:,t+1])
				log_beta[:,t] = self.state_update(y[t],log_beta_postdict[:,t],self.transLDSdict,self.emisLDSdict,self.E_h_dict,self.E_o_dict,self.S,self.S_h,self.S_o,h_t,h_t_old,H_t,H_t_old,g_t_t_old,G_t_t_old)
			else:
				log_beta_postdict[:,t] = self.state_postdict(self.A,log_beta[:,t+1])
				log_beta[:,t] = self.state_update_z1(y[t],log_beta_postdict[:,t],self.transLDSdict,self.emisLDSdict,self.pi_mLDSdict,self.pi_sLDSdict,self.E_o_dict,self.S,self.S_h,self.S_o,h_t,H_t)
		return log_beta, log_beta_postdict

	def state_update_z1(self,y_t,log_p,transLDSdict,emisLDSdict,pi_m_dict,pi_s_dict,E_o_dict,S,S_h,S_o,h_1,H_1):
		b = []
		for s in range(S):
			transLDS = transLDSdict[s]
			emisLDS = emisLDSdict[s]
			E_o = E_o_dict[s]
			pi_m = pi_m_dict[s]
			pi_s = pi_s_dict[s]
			pi_s_inv = np.linalg.inv(pi_s+10**-5)
			h1_h1 = H_1 + np.dot(h_1,h_1.T)
			pdf = -0.5 * np.log(np.linalg.det(pi_s))
			pdf += -0.5 * (np.trace(np.dot(pi_s_inv,h1_h1)) - 2*np.dot(h_1.T,np.dot(pi_s_inv,pi_m)) + np.dot(pi_m.T,np.dot(pi_s_inv,pi_m)))
			pdf += self.calculate_o_log_pdf(emisLDS,E_o,y_t,h_1,H_1,S_o)
			b.append(pdf)
		b = np.array(b).reshape(S,)
		return b + log_p

	def state_update(self,y_t,log_p,transLDSdict,emisLDSdict,E_h_dict,E_o_dict,S,S_h,S_o,h_t,h_t_old,H_t,H_t_old,g_t_t_old,G_t_t_old):
		b = []
		for s in range(S):
			transLDS = transLDSdict[s]
			emisLDS = emisLDSdict[s]
			E_h = E_h_dict[s]
			E_o = E_o_dict[s]
			pdf = self.calculate_h_log_pdf(transLDS,E_h,h_t,h_t_old,H_t,H_t_old,g_t_t_old,G_t_t_old,S_h)
			pdf += self.calculate_o_log_pdf(emisLDS,E_o,y_t,h_t,H_t,S_o)
			b.append(pdf)
		b = np.array(b).reshape(S,)
		return b + log_p

	#calculate the log-pdf comes from continuous latent variables
	@staticmethod
	def calculate_h_log_pdf(A,E,h_t,h_t_old,H_t,H_t_old,g_t_t_old,G_t_t_old,S_h):
		if S_h == 1:
			pdf = -0.5 * np.log(E)
			E_inv = 1./E
		else:
			pdf = -0.5 * np.log(np.linalg.det(E))
			E_inv = np.linalg.inv(E+10**-5)
		ht_ht = H_t + np.dot(h_t,h_t.T)
		ht_htold = G_t_t_old[0:S_h,S_h:2*S_h] + np.dot(h_t,h_t_old.T)
		htold_htold = H_t_old + np.dot(h_t_old,h_t_old.T)
		if S_h == 1:
			pdf += -0.5 * (np.dot(E_inv,ht_ht) - 2*np.dot(A.T,np.dot(E_inv,ht_htold)) + np.dot(A.T,np.dot(E_inv,np.dot(A,htold_htold))))
		else:
			pdf += -0.5 * (np.trace(np.dot(E_inv,ht_ht)) - 2*np.trace(np.dot(A.T,np.dot(E_inv,ht_htold))) + np.trace(np.dot(A.T,np.dot(E_inv,np.dot(A,htold_htold)))))
		return pdf

	#calculate the log-pdf comes from observations
	@staticmethod
	def calculate_o_log_pdf(B,E,y_t,h_t,H_t,S_o):
		if S_o == 1:
			pdf = -0.5 * np.log(E)
			E_inv = 1./E
		else:
			pdf = -0.5 * np.log(np.linalg.det(E))
			E_inv = np.linalg.inv(E+10**-5)
		ht_ht = H_t + np.dot(h_t,h_t.T)
		try:
			pdf += -0.5 * (np.dot(np.dot(y_t.T,E_inv),y_t) - 2*np.dot(np.dot(np.dot(h_t.T,B.T),E_inv),y_t) + np.trace(np.dot(B.T,np.dot(E_inv,np.dot(B,ht_ht)))))
		except:
			pdf += -0.5 * (np.dot(np.dot(y_t.T,E_inv),y_t) - 2*np.dot(np.dot(np.dot(h_t.T,B.T),E_inv),y_t) + np.dot(B.T,np.dot(E_inv,np.dot(B,ht_ht))))
		return pdf

	#calculate the pdf vector required for EM of SSSM
	def probs_vector(self,y_t,transLDSdict,emisLDSdict,E_h_dict,E_o_dict,S,S_h,S_o,h_t,h_t_old,H_t,H_t_old,g_t_t_old,G_t_t_old):
		b = []
		for s in range(S):
			transLDS = transLDSdict[s]
			emisLDS = emisLDSdict[s]
			E_h = E_h_dict[s]
			E_o = E_o_dict[s]
			pdf = self.calculate_h_log_pdf(transLDS,E_h,h_t,h_t_old,H_t,H_t_old,g_t_t_old,G_t_t_old,S_h)
			pdf += self.calculate_o_log_pdf(emisLDS,E_o,y_t,h_t,H_t,S_o)
			b.append(pdf) #logpdf
		b = np.array(b).reshape(S,)
		return b

	def probs_vector_z1(self,y_t,transLDSdict,emisLDSdict,pi_m_dict,pi_s_dict,E_o_dict,S,S_h,S_o,h_1,H_1):
		b = []
		for s in range(S):
			transLDS = transLDSdict[s]
			emisLDS = emisLDSdict[s]
			E_o = E_o_dict[s]
			pi_m = pi_m_dict[s]
			pi_s = pi_s_dict[s]
			pi_s_inv = np.linalg.inv(pi_s+10**-5)
			h1_h1 = H_1 + np.dot(h_1,h_1.T)
			pdf = -0.5 * np.log(np.linalg.det(pi_s))
			pdf += -0.5 * (np.trace(np.dot(pi_s_inv,h1_h1)) - 2*np.dot(h_1.T,np.dot(pi_s_inv,pi_m)) + np.dot(pi_m.T,np.dot(pi_s_inv,pi_m)))
			pdf += self.calculate_o_log_pdf(emisLDS,E_o,y_t,h_1,H_1,S_o)
			b.append(pdf) #logpdf
		b = np.array(b).reshape(S,)
		return b

	#Inference Forward pass
	def probs_matrix(self,y):
		PM = np.zeros((self.S,self.K))
		for k in range(self.K):
			h_t, H_t = self.h_list[k], self.H_list[k]
			if k == 0:
				PM[:,k] = self.probs_vector_z1(y[k],self.transLDSdict,self.emisLDSdict,self.pi_mLDSdict,self.pi_sLDSdict,self.E_o_dict,self.S,self.S_h,self.S_o,h_t,H_t)
			else:
				h_t_old, H_t_old = self.h_list[k-1], self.H_list[k-1]
				g_t_t_old,G_t_t_old = self.g_list[k-1], self.G_list[k-1]
				PM[:,k] = self.probs_vector(y[k],self.transLDSdict,self.emisLDSdict,self.E_h_dict,self.E_o_dict,self.S,self.S_h,self.S_o,h_t,h_t_old,H_t,H_t_old,g_t_t_old,G_t_t_old)
		return PM
