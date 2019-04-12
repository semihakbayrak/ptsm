import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
#add parent directory to path
import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
	sys.path.insert(0, parentPath)
#from utils import randgen, log_sum_exp, normalize, normalize_exp
sys.path.insert(1, parentPath+"/HMM")
sys.path.insert(2, parentPath+"/LDS")
from utils import randgen, normalize_exp
#from hmm import continuous_observation_HMM as cHMM
from adaptedCHMM import adaptedCHMM as cHMM
from lds import LDS

class SSSM:
	def __init__(self,M,pi_d,A,B,pi_m,pi_s,K,S,O,E_h,E_o,T=0):
		self.M = M #Transition matrix of discrete hidden variables
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
		#Dictionary to tensor
		self.A_tensor = np.zeros((self.K,self.S,self.S))
		for k in range(self.K): self.A_tensor[k,:,:] = self.A[k]
		self.E_h_tensor = np.zeros((self.K,self.S,self.S))
		for k in range(self.K): self.E_h_tensor[k,:,:] = self.E_h[k]
		self.B_tensor = np.zeros((self.K,self.O,self.S))
		for k in range(self.K): self.B_tensor[k,:,:] = self.B[k]
		self.E_o_tensor = np.zeros((self.K,self.O,self.O))
		for k in range(self.K): self.E_o_tensor[k,:,:] = self.E_o[k]

	def generate_data(self,T):
		#T is the number of time slices to simulate
		self.T = T
		discrete_states = []
		continuous_states = []
		observations = []
		for t in range(T):
			if t == 0:
				d_state = randgen(self.pi_d)
				if self.S == 1:
					c_state = np.random.normal(self.pi_m[d_state],self.pi_s[d_state])
				else:
					c_state = np.random.multivariate_normal(self.pi_m[d_state],self.pi_s[d_state])
			else:
				d_state = randgen(self.M[:,discrete_states[t-1]])
				mu = np.dot(self.A[d_state],continuous_states[t-1])
				if self.S == 1:
					c_state = np.random.normal(mu,self.E_h[d_state])
				else:
					c_state = np.random.multivariate_normal(mu,self.E_h[d_state])
			discrete_states.append(d_state)
			continuous_states.append(c_state)
			mu = np.dot(self.B[d_state],continuous_states[t])
			if self.O == 1:
				observation = np.random.normal(mu,self.E_o[d_state])
			else:
				observation = np.random.multivariate_normal(mu,self.E_o[d_state])
			observations.append(observation)
		return discrete_states,continuous_states,observations

	#Structured VI to infer continuous and discrete latent states for known model parameters
	def structured_vi(self,y,num_iterations=100):
		#Initialization of q(z)
		Q_z = np.random.rand(self.K,self.T)
		Q_z = Q_z/np.sum(Q_z,axis=0)
		for _ in range(num_iterations):
			A_expectations = np.einsum('kij,kt->tij',self.A_tensor,Q_z)
			E_h_expectations = np.einsum('kij,kt->tij',self.E_h_tensor,Q_z)
			B_expectations = np.einsum('kij,kt->tij',self.B_tensor,Q_z)
			E_o_expectations = np.einsum('kij,kt->tij',self.E_o_tensor,Q_z)
			#evaluate q(x)
			#approximate posterior marginals q(x_t)
			#filtering
			f_list, F_list = [], []
			f = np.zeros(self.S).reshape(self.S,1)
			F = np.zeros((self.S,self.S))
			for t in range(self.T):
				A_t = A_expectations[t]
				B_t = B_expectations[t]
				E_h_t = E_h_expectations[t]
				E_o_t = E_o_expectations[t]
				f, F = LDS.filter_one_step(y[t],A_t,B_t,E_h_t,E_o_t,f,F,self.S,self.O)
				#print f
				f_list.append(f)
				F_list.append(F)
			#smoothing
			h_list, H_list = [], []
			h, H = f_list[-1], F_list[-1]
			h_list.append(h)
			H_list.append(H)
			for i in range(self.T-1):
				t = self.T - i - 2
				f_t = f_list[t]
				F_t = F_list[t]
				A_t_plus_1 = A_expectations[t+1]
				E_h_t_plus_1 = E_h_expectations[t+1]
				h, H = LDS.smooth_one_step(A_t_plus_1, E_h_t_plus_1, f_t, F_t, h, H,self.S,self.O)
				h_list.append(h)
				H_list.append(H)
			h_list.reverse(), H_list.reverse()
			#approximate posterior joints q(x_t_plus_1,x_t)
			g_list, G_list = [], []
			for t in range(self.T-1):
				g = np.zeros((2*self.S,1))
				G = np.eye(2*self.S)
				g[0:self.S] = h_list[t+1]
				G[0:self.S,0:self.S] = H_list[t+1]
				A_t_plus_1 = A_expectations[t+1]
				E_h_t_plus_1 = E_h_expectations[t+1]
				P1 = F_list[t]
				P21 = np.dot(A_t_plus_1,F_list[t])
				P12 = P21.T
				P2 = np.dot(P21,A_t_plus_1.T) + E_h_t_plus_1
				P2inv = np.linalg.inv(P2+np.eye(self.S)*10**-8)
				g[self.S:2*self.S] = f_list[t] - np.dot(np.dot(P12,P2inv),np.dot(A_t_plus_1,f_list[t])) + np.dot(np.dot(P12,P2inv),h_list[t+1])
				G_bot_left = np.dot(P12,np.dot(P2inv,H_list[t+1]))
				G_top_right = G_bot_left.T
				G_bot_right = np.dot(G_bot_left,np.dot(P2inv.T,P21)) + P1 - np.dot(np.dot(P12,P2inv),P21)
				G[self.S:2*self.S,0:self.S] = G_bot_left
				G[0:self.S,self.S:2*self.S] = G_top_right
				G[self.S:2*self.S,self.S:2*self.S] = G_bot_right
				g_list.append(g)
				G_list.append(G)
			#evaluate q(z)
			hmm = cHMM(self.M,self.pi_d,self.K,self.S,self.O,self.A,self.B,self.pi_m,self.pi_s,self.E_h,self.E_o,h_list,H_list,g_list,G_list,self.T)
			log_gamma = hmm.forward_backward(y)
			gamma = normalize_exp(log_gamma)
			Q_z = gamma

		return h_list,H_list,gamma
