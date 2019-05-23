import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from utils import randgen, normalize_exp
from .adaptedCHMM import adaptedCHMM as cHMM
from ..LDS import LDS

class SSSM(object):
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
		self.dict_to_tensor()
		self.Q_z = self.initialize_Q_z(self.K,self.T) #Initialize q(z):Probability of discrete states over time

	def dict_to_tensor(self):
		self.A_tensor = np.zeros((self.K,self.S,self.S))
		for k in range(self.K): self.A_tensor[k,:,:] = self.A[k]
		self.E_h_tensor = np.zeros((self.K,self.S,self.S))
		for k in range(self.K): self.E_h_tensor[k,:,:] = self.E_h[k]
		self.B_tensor = np.zeros((self.K,self.O,self.S))
		for k in range(self.K): self.B_tensor[k,:,:] = self.B[k]
		self.E_o_tensor = np.zeros((self.K,self.O,self.O))
		for k in range(self.K): self.E_o_tensor[k,:,:] = self.E_o[k]

	#method to modify transition matrix of hmm - required to implement dirichlet prior
	def change_M(self,new_M):
		self.M = new_M

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

	#Methods for inference
	@staticmethod
	def initialize_Q_z(K,T):
		Q_z = np.random.rand(K,T)
		Q_z = Q_z/np.sum(Q_z,axis=0)
		return Q_z

	#Expectation of a tensor w.r.t. Q through time
	@staticmethod
	def expectation_through_time(M,Q):
		M_expectations = np.einsum('kij,kt->tij',M,Q)
		return M_expectations

	#Inference-Filtering for LDS part
	def filtering(self,y,A_expectations,B_expectations,E_h_expectations,E_o_expectations,S,O,T):
		f_list, F_list = [], []
		f = np.zeros(S).reshape(S,1)
		F = np.zeros((S,S))
		for t in range(T):
			A_t = A_expectations[t]
			B_t = B_expectations[t]
			E_h_t = E_h_expectations[t]
			E_o_t = E_o_expectations[t]
			f, F = LDS.filter_one_step(y[t],A_t,B_t,E_h_t,E_o_t,f,F,S,O)
			f_list.append(f)
			F_list.append(F)
		return f_list,F_list

	#Inference-Smoothing for LDS part
	def smoothing(self,y,A_expectations,E_h_expectations,f_list,F_list,S,O,T):
		h_list, H_list = [], []
		h, H = f_list[-1], F_list[-1]
		h_list.append(h)
		H_list.append(H)
		for i in range(T-1):
			t = T - i - 2
			f_t = f_list[t]
			F_t = F_list[t]
			A_t_plus_1 = A_expectations[t+1]
			E_h_t_plus_1 = E_h_expectations[t+1]
			h, H = LDS.smooth_one_step(A_t_plus_1, E_h_t_plus_1, f_t, F_t, h, H,S,O)
			h_list.append(h)
			H_list.append(H)
		h_list.reverse(), H_list.reverse()
		return h_list,H_list

	#approximate posterior joints q(x_t_plus_1,x_t)
	def post_joint_continuous_vars_one_step(self,f_t,F_t,h_t_plus_1,H_t_plus_1,A_t_plus_1,E_h_t_plus_1):
		g = np.zeros((2*self.S,1))
		G = np.eye(2*self.S)
		g[0:self.S] = h_t_plus_1
		G[0:self.S,0:self.S] = H_t_plus_1
		P1 = F_t
		P21 = np.dot(A_t_plus_1,F_t)
		P12 = P21.T
		P2 = np.dot(P21,A_t_plus_1.T) + E_h_t_plus_1
		P2inv = np.linalg.inv(P2+np.eye(self.S)*10**-8)
		g[self.S:2*self.S] = f_t - np.dot(np.dot(P12,P2inv),np.dot(A_t_plus_1,f_t)) + np.dot(np.dot(P12,P2inv),h_t_plus_1)
		G_bot_left = np.dot(P12,np.dot(P2inv,H_t_plus_1))
		G_top_right = G_bot_left.T
		G_bot_right = np.dot(G_bot_left,np.dot(P2inv.T,P21)) + P1 - np.dot(np.dot(P12,P2inv),P21)
		G[self.S:2*self.S,0:self.S] = G_bot_left
		G[0:self.S,self.S:2*self.S] = G_top_right
		G[self.S:2*self.S,self.S:2*self.S] = G_bot_right
		return g,G

	def post_joint_continuous_vars(self,f_list,F_list,h_list,H_list,A_expectations,E_h_expectations,T):
		g_list, G_list = [], []
		for t in range(T-1):
			g,G = self.post_joint_continuous_vars_one_step(f_list[t],F_list[t],h_list[t+1],H_list[t+1],A_expectations[t+1],E_h_expectations[t+1])
			g_list.append(g)
			G_list.append(G)
		return g_list,G_list

	#Structured VI to infer continuous and discrete latent states for known model parameters
	def structured_vi(self,y,num_iterations=100):
		for _ in range(num_iterations):
			#Expectation LDS tensors w.r.t. Q_z through time
			A_expectations = self.expectation_through_time(self.A_tensor,self.Q_z)
			E_h_expectations = self.expectation_through_time(self.E_h_tensor,self.Q_z)
			B_expectations = self.expectation_through_time(self.B_tensor,self.Q_z)
			E_o_expectations = self.expectation_through_time(self.E_o_tensor,self.Q_z)
			#evaluate q(x)
			#approximate posterior marginals q(x_t)
			#filtering
			f_list, F_list = self.filtering(y,A_expectations,B_expectations,E_h_expectations,E_o_expectations,self.S,self.O,self.T)
			#smoothing
			h_list, H_list = self.smoothing(y,A_expectations,E_h_expectations,f_list,F_list,self.S,self.O,self.T)
			#approximate posterior joints q(x_t_plus_1,x_t)
			g_list, G_list = self.post_joint_continuous_vars(f_list,F_list,h_list,H_list,A_expectations,E_h_expectations,self.T)
			#evaluate q(z)
			hmm = cHMM(self.M,self.pi_d,self.K,self.S,self.O,self.A,self.B,self.pi_m,self.pi_s,self.E_h,self.E_o,h_list,H_list,g_list,G_list,self.T)
			log_gamma = hmm.forward_backward(y)
			gamma = normalize_exp(log_gamma)
			self.Q_z = gamma

		return h_list,H_list,gamma

	#Structured VI for E-step of em
	def structured_vi_for_em(self,y,num_iterations=100):
		for _ in range(num_iterations):
			#Expectation LDS tensors w.r.t. Q_z through time
			A_expectations = self.expectation_through_time(self.A_tensor,self.Q_z)
			E_h_expectations = self.expectation_through_time(self.E_h_tensor,self.Q_z)
			B_expectations = self.expectation_through_time(self.B_tensor,self.Q_z)
			E_o_expectations = self.expectation_through_time(self.E_o_tensor,self.Q_z)
			#evaluate q(x)
			#approximate posterior marginals q(x_t)
			#filtering
			f_list, F_list = self.filtering(y,A_expectations,B_expectations,E_h_expectations,E_o_expectations,self.S,self.O,self.T)
			#smoothing
			h_list, H_list = self.smoothing(y,A_expectations,E_h_expectations,f_list,F_list,self.S,self.O,self.T)
			#approximate posterior joints q(x_t_plus_1,x_t)
			g_list, G_list = self.post_joint_continuous_vars(f_list,F_list,h_list,H_list,A_expectations,E_h_expectations,self.T)
			#evaluate q(z)
			hmm = cHMM(self.M,self.pi_d,self.K,self.S,self.O,self.A,self.B,self.pi_m,self.pi_s,self.E_h,self.E_o,h_list,H_list,g_list,G_list,self.T)
			log_gamma = hmm.forward_backward(y)
			gamma = normalize_exp(log_gamma)
			self.Q_z = gamma
		log_alpha, log_alpha_predict = hmm.forward_pass(y)
		alpha = normalize_exp(log_alpha)
		log_beta, log_beta_postdict = hmm.backward_pass(y)
		beta = normalize_exp(log_beta)
		beta_postdict = normalize_exp(log_beta_postdict)
		prob_matrix = hmm.probs_matrix(y)
		return f_list,F_list,h_list,H_list,g_list,G_list,gamma,log_alpha,log_beta_postdict,prob_matrix

	#Parameter estimation with EM
	def EM(self,y,M_init=np.nan,pi_d_init=np.nan,A_init={0:np.nan},B_init={0:np.nan},pi_m_init={0:np.nan},pi_s_init={0:np.nan},E_h_init={0:np.nan},E_o_init={0:np.nan},estimate=['M','A','B','pi_m','pi_s','pi_d','E_h','E_o'],num_iterations_EM=20,num_iterations_SVI=40):
		#parameters to be estimated
		M_est = True if 'M' in estimate else False
		A_est = True if 'A' in estimate else False
		B_est = True if 'B' in estimate else False
		pi_m_est = True if 'pi_m' in estimate else False
		pi_s_est = True if 'pi_s' in estimate else False
		pi_d_est = True if 'pi_d' in estimate else False
		E_h_est = True if 'E_h' in estimate else False
		E_o_est = True if 'E_o' in estimate else False
		#initialization
		if np.isnan(M_init).any():
			if M_est == True:
				self.M = np.random.rand(self.K,self.K)
				self.M = self.M/np.sum(self.M,axis=0)
			else:
				pass
		if np.isnan(pi_d_init).any():
			if pi_d_est == True:
				self.pi_d = np.ones(self.K)/self.K
			else:
				pass
		if np.isnan(A_init[0]).any():
			if A_est == True:
				self.A = {}
				for k in range(self.K): self.A[k] = np.random.rand(self.S,self.S)
			else:
				pass	
		if np.isnan(B_init[0]).any():
			if B_est == True:
				self.B = {}
				for k in range(self.K): self.B[k] = np.random.rand(self.O,self.S)
			else:
				pass
		if np.isnan(pi_m_init[0]).any():
			if pi_m_est == True:
				self.pi_m = {}
				for k in range(self.K): self.pi_m[k] = np.random.rand(self.S)
			else:
				pass
		if np.isnan(pi_s_init[0]).any():
			if pi_s_est == True:
				self.pi_s = {}
				for k in range(self.K): self.pi_s[k] = np.eye(self.S)
			else:
				pass
		if np.isnan(E_h_init[0]).any():
			if E_h_est == True:
				self.E_h = {}
				for k in range(self.K): self.E_h[k] = np.eye(self.S)
			else:
				pass
		if np.isnan(E_o_init[0]).any():
			if E_o_est == True:
				self.E_o = {}
				for k in range(self.K): self.E_o[k] = np.eye(self.O) if self.O>1 else 1.0
			else:
				pass
		self.dict_to_tensor()
		#EM iterations
		for iteration in range(num_iterations_EM):
			#E-step
			f_list,F_list,h_list,H_list,g_list,G_list,gamma,log_alpha,log_beta_postdict,prob_matrix = self.structured_vi_for_em(y,num_iterations=num_iterations_SVI)
			#M-step
			#pi_m update
			if pi_m_est == True:
				for k in range(self.K): self.pi_m[k] = h_list[0]
			#pi_s update
			if pi_s_est == True:
				for k in range(self.K): self.pi_s[k] = H_list[0]
			#pi_d update
			if pi_d_est == True:
				self.pi_d = gamma[:,0]
			#A update
			if A_est == True:
				for k in range(self.K):
					sum1, sum2 = 0, 0
					for i in range(self.T-1):
						t = i + 1
						P2 = np.dot(self.A[k],np.dot(F_list[t-1],self.A[k].T)) + self.E_h[k]
						P21 = np.dot(self.A[k],F_list[t-1])
						sum1 += gamma[k,t]*(np.dot(H_list[t].T, np.dot(np.linalg.inv(P2.T+np.eye(self.S)*10**-5),P21)) + np.dot(h_list[t], h_list[t-1].T))
						sum2 += gamma[k,t]*(H_list[t-1] + np.dot(h_list[t-1],h_list[t-1].T))
					self.A[k] = np.dot(sum1, np.linalg.inv(sum2+np.eye(self.S)*10**-5))
			#B update
			if B_est == True:
				for k in range(self.K):
					sum1, sum2 = 0, 0
					for t in range(self.T):
						sum1 += gamma[k,t]*(np.dot(y[t].reshape(self.O,1), h_list[t].T))
						sum2 += gamma[k,t]*(H_list[t] + np.dot(h_list[t],h_list[t].T))
					self.B[k] = np.dot(sum1, np.linalg.inv(sum2+np.eye(self.S)*10**-5))
			#E_o update
			if E_o_est == True:
				for k in range(self.K):
					sum1 = 0 
					sum3 = 0
					for t in range(self.T):
						sum1 += gamma[k,t]*np.dot((np.dot(y[t].reshape(self.O,1), h_list[t].T)),self.B[k].reshape(self.O,1).T)
						sum3 += gamma[k,t]*(np.dot(y[t].reshape(self.O,1), y[t].reshape(1,self.O)))
					self.E_o[k] = (sum3 - sum1)/self.T
			#E_h update
			if E_h_est == True:
				for k in range(self.K):
					sum2 = 0
					sum1 = 0
					for i in range(self.T-1):
						t = i + 1
						sum2 += gamma[k,t]*(H_list[t] + np.dot(h_list[t],h_list[t].T))
						P2 = np.dot(self.A[k],np.dot(F_list[t-1],self.A[k].T)) + self.E_h[k]
						P21 = np.dot(self.A[k],F_list[t-1])
						sum1 += gamma[k,t]*np.dot(self.A[k],(np.dot(H_list[t].T, np.dot(np.linalg.inv(P2.T+np.eye(self.S)*10**-5),P21)) + np.dot(h_list[t], h_list[t-1].T)).T)
					self.E_h[k] = (sum2 - sum1)/(self.T-1)
			#M update
			if M_est == True:
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
				M_estimated = M_estimated/np.sum(M_estimated,axis=0)
				self.M = M_estimated
		#Run and return the smoothing one more time after parameters are estimated
		#h_list, H_list, gamma = self.structured_vi(y)
		return self.M, self.A, self.B, self.pi_m, self.pi_s, self.E_h, self.E_o, h_list, H_list, gamma, log_alpha,log_beta_postdict,prob_matrix
