import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

'''
LDS class works with the generic array definition of numpy.
For example: the dimensionalities S,O can take positive integer values 1,2,3,4,...
Model parameter definition is compatible with the random array definition of numpy
A = np.random.rand(S,S)
B = np.random.rand(O,S)
pi_m = np.random.rand(S)
pi_s = np.eye(S)
E_h = np.eye(S)
E_o = np.eye(O)
for all valid values of S and O i.e. 1,2,3,4,...
'''

class LDS:
	def __init__(self,A,B,pi_m,pi_s,S,O,E_h,E_o,K=0):
		self.A = A #Transition matrix
		self.B = B #Emission matrix
		self.pi_m = pi_m #prior state mean
		self.pi_s = pi_s #prior state covariance
		self.S = S #Dimensionality of states
		self.O = O #Dimensionality of observations
		self.E_h = E_h #Covariance for state noise
		self.E_o = E_o #Covariance for observation noise
		self.K = K #Number of time slices

	#Generate simulated data
	def generate_data(self,K):
		#K is the number of time slices to simulate
		self.K = K
		states = []
		observations = []
		for k in range(K):
			if k == 0:
				if self.S == 1:
					state = np.random.normal(self.pi_m,self.pi_s)
				else:
					state = np.random.multivariate_normal(self.pi_m,self.pi_s)
			else:
				mu = np.dot(self.A,states[k-1])
				if self.S == 1:
					state = np.random.normal(mu,self.E_h)
				else:
					state = np.random.multivariate_normal(mu,self.E_h)
			states.append(state)
			mu = np.dot(self.B,states[k])
			if self.O == 1:
				observation = np.random.normal(mu,self.E_o)
			else:
				observation = np.random.multivariate_normal(mu,self.E_o)
			observations.append(observation)
		return states,observations

	#methods for filtering and smoothing
	@staticmethod
	def filter_one_step(y_t, A_t, B_t, E_h_t, E_o_t, f_t_old, F_t_old, S, O):
		m_h = np.dot(A_t,f_t_old)
		m_o = np.dot(B_t,m_h)
		E_hh = np.dot(A_t,np.dot(F_t_old,A_t.T)) + E_h_t
		E_oo = np.dot(B_t,np.dot(E_hh,B_t.T)) + E_o_t
		E_oh = np.dot(B_t,E_hh)
		E_ho = E_oh.T
		f_t = m_h + np.dot(np.dot(E_ho,np.linalg.inv(E_oo+np.eye(O)*10**-8)),(y_t.reshape(O,1)-m_o))
		F_t = E_hh - np.dot(np.dot(E_ho,np.linalg.inv(E_oo+np.eye(O)*10**-8)),E_oh)
		return f_t, F_t

	@staticmethod
	def smooth_one_step(A_t_plus_1, E_h_t_plus_1, f_t, F_t, h_t_plus_1, H_t_plus_1, S, O):
		P1 = F_t
		P21 = np.dot(A_t_plus_1,F_t)
		P12 = P21.T
		P2 = np.dot(P21,A_t_plus_1.T) + E_h_t_plus_1
		P2inv = np.linalg.inv(P2+np.eye(S)*10**-8)
		h_t = f_t - np.dot(np.dot(P12,P2inv),np.dot(A_t_plus_1,f_t)) + np.dot(np.dot(P12,P2inv),h_t_plus_1)
		H_t = np.dot(np.dot(P12,np.dot(P2inv,H_t_plus_1)),np.dot(P2inv.T,P21)) + P1 - np.dot(np.dot(P12,P2inv),P21)
		return h_t, H_t

	#Inference-Filtering
	def filtering(self,y,f=0,F=0,f_list=None,F_list=None,count=0):
		f_list = [] if f_list==None else f_list
		F_list = [] if F_list==None else F_list
		if type(f) == int:
			if f == 0:
				if self.S>1:
					#f = np.zeros(self.S).reshape(self.S,1)
					f = self.pi_m.reshape(self.S,1)
		if type(F) == int:
			if F == 0:
				if self.S>1:
					#F = np.zeros((self.S,self.S))
					F = self.pi_s
		f_new, F_new = self.filter_one_step(y[count],self.A,self.B,self.E_h,self.E_o,f,F,self.S,self.O)
		f_list.append(f_new)
		F_list.append(F_new)
		count = count + 1
		if count < len(y):
			self.filtering(y,f_new,F_new,f_list,F_list,count)
			return f_list, F_list
		else:
			return f_list, F_list

	#Inference-Backward
	def backward(self,y,g=0,G=0,g_list=None,G_list=None,count=0):
		g_list = [] if g_list==None else g_list
		G_list = [] if G_list==None else G_list
		if count == 0:
			g_new = np.dot(np.linalg.pinv(self.B),y[self.K-count-1].reshape(self.O,1))
			G_new = np.dot(np.linalg.pinv(self.B),np.dot(self.E_o,np.linalg.pinv(self.B.T)))
			g_list.append(g_new)
			G_list.append(G_new)
			count += 1
			self.backward(y,g_new,G_new,g_list,G_list,count)
			return g_list, G_list
		elif count < len(y):
			m_h = np.dot(np.linalg.inv(self.A),g)
			m_o = np.dot(self.B,m_h)
			E_hh = np.dot(np.linalg.inv(self.A),np.dot(G + self.E_h, np.linalg.inv(self.A.T)))
			E_oo = np.dot(self.B,np.dot(E_hh,self.B.T)) + self.E_o
			E_oh = np.dot(self.B,E_hh)
			E_ho = E_oh.T
			g_new = m_h + np.dot(np.dot(E_ho,np.linalg.inv(E_oo)),(y[count].reshape(self.O,1)-m_o))
			G_new = E_hh - np.dot(np.dot(E_ho,np.linalg.inv(E_oo)),E_oh)
			g_list.append(g_new)
			G_list.append(G_new)
			count = count + 1
			self.backward(y,g_new,G_new,g_list,G_list,count)
			return g_list, G_list
		else:
			return g_list.reverse(), G_list.reverse()

	#Inference-Smoothing but unstable because uses backward
	def smoothing_unstable(self,y):
		f_list, F_list = self.filtering(y)
		g_list, G_list = self.backward(y)
		h_list, H_list = [], []
		for t in range(self.K):
			if t != self.K-1:
				F, G = F_list[t], G_list[t+1]
				f, g = f_list[t], g_list[t+1]
				H = np.linalg.inv(np.linalg.inv(F) + np.dot(self.A.T,np.dot(np.linalg.inv(G+self.E_h),self.A)))
				h = np.dot(H, (np.dot(np.linalg.inv(F),f) + np.dot(self.A.T,np.dot(np.linalg.inv(G+self.E_h),g))))
				h_list.append(h)
				H_list.append(H)
			else:
				H, h = F_list[t], f_list[t]
				h_list.append(h)
				H_list.append(H)
		return h_list, H_list

	#Inference-Smoothing smoothing works like correction for filtering
	def smoothing(self,y,h=0,H=0,h_list=None,H_list=None,f_list=None,F_list=None,count=0):
		f_list = [] if f_list==None else f_list
		F_list = [] if F_list==None else F_list
		h_list = [] if h_list==None else h_list
		H_list = [] if H_list==None else H_list
		if count == 0:
			f_list, F_list = self.filtering(y)
			h_new, H_new = f_list[-1], F_list[-1]
			h_list.append(h_new)
			H_list.append(H_new)
			count += 1
			self.smoothing(y,h_new,H_new,h_list,H_list,f_list, F_list,count)
			return h_list, H_list
		elif count < len(y):
			f = f_list[self.K-1-count]
			F = F_list[self.K-1-count]
			h_new, H_new = self.smooth_one_step(self.A,self.E_h,f,F,h,H,self.S,self.O)
			h_list.append(h_new)
			H_list.append(H_new)
			count = count + 1
			self.smoothing(y,h_new,H_new,h_list,H_list,f_list,F_list,count)
			return h_list, H_list
		else:
			return h_list.reverse(), H_list.reverse()

	#Parameter estimation with EM
	def EM(self,y,A_init=np.nan,B_init=np.nan,pi_m_init=np.nan,pi_s_init=np.nan,E_h_init=np.nan,E_o_init=np.nan,estimate=['A','B','pi_m','pi_s','E_h','E_o'],num_iterations=5):
		#parameters to be estimated
		A_est = True if 'A' in estimate else False
		B_est = True if 'B' in estimate else False
		pi_m_est = True if 'pi_m' in estimate else False
		pi_s_est = True if 'pi_s' in estimate else False
		E_h_est = True if 'E_h' in estimate else False
		E_o_est = True if 'E_o' in estimate else False
		#initialization
		if np.isnan(A_init):
			self.A = np.random.rand(self.S,self.S) if A_est == True	else self.A	
		if np.isnan(B_init):
			self.B = np.random.rand(self.O,self.S) if B_est == True else self.B
			#self.B = np.ones((self.O,self.S)) if B_est == True else self.B
		if np.isnan(pi_m_init):
			self.pi_m = np.random.rand(self.S) if pi_m_est == True else self.pi_m
		if np.isnan(pi_s_init):
			self.pi_s = np.eye(self.S) if pi_s_est == True else self.pi_s
		if np.isnan(E_h_init):
			self.E_h = np.eye(self.S) if E_h_est == True else self.E_h
		if np.isnan(E_o_init):
			self.E_o = np.eye(self.O) if E_o_est == True else self.E_o
		#EM iterations
		for _ in range(num_iterations):
			#E-step
			h_list, H_list = self.smoothing(y)
			f_list, F_list = self.filtering(y)
			#M-step
			self.pi_m = h_list[0] if pi_m_est == True else self.pi_m #pi_m update
			self.pi_s = H_list[0] if pi_s_est == True else self.pi_s #pi_s update
			#A update
			if A_est == True:
				sum1, sum2 = 0, 0
				for i in range(self.K-1):
					t = i + 1
					P2 = np.dot(self.A,np.dot(F_list[t-1],self.A.T)) + self.E_h
					P21 = np.dot(self.A,F_list[t-1])
					sum1 += np.dot(H_list[t].T, np.dot(np.linalg.inv(P2.T+np.eye(self.S)*10**-5),P21)) + np.dot(h_list[t], h_list[t-1].T)
					sum2 += H_list[t-1] + np.dot(h_list[t-1],h_list[t-1].T)
				self.A = np.dot(sum1, np.linalg.inv(sum2+np.eye(self.S)*10**-5))
			#B update
			sum1, sum2 = 0, 0
			for t in range(self.K):
				sum1 += np.dot(y[t].reshape(self.O,1), h_list[t].T)
				sum2 += H_list[t] + np.dot(h_list[t],h_list[t].T)
			self.B = np.dot(sum1, np.linalg.inv(sum2+np.eye(self.S)*10**-5)) if B_est == True else self.B
			#E_o update
			sum1 = np.dot(sum1, self.B.T)
			sum3 = 0
			if E_o_est == True:
				for t in range(self.K):
					sum3 += np.dot(y[t].reshape(self.O,1), y[t].reshape(1,self.O))
				self.E_o = (sum3 - sum1)/self.K
			#E_h update
			if E_h_est == True:
				sum2 = sum2 - H_list[0] - np.dot(h_list[0],h_list[0].T)
				sum1 = 0
				for i in range(self.K-1):
					t = i + 1
					P2 = np.dot(self.A,np.dot(F_list[t-1],self.A.T)) + self.E_h
					P21 = np.dot(self.A,F_list[t-1])
					sum1 += (np.dot(H_list[t].T, np.dot(np.linalg.inv(P2.T+np.eye(self.S)*10**-5),P21)) + np.dot(h_list[t], h_list[t-1].T)).T
				sum1 = np.dot(self.A, sum1)
				self.E_h = (sum2 - sum1)/(self.K-1)
		#Run and return the smoothing one more time after parameters are estimated
		h_list, H_list = self.smoothing(y)
		return self.A, self.B, self.pi_m, self.pi_s, self.E_h, self.E_o, h_list, H_list
