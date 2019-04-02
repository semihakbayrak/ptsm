import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
#add parent directory to path
import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

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

    #Inference-Filtering
    def filtering(self,y,f=0,F=0,f_list=[],F_list=[],count=0):
        if type(f) == int:
            if f == 0:
                if self.S>1:
                    f=np.zeros(self.S).reshape(self.S,1)
        if type(F) == int:
            if F == 0:
                if self.S>1:
                    np.zeros((self.S,self.S))
        m_h = np.dot(self.A,f)
        m_o = np.dot(self.B,m_h)
        E_hh = np.dot(self.A,np.dot(F,self.A.T)) + self.E_h
        E_oo = np.dot(self.B,np.dot(E_hh,self.B.T)) + self.E_o
        E_oh = np.dot(self.B,E_hh)
        E_ho = E_oh.T
        f_new = m_h + np.dot(np.dot(E_ho,np.linalg.inv(E_oo)),(y[count].reshape(self.O,1)-m_o))
        F_new = E_hh - np.dot(np.dot(E_ho,np.linalg.inv(E_oo)),E_oh)
        f_list.append(f_new)
        F_list.append(F_new)
        count = count + 1
        if count < len(y):
            self.filtering(y,f_new,F_new,f_list,F_list,count)
            return f_list, F_list
        else:
            return f_list, F_list

    #Inference-Backward
    def backward(self,y,g=0,G=0,g_list=[],G_list=[],count=0):
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

    #Inference-Smoothing
    def smoothing(self,y):
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
