import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
#add parent directory to path
import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

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

    #Inference-Smoothing
    def smoothing(self,y,g=0,G=0,f_list=[],F_list=[],g_list=[],G_list=[],count=0):
        if count == 0:
            f_list, F_list = self.filtering(y)
            g_new = f_list[-1]
            G_new = F_list[-1]
            g_list.append(g_new)
            G_list.append(G_new)
            count += 1
            self.smoothing(y,g_new,G_new,f_list,F_list,g_list,G_list,count)
            return g_list, G_list
        elif count < len(y):
            f = f_list[len(y)-count-1]
            F = F_list[len(y)-count-1]
            m_h = np.dot(self.A,f)
            E_hf_hf = np.dot(self.A,np.dot(F,self.A.T)) + self.E_h
            E_hf_h = np.dot(self.A,F)
            E_h_hf = E_hf_h.T
            E_n = F - np.dot(np.dot(E_h_hf,np.linalg.inv(E_hf_hf)),E_hf_h)
            A_n = np.dot(E_h_hf,np.linalg.inv(E_hf_hf))
            m_n = f - np.dot(A_n,m_h)
            g_new = np.dot(A_n,g) + m_n
            G_new = np.dot(A_n,np.dot(G,A_n.T)) + E_n
            g_list.append(g_new)
            G_list.append(G_new)
            count = count + 1
            self.smoothing(y,g_new,G_new,f_list,F_list,g_list,G_list,count)
            return g_list, G_list
        else:
            return g_list.reverse(), G_list.reverse()
