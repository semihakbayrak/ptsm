import numpy as np
import matplotlib.pyplot as plt
#add parent directory to path
import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
from utils import randgen, log_sum_exp, normalize, normalize_exp

class discrete_observation_HMM:
    def __init__(self,A,B,pi,S,K=0):
        self.A = A #Transition matrix
        self.B = B #Emission matrix
        self.pi = pi #prior state distributions for first hidden variable
        self.S = S #Number of states
        self.K = K #Number of time slices

    #Generate simulated data
    def generate_data(self,K):
        #K is the number of time slices to simulate
        self.K = K
        states = []
        observations = []
        for k in range(K):
            if k == 0:
                state = randgen(self.pi)
            else:
                state = randgen(self.A[:,states[k-1]])
            observation = randgen(self.B[:,state])
            states.append(state)
            observations.append(observation)
        return states,observations

    #Inference Forward pass
    def forward_pass(self,y):
        log_alpha = np.zeros((self.S,self.K))
        log_alpha_predict = np.zeros((self.S,self.K))
        for k in range(self.K):
            if k == 0:
                log_alpha_predict[:,k] = np.log(self.pi)
            else:
                log_alpha_predict[:,k] = self.state_predict(self.A,log_alpha[:,k-1])
            log_alpha[:,k] = self.state_update(self.B[y[k],:],log_alpha_predict[:,k])
        return log_alpha, log_alpha_predict

    #Inference Backward pass
    def backward_pass(self,y):
        log_beta = np.ones((self.S,self.K))
        log_beta_postdict = np.zeros((self.S,self.K))
        for k in range(self.K):
            t = self.K - 1 - k
            if t == (self.K - 1):
                pass
            else:
                log_beta_postdict[:,t] = self.state_postdict(self.A,log_beta[:,t+1])
            log_beta[:,t] = self.state_update(self.B[y[t],:],log_beta_postdict[:,t])
        return log_beta, log_beta_postdict

    @staticmethod
    def state_predict(A,log_p):
        return log_sum_exp(A,log_p)

    @staticmethod
    def state_update(b,log_p):
        return np.log(b).T + log_p

    @staticmethod
    def state_postdict(A,log_p):
        return log_sum_exp(A,log_p)

    #Inference Forward-Backward for smoothing
    def forward_backward(self,y):
        log_alpha, _ = self.forward_pass(y)
        _, log_beta_postdict = self.backward_pass(y)
        log_gamma = log_alpha + log_beta_postdict
        return log_gamma

    #Most probable path with Viterbi algorithm
    def viterbi(self,y):
        mp_path = [0]*self.K
        log_alpha, _ = self.forward_pass(y)
        for k in range(self.K):
            t = self.K - 1 - k
            if t == (self.K - 1):
                mp_path[t] = np.argmax(log_alpha[:,t])
            else:
                mp_path[t] = np.argmax(log_alpha[:,t]*(self.A[mp_path[t+1],:].T))
        return mp_path

    #Baum-Welch algorithm for parameter estimation
    def parameter_estimation_em(self,y_list, num_of_epochs=20):
        N = len(y_list) #number of observations
        #Initialization
        A_estimated = np.random.rand(self.S,self.S)
        A_estimated = A_estimated/np.sum(A_estimated,axis=0)
        B_estimated = np.random.rand(self.S,self.S)
        B_estimated = B_estimated/np.sum(B_estimated,axis=0)
        #B_estimated = np.eye(self.S) + 1e-5
        #B_estimated = B_estimated/np.sum(B_estimated,axis=0)
        pi_estimated = np.ones(self.S)/self.S
        self.A = A_estimated
        self.B = B_estimated
        self.pi = pi_estimated
        for epoch in range(num_of_epochs):
            gamma_dict, alpha_dict, beta_postdict_dict = {}, {}, {}
            #E-step
            for n in range(N):
                y = y_list[n]
                log_gamma = self.forward_backward(y)
                gamma = normalize_exp(log_gamma,axis=0)
                log_alpha, log_alpha_predict = self.forward_pass(y)
                alpha = normalize_exp(log_alpha,axis=0)
                log_beta, log_beta_postdict = self.backward_pass(y)
                beta = normalize_exp(log_beta,axis=0)
                beta_postdict = normalize_exp(log_beta_postdict,axis=0)
                gamma_dict[n] = gamma
                alpha_dict[n] = alpha
                beta_postdict_dict[n] = beta_postdict
            #M-step
            A_new = np.zeros((self.S,self.S))
            B_new = np.zeros((self.S,self.S))
            pi_estimated = 0
            for n in range(N):
                y = y_list[n]
                gamma = gamma_dict[n]
                alpha = alpha_dict[n]
                beta_postdict = beta_postdict_dict[n]
                pi_estimated = pi_estimated + gamma[:,0]
                for t in range(self.K):
                    if t != 0:
                        A_new = A_new + A_estimated*(B_estimated[y[t],:]*(alpha[:,t-1]*beta_postdict[:,t]).T)
                    B_new[y[t],:] = B_new[y[t],:] + gamma[:,t].T
            pi_estimated = pi_estimated/pi_estimated.sum()
            A_estimated = A_new/np.sum(A_new,axis=0)
            B_estimated = B_new/np.sum(B_new,axis=0)
            #A_estimated = normalize(A_new)
            #B_estimated = normalize(B_new)
            self.pi = pi_estimated
            self.A = A_estimated
            self.B = B_estimated
        return A_estimated, B_estimated, pi_estimated
