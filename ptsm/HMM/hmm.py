#HMM is mostly adapted from Ali Taylan Cemgil's notes and codes
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from ..utils import randgen, log_sum_exp, normalize_exp

class discrete_observation_HMM:
    def __init__(self,A,B,pi,S,O,K=0):
        self.A = A #Transition matrix
        self.B = B #Emission matrix
        self.pi = pi #prior state distributions for first hidden variable
        self.S = S #Number of possible states
        self.O = O #Number of possible observations
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
    def parameter_estimation_em(self,y, num_of_epochs=20):
        #Initialization
        A_estimated = np.random.rand(self.S,self.S)
        A_estimated = A_estimated/np.sum(A_estimated,axis=0)
        B_estimated = np.random.rand(self.O,self.S)
        B_estimated = B_estimated/np.sum(B_estimated,axis=0)
        pi_estimated = np.ones(self.S)/self.S
        self.A = A_estimated
        self.B = B_estimated
        self.pi = pi_estimated
        for epoch in range(num_of_epochs):
            #E-step
            log_gamma = self.forward_backward(y)
            gamma = normalize_exp(log_gamma)
            log_alpha, log_alpha_predict = self.forward_pass(y)
            alpha = normalize_exp(log_alpha)
            log_beta, log_beta_postdict = self.backward_pass(y)
            beta = normalize_exp(log_beta)
            beta_postdict = normalize_exp(log_beta_postdict)
            #M-step
            pi_estimated = gamma[:,0]
            self.pi = pi_estimated

            A_count = np.zeros((self.S,self.S))
            for t in range(self.K):
                if t != 0:
                    A_new = A_estimated*(B_estimated[y[t],:].reshape(1,self.S).T)*(alpha[:,t-1].reshape(self.S,1).T)*beta_postdict[:,t]
                    A_new = A_new/A_new.sum()
                    A_count = A_count + A_new
            A_estimated = A_count/(np.sum(gamma[:,0:-1],axis=1).reshape(self.S,1))
            A_estimated = A_estimated/np.sum(A_estimated,axis=0)
            self.A = A_estimated

            B_count = np.zeros((self.O,self.S))
            for t in range(self.K):
                B_count[y[t],:] = B_count[y[t],:] + gamma[:,t].reshape(self.S,1).T
            B_estimated = B_count/np.sum(B_count,axis=0)
            self.B = B_estimated
        return A_estimated, B_estimated, pi_estimated

class continuous_observation_HMM:
    def __init__(self,A,C,R,pi,S,O,K=0):
        self.A = A #Transition matrix
        self.C = C #Matrix of means(O-by-S dimensional)
        self.R = R #Observation noise matrix(O-by-O dimensional)
        self.pi = pi #prior state distributions for first hidden variable
        self.S = S #Number of possible states
        self.O = O #Dimensionality of observations
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
            if self.O == 1:
                mu = self.C[state]
                observation = np.random.normal(mu,self.R)
            else:
                mu = self.C[:,state]
                observation = np.random.multivariate_normal(mu,self.R)
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
            log_alpha[:,k] = self.state_update(y[k],log_alpha_predict[:,k],self.C,self.R,self.S,self.O)
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
            log_beta[:,t] = self.state_update(y[t],log_beta_postdict[:,t],self.C,self.R,self.S,self.O)
        return log_beta, log_beta_postdict

    @staticmethod
    def state_predict(A,log_p):
        return log_sum_exp(A,log_p)

    @staticmethod
    def state_update(y_t,log_p,C,R,S,O):
        b = []
        for s in range(S):
            if O == 1:
                pdf = scipy.stats.norm(C[s], R).pdf(y_t)
            else:
                pdf = scipy.stats.multivariate_normal(C[:,s], R).pdf(y_t)
            b.append(pdf)
        b = np.array(b)
        return np.log(b) + log_p

    @staticmethod
    def state_postdict(A,log_p):
        return log_sum_exp(A,log_p)

    @staticmethod
    def probs_vector(y_t,C,R,S,O):
        b = []
        for s in range(S):
            if O == 1:
                pdf = scipy.stats.norm(C[s], R).pdf(y_t)
            else:
                pdf = scipy.stats.multivariate_normal(C[:,s], R).pdf(y_t)
            b.append(pdf)
        b = np.array(b)
        return b

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
    def parameter_estimation_em(self,y, num_of_epochs=20):
        if self.O == 1:
            #Initialization
            A_estimated = np.random.rand(self.S,self.S)
            A_estimated = A_estimated/np.sum(A_estimated,axis=0)
            C_estimated = []
            R_estimated = 1.
            y_max = np.max(y)
            y_min = np.min(y)
            gap = (y_max-y_min)/(self.S-1.)
            for s in range(self.S):
                C_estimated.append(y_min+gap*s)
            C_estimated = np.array(C_estimated)
            pi_estimated = np.ones(self.S)/self.S
            self.A = A_estimated
            self.C = C_estimated
            self.R = R_estimated
            self.pi = pi_estimated
            for epoch in range(num_of_epochs):
                #E-step
                log_gamma = self.forward_backward(y)
                gamma = normalize_exp(log_gamma)
                log_alpha, log_alpha_predict = self.forward_pass(y)
                alpha = normalize_exp(log_alpha)
                log_beta, log_beta_postdict = self.backward_pass(y)
                beta = normalize_exp(log_beta)
                beta_postdict = normalize_exp(log_beta_postdict)
                #M-step
                pi_estimated = gamma[:,0]
                self.pi = pi_estimated

                A_count = np.zeros((self.S,self.S))
                for t in range(self.K):
                    if t != 0:
                        b = self.probs_vector(y[t],self.C,self.R,self.S,self.O)
                        A_new = A_estimated*(b.reshape(1,self.S).T)*(alpha[:,t-1].reshape(self.S,1).T)*beta_postdict[:,t]
                        A_new = A_new/A_new.sum()
                        A_count = A_count + A_new
                A_estimated = A_count/(np.sum(gamma[:,0:-1],axis=1).reshape(self.S,1))
                A_estimated = A_estimated/np.sum(A_estimated,axis=0)
                self.A = A_estimated

                C_count = np.ones(self.S) - 1
                for t in range(self.K):
                    C_count = C_count + gamma[:,t]*y[t]
                C_estimated = C_count/np.sum(gamma,axis=1)
                self.C = C_estimated

                R_count = 0
                for t in range(self.K):
                    R_count = R_count + np.sum(gamma[:,t]*(y[t]-C_estimated)**2)
                R_estimated = R_count/self.K
                self.R = R_estimated
        else:
            #Initialization
            A_estimated = np.random.rand(self.S,self.S)
            A_estimated = A_estimated/np.sum(A_estimated,axis=0)
            C_estimated = np.random.rand(self.O,self.S)
            R_estimated = np.eye(self.O)
            '''
            R_estimated = np.eye(self.O)
            y_max = np.max(y)
            y_min = np.min(y)
            gap = (y_max-y_min)/(self.S-1.)
            for s in range(self.S):
                C_estimated.append(y_min+gap*s)
            C_estimated = np.array(C_estimated)
            '''
            pi_estimated = np.ones(self.S)/self.S
            self.A = A_estimated
            self.C = C_estimated
            self.R = R_estimated
            self.pi = pi_estimated
            for epoch in range(num_of_epochs):
                #E-step
                log_gamma = self.forward_backward(y)
                gamma = normalize_exp(log_gamma)
                log_alpha, log_alpha_predict = self.forward_pass(y)
                alpha = normalize_exp(log_alpha)
                log_beta, log_beta_postdict = self.backward_pass(y)
                beta = normalize_exp(log_beta)
                beta_postdict = normalize_exp(log_beta_postdict)
                #M-step
                pi_estimated = gamma[:,0]
                self.pi = pi_estimated

                A_count = np.zeros((self.S,self.S))
                for t in range(self.K):
                    if t != 0:
                        b = self.probs_vector(y[t],self.C,self.R,self.S,self.O)
                        A_new = A_estimated*(b.reshape(1,self.S).T)*(alpha[:,t-1].reshape(self.S,1).T)*beta_postdict[:,t]
                        A_new = A_new/A_new.sum()
                        A_count = A_count + A_new
                A_estimated = A_count/(np.sum(gamma[:,0:-1],axis=1).reshape(self.S,1))
                A_estimated = A_estimated/np.sum(A_estimated,axis=0)
                self.A = A_estimated

                C_count = np.zeros((self.O,self.S))
                for t in range(self.K):
                    C_count = C_count + np.outer(y[t],gamma[:,t])
                C_estimated = C_count/np.sum(gamma,axis=1)
                self.C = C_estimated

                R_count = 0
                for t in range(self.K):
                    R_new = np.dot(gamma[:,t].reshape(1,self.S)*(y[t].reshape(self.O,1)-C_estimated),(y[t].reshape(self.O,1)-C_estimated).T)
                    R_count = R_count + R_new
                R_estimated = R_count/self.K
                self.R = R_estimated

        return A_estimated, C_estimated, R_estimated, pi_estimated
