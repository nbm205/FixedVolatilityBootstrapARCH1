import numpy as np
from scipy.optimize import minimize_scalar
from multiprocessing import Pool
import multiprocessing
from scipy import stats
from scipy.stats import t as tt
import time 
import pandas as pd 


def generate(omega, alpha, x0, T):
    """Generate ARCH(1) data
    x(t) = s(t)z(t)
    sigma2(t) = 1 + ax(t-1)^2
    with x0 given. row of z contains T shocks.
    """
    z = np.random.normal(size=(T-1, ))
    # z = tt.rvs(8,size=T-1)*np.sqrt(((8-2)/8)) # t-distributed innovations
    x = np.zeros(T)
    x[0] = x0 # start value
    
    # loop over t 
    for t in range(1, T):
        x[t]=np.sqrt(omega+alpha*x[t-1]**2)*z[t-1] # we create T-1 innovations, so take t-1 index as z_t.
    return x, z

# --------------------------------------------------------------------------- # 

def ll(x, omega, alpha):
    """Log likelihood function"""
    T = x.shape[0] 
    l = np.sum(-0.5*(np.log(omega + alpha*x[0:T-1]**2)+x[1:T]**2/(omega + alpha*x[0:T-1]**2))) 
    return -l # We want to minimize, so take negative of log likelihood. 

def score(x, omega, alpha):
    """Score vector as given below"""
    T = x.shape[0]
    score = np.sum(-0.5*(1-x[1:T]**2/(omega + alpha*x[0:T-1]**2))*(x[0:T-1]**2/(omega + alpha*x[0:T-1]**2)))
    return -score # Notice the negative sign!

def hessian(x, omega, alpha):
    """Hessian vector as given below"""
    T = x.shape[0]
    hess = np.sum(0.5*(2*x[1:T]**2/(omega + alpha*x[0:T-1]**2)-1)*x[0:T-1]**4/((omega + alpha*x[0:T-1]**2)**2))
    return -hess # Notice the negative sign!

def optimize(x,omega):
    """Solver for non-linear MLE"""
    fll = lambda alpha: ll(x, omega,alpha) 
    alpha_hat = minimize_scalar(fll,bounds=(0.001,100), method='bounded')
    return alpha_hat.x

def LRtest(x,omega,alpha,alpha_bar):
    """Returns Likelihood Ratio test statistic"""
    unrestricted_ll = -ll(x,omega,alpha)
    restricted_ll = -ll(x,omega,alpha_bar)
    return 2*(unrestricted_ll-restricted_ll)

# --------------------------------------------------------------------------- # 


def sigma2_estimate(x, omega_est, alpha_est):
    """Computes estimated sigma2 which is needed for bootstrap DGP"""
    sigma2_est = omega_est + alpha_est*x**2
    sigma2_est = np.concatenate(([1],sigma2_est))
    sigma2_est = np.delete(sigma2_est,-1)
    return sigma2_est

def generate_bootstrap(x, sigma2_est):
    """Generate Bootstrap ARCH(1) data from original data
    based on i.i.d. draws from standardized residuals
    """
    T = len(x)
    x_star = np.zeros(T)
    sigma_est = np.sqrt(sigma2_est)
    ## Generate standardize residuals
    z_est = x/sigma_est
    z_bar = np.mean(z_est)
    z_s = (z_est-z_bar)/(np.sqrt(1/T*np.sum((z_est-z_bar)**2)))
    # Compute bootstrap x_star
    x_star = sigma_est*np.random.choice(z_s,T)
    x_star[0] = x[0]
    return x_star, z_s


# --------------------------------------------------------------------------- # 

def bootstrap_ll(x, x_star, omega, alpha):
    """Bootstrap log likelihood function"""
    T = x.shape[0] 
    l = np.sum(-0.5*(np.log(omega + alpha*x[0:T-1]**2)+x_star[1:T]**2/(omega + alpha*x[0:T-1]**2))) 
    return -l # We want to minimize, so take negative of log likelihood. 

def bootstrap_score(x, x_star, omega, alpha):
    """Score vector as given below"""
    T = x.shape[0]
    score = np.sum(-0.5*(1-x_star[1:T]**2/(omega + alpha*x[0:T-1]**2))*(x[0:T-1]**2/(omega + alpha*x[0:T-1]**2)))
    return -score # Notice the negative sign!

def bootstrap_hessian(x, x_star, omega, alpha):
    """Hessian vector as given below"""
    T = x.shape[0]
    hess = np.sum(0.5*(2*x_star[1:T]**2/(omega + alpha*x[0:T-1]**2)-1)*x[0:T-1]**4/((omega + alpha*x[0:T-1]**2)**2))
    return -hess # Notice the negative sign!


def bootstrap_optimize(x,x_star,omega):
    """Solver for non-linear Bootstrap MLE"""
    fll_star = lambda alpha: bootstrap_ll(x,x_star, omega,alpha) 
    alpha_hat_star = minimize_scalar(fll_star,bounds=(0.001,100), method='bounded')
    return alpha_hat_star.x

def bootstrap_LRtest(x,x_star,omega,alpha,alpha_bar):
    unrestricted_ll = -bootstrap_ll(x,x_star,omega,alpha)
    restricted_ll = -bootstrap_ll(x,x_star,omega,alpha_bar)
    return 2*(unrestricted_ll-restricted_ll)

def bootstrap_p_value(LRtest,Bootstrap_LRtest,B):
    bol_array = LRtest<=Bootstrap_LRtest
    return np.sum(bol_array)/B

def ERF(p_val_array,N):
    bol_array = p_val_array<=0.05
    return np.sum(bol_array)/N


np.random.seed(1531)
def bootstrap_simulate(x, sigma2_est, B, alpha_bar, omega=1):
    LR_array = np.zeros(B)
    for b in range(0,B):
        x_star, z_s = generate_bootstrap(x,sigma2_est)
        alpha_hat_star = bootstrap_optimize(x,x_star,omega)
        LR_test = bootstrap_LRtest(x,x_star,omega,alpha_hat_star,alpha_bar)
        LR_array[b] = LR_test
    return LR_array


# --------------------------------------------------------------------------- # 


omega=1 # Omega parameter
alpha=1.5 # ARCH parameter
alpha_bar = 1.5 # Null hypothesis
x_0=0 # start value - mean zero
N=100 # Replications - takes approx 2 hours when N=10000
B=399 # Bootstrap Simulations
T_values = [10, 25, 50, 200, 1000]

def montecarlo(T):
    print(f"Running MC experiment with T = {T}")
    p_val_array_classic = np.zeros(N,dtype=float)
    p_val_array_iid = np.zeros(N,dtype=float)
    LR_stat_array = np.zeros(N,dtype=float)
    for n in range(0, N):
        # Classic
        x, z = generate(omega,alpha,x_0,T)
        alpha_hat = optimize(x,omega)
        LR_test = LRtest(x,omega,alpha_hat,alpha_bar)
        p_val_array_classic[n] = 1-stats.chi2.cdf(LR_test,1)
        LR_stat_array[n] = LR_test
        # Compute sigma2_est
        sigma2_est = sigma2_estimate(x,omega,alpha_hat)
        # # IID Bootstrap
        LR_array_iid = bootstrap_simulate(x,sigma2_est,B,alpha_hat)
        p_val_array_iid[n] = bootstrap_p_value(LR_test,LR_array_iid,B)
    erf_classic = ERF(p_val_array_classic,N) 
    erf_iid = ERF(p_val_array_iid,N)
    return erf_classic, erf_iid 


def timer(func): 
    """Timer decorator"""
    def wrapper(*args, **kwargs):
        st = time.time()
        output = func(*args, **kwargs)
        et = time.time() 
        print(f"Time elapsed: {et - st:.2f} \nwith Ts = {T_values}")
        return output 
    return wrapper 

@timer 
def main():
    num_cores = multiprocessing.cpu_count()  # #cores on your computer 
    with Pool(num_cores) as p:  # Run in parallel 
        output_tuples = p.map(montecarlo, T_values)
    df = pd.DataFrame(output_tuples, columns=["classic", "iid"], index=T_values)
    print("Results:", df, sep="\n")
    df.to_csv("results_sim.csv")  # Save results to csv in folder 

# --------------------------------------------------------------------------- # 

if __name__ == '__main__':
    main()
