# Simulation of CRC memory with given parameters
import logging
import argparse
import numpy as np
import nlopt

ii = 1j

# number of cavity
NC = 6
Q_factor = 3*10**6;

logging.basicConfig(filename='memory-optimization.log', filemode='a',format='%(asctime)s - %(message)s',level=logging.INFO)




def Um(omega,kappa,gamma,Delta):
    # Transfer function
    U = np.zeros((kappa.size,omega.size),dtype=np.complex64)
    for i in  np.arange(kappa.size):
        U[i]=(kappa[i]/2-gamma[i] - ii*(Delta[i] - omega))/(kappa[i]/2 + gamma[i] + ii*(Delta[i]-omega))
    return np.prod(U,axis=0)


def Aout(t,omega,domega,kappa,gamma,Delta):
    # Output field 
    return np.trapz(np.exp(-ii*omega*t)*Ain(omega,domega)*Um(omega,kappa,gamma,Delta), omega)/np.sqrt(2*np.pi)

def Ain(omega,domega):
    # Input spectrum
    # For example normalized Gaussian spectrum is considered
    return np.sqrt(np.exp(-omega**2/(2*domega**2))/(domega*np.sqrt(2*np.pi)))



def spectrum_eff(tau, omega, domega, kappas,gammas,Deltas):
    return np.trapz(np.exp(-ii*omega*tau)*(np.abs(Ain(omega,domega))**2)*Um(omega,kappas,gammas,Deltas), omega)

def eff_time(domega, number_cavity, gammas, Deltas, kappas):
    omega = np.linspace(-10,10,400)
    time = np.linspace(-3,20,400)
    it = np.arange(time.size)
    dt = time[1]-time[0]
    # Array for storing field dynamics
    Atout = np.zeros(time.size,np.complex64)
    
    # Calculate input and output fields
    for i in it:
        Atout[i] = Aout(t=time[i], omega=omega,domega = domega, kappa=kappas, gamma=gammas, Delta=Deltas)
        
    time_ind= np.argmax(np.abs(Atout)**2)
    tau = time[time_ind]
    eff = spectrum_eff(tau, omega, domega, kappas,gammas,Deltas)
    logging.info(f'kappas: {kappas}')
    logging.info(f'Input Gauissan pulse duration: {1/domega:2f} ns')
    logging.info(f'Delta: {Deltas}')
    logging.info(f'Eff: {np.real(eff):2f}')
    logging.info(f'Storage time: {tau:2f} ns')
    
    return tau*eff   



def memory_optimization(parameters, grad):
    wvl = 1550*10**(-9);
    c = 3*10**8;
    freq = c/wvl
    decay_rate = freq/Q_factor*10**(-9)
    [domega, band] = parameters[0:2]
    kappas = parameters[2:]
    Deltas = np.linspace(start=-band/2, stop=band/2, num=NC)
    gammas = decay_rate*np.ones(NC)
    
    et = eff_time(domega, number_cavity, gammas, Deltas, kappas)
    
    return np.real(et)


def optimize_memory(number_cavity, max_eval):
    # The parameters to optimize:
    # band
    # domega
    # g
    NC = number_cavity
    logging.info('The optimization procedure for memory has started')
    
    band_0 = 6
    band_max = 8
    band_min = 2
    domega_0 = 2
    domega_max = 8
    domega_min = 0.5
    
    
    Deltas0 = np.linspace(start=-band_0/2, stop=band_0/2, num=number_cavity)
    Delta = Deltas0[1]-Deltas0[0]
    
    kappas0 = np.ones(number_cavity)*Delta/2
    kappas_min = np.ones(number_cavity)*Delta*0.2
    kappas_max = np.ones(number_cavity)*Delta*10
    
    par0=np.zeros(number_cavity+2)
    par0[0:2] = [band_0,domega_0]
    par0[2:] = kappas0
    
    
    par_min=np.zeros(number_cavity+2)
    par_min[0:2] = [band_0,domega_0]
    par_min[2:] = kappas_min
    
    
    par_max=np.zeros(number_cavity+2)
    par_max[0:2] = [band_max,domega_max]
    par_max[2:] = kappas_max
    
    
    
    opt = nlopt.opt(nlopt.GN_ESCH, number_cavity+2) #LN_COBYLA #LN_BOBYQA GN_ISRES
    opt.set_max_objective(memory_optimization)
    initialVector = (par_max-par_min)*np.random.rand(number_cavity+2)+par_min
    optmizeVectorMin = par_min
    optmizeVectorMax =  par_max

    opt.set_lower_bounds(optmizeVectorMin)
    opt.set_upper_bounds(optmizeVectorMax)
    opt.set_ftol_rel(0.005)
    opt.set_xtol_rel(0.005)
    opt.max_eval = max_eval
    
    x = opt.optimize(initialVector)
    maxf = opt.last_optimum_value()
    
    return maxf


def main(args):
    number_cavity = args.nc
    max_eval = args.me
    max_eff = optimize_memory(number_cavity, max_eval)
    print(f'maximum storage by effeciency product is {max_eff}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', type=int, default=6, help='Number of cavities')
    parser.add_argument('-me', type=int, default=10, help='Maximum number of evaluation')
    args = parser.parse_args()
    main(args)
