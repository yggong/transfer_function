# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 20:15:41 2019
use the saved data to plot graphs
@author: Gong and Gao
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sici

# The full analytical formulas for the transfer functions

def Get_Rut(u):
    gamma = np.pi/3.
    rut = (  0.5-2./u**2. + (-1./6.+2./u**2.)*np.cos(gamma) + 0.25/u**2.*(1.+1./np.sin(gamma/2.)**2.)*np.cos(2.*u*np.sin(gamma/2.))    
        + ((-3.+np.cos(gamma))/u**3.+(-21.+28.*np.cos(gamma)-7.*np.cos(2.*gamma))/u)/16./np.sin(gamma/2.)**3.*np.sin(2.*u*np.sin(gamma/2.)) 
        + 4.*np.sin(gamma/2.)**2.*(sici(2.*u*np.sin(gamma/2.))[1]-sici(2.*u)[1]-np.log(np.sin(gamma/2.))) 
        + ((2.-2.*np.cos(gamma))/u**3.+(1.-np.cos(gamma))/u + 2.*np.cos(gamma/2.)**2.*(2.*sici(2.*u)[0]
            -sici(2.*u*(1+np.sin(gamma/2.)))[0]-sici(2.*u*(1.-np.sin(gamma/2.)))[0]))*np.sin(2.*u)
        + ((1.-np.cos(gamma))/6.+(-2.+2.*np.cos(gamma))/u**2. + 2.*np.cos(gamma/2.)**2.*(2.*sici(2.*u)[1]
            -sici(2.*u*(1.+np.sin(gamma/2.)))[1]-sici(2.*u*(1.-np.sin(gamma/2.)))[1]+2.*np.log(np.cos(gamma/2.))))*np.cos(2.*u)   
        )/u**2./4.
    return 2.*rut # R_tensor = 2* R_+

def Get_Rub(u):
    gamma = np.pi/3.
    cg = np.cos(gamma)
    sg2 = np.sin(gamma/2.)**2.
    rub = ( 1./6.*(3.-cg)+2.*sg2*np.sin(2.*u)/u*(2./u**2.-1.)-2./u**2.*(1.-cg)+sg2*np.cos(2.*u)*(1./3.-4./u**2.) 
          +(3.-cg)/sg2*np.cos(2.*u*np.sin(gamma/2.))/8./u**2+np.sin(2.*u*np.sin(gamma/2.))/16./u/np.sin(gamma/2.)**3 
              *(3.+np.cos(2.*gamma)-4.*cg+(cg-3.)/u**2.) )/u**2./2.
    return rub

def Get_Ruv(u):
    gamma = np.pi/3.
    cg2 = np.cos(gamma/2.)
    sg2 = np.sin(gamma/2.)
    ruv = ( np.cos(2.*u)*( -2.*np.log(cg2)-2.*sici(2.*u)[1]+sici(2.*u*(1.-sg2))[1]+sici(2.*u*(1.+sg2))[1]
           +8.*sg2**2.*(1./u**2.-1./3.))+(1.+sg2**2.)*(1.+4.*u**2.*sg2**2.)*np.sin(2.*u*sg2)/4./u**3./sg2**3.
           +np.sin(2.*u)*(sici(2.*u*(1.-sg2))[0]+sici(2.*u*(1.+sg2))[0]-2.*sici(2.*u)[0]+4.*sg2**2.*(1.-2./u**2.)/u) 
            -8.*(1./3.+sg2**2.*(1./3.-1./u**2.))-(1.+1./sg2**2.)*np.cos(2.*u*sg2)/u**2./2.  
            +2.*(np.log(2.*u*sg2)+np.euler_gamma-sici(2.*u*sg2)[1]) )/2./u**2. 
    return 2.*ruv

def Get_Rul(u):
    gamma = np.pi/3.
    cg2 = np.cos(gamma/2.)
    sg2 = np.sin(gamma/2.)
    sg=np.sin(gamma)
    cg=np.cos(gamma)
    rul=1./(96.*u**5)*(4.*u**3*(39. + 17.*np.cos(2.*u) - 28.*np.cos(u)**2.*cg) +
                     3.*sg2**(-3.)*(u*(-40. + 7.*np.cos(2.*u*sg2))*sg2 - 
        64.*(u*np.cos(2.*u) + (-1. + u**2)*np.sin(2.*u))*sg2**5 - 
       u*(-20. + np.cos(2.*u*sg2))*np.sin(3.*gamma/2.) - 4.*u*np.sin(5.*gamma/2.) 
       + (-3. - 5.* u**2 + cg + 4.* u**2 *cg + u**2 *np.cos(2.*gamma))*np.sin(2.*u*sg2))
        )+ 1./(8. *u**2)* ((2.*u + (3.- 2.* cg + cg2**(-2.))* np.sin(2.* u))* sici(2.* u)[0] + 
          cg2**(-2) *(-np.sin(2.* u)* (sici(2.* u *(1. + sg2))[0] + sici(2.* u - 2.* u* sg2)[0]) + 
          2./(-1 + cg)*cg**3 *np.sin(u-u *cg)*(
               -sici(u-u*cg)[0]+sici(u + u* cg)[0] + 
           sici(u-u*cg-2.*u*sg2)[0] + sici(u-u*cg+2.*u*sg2)[0]))) + 1/(8.* u**2)*(
                4.* cg*(cg/sg)**2*np.cos(u-u* cg)*(
             sici(u- u*cg)[1]+sici(u + u*cg)[1]-sici(u*(-1+cg+2.*sg2))[1]-sici(u-u* cg + 2.*u*sg2)[1]) 
          + sg2**(-2.)*np.cos(2.*u)*((sici(2.*u*(1-sg2))[1]-np.log(2*u*(1-sg2))-np.euler_gamma) + (
          sici(2.*u*(1+sg2))[1] - np.log(2.*u*(1+sg2))- np.euler_gamma)) + 1./2.*(-3. + 
          np.cos(2.*gamma))* (2. - 2.* np.cos(2.* u) + 4.* np.cos(u)**2 *cg)
          *sg**(-2)* (sici(2.* u)[1] - np.log(2.* u) - np.euler_gamma) + 
          8.*sg**(-2)* (sici(2.*u*sg2)[1]-np.log(2.*u*sg2)- np.euler_gamma) - 
          4.*sg**(-2)* np.cos(2.* u)*((sici(2.*u*(1.+sg2))[1]-np.log(2.*u*(1.+sg2))- 
         np.euler_gamma) + (sici(2.*u*(1.-sg2))[1]-np.log(2.*u*(1.-sg2))-np.euler_gamma)) 
            + (9.+np.cos(2.*u))*(sici(2.*u)[1]-np.log(2.*u)-np.euler_gamma)- 
           2.*cg2**(-2)*(sici(2.*u*sg2)[1]-np.log(2.*u*sg2) - np.euler_gamma))

          
    return rul
              
# The approximate formulas for the transfer functions

def get_rta(u):
    g=pi/3.
    rta = (2.*np.sin(g)**2.)/5./(1.+(2.*u**2.*np.sin(g)**2.)/(5.*(1./4. - np.cos(g)/12.-
                           2.*np.sin(g/2.)**2.*np.log(np.sin(g/2.)))+5.*np.cos(2.*u)*(np.sin(g/2.)**2./6.+
                                np.cos(g/2.)**2.*np.log(np.cos(g/2.)**2.))))
    return rta

def get_rta1(u):
    return 0.3/(1.+0.6*u**2.)


def get_rva(u):
    g=pi/3.
    rva = (2.*np.sin(g)**2.)/5./(1.+0.4*np.sin(g)**2.*u**2./((2.*(np.log(3./u+2.*u*np.sin(g/2.))-2.+2.*np.cos(g)/3.+np.euler_gamma))-
                     np.cos(2.*u)*(8.*np.sin(g/2.)**2./3.+np.log(np.cos(g/2.)**2.))))
    return rva

def get_rba(u): 
    g=pi/3.
    rba = (np.sin(g)**2.)/15./(1.+
          0.8*np.sin(g)**2.*u**2./(3.-np.cos(g)+(1.-np.cos(g))*np.cos(2.*u)))
    return rba

def get_rla(u): 
    g=pi/3.
    rll = (np.sin(g)**2.)/15.
    rlh1 = 2.*pi*u + 1.-np.log(2.*u)*(3.+2./np.sin(g/2.)**2.-4.*np.cos(g)+
                               (np.cos(g)-np.cos(2.*g))*(np.cos(2.*u))/np.cos(g/2.)**2.)
    rla = rlh1/(16.*u**2.+rlh1/rll)
    return rla

pi=np.pi

#Plot the transfer functions

x=np.arange(-3.,2.,0.01)
x=10.**x
tqrt=Get_Rut(x)
tqrta=get_rta(x)
tqrta1=get_rta1(x)
tqrv=Get_Ruv(x)
tqrva=get_rva(x)
tqrb=Get_Rub(x)
tqrba=get_rba(x)
tqrl=Get_Rul(x)
tqrla=get_rla(x)


#plot R_A(u)
fig=plt.figure(figsize=(8,6))
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('font', family='calibri')

ax1 = fig.add_subplot(221)
y_tick_locations=np.array([.0001,.001, 0.01, 0.1, 1])
x_tick_locations=np.array([.001, .01, .1, 1, 10.,100.])
plt.title('Tensor mode')

ax1.set_xscale('log')

ax1.loglog(x,tqrt,color='k',label=r'$Analytical$')
ax1.loglog(x,tqrta,":r",label=r'$Approximation$')
ax1.loglog(x,tqrta1,"b",label=r'$Approximation A$')

ax1.set_xticks(x_tick_locations)
ax1.set_yticks(y_tick_locations)
ax1.set_xlabel(r'$u=2\pi fL/c$', labelpad=5,fontsize=15)
ax1.set_ylabel(r'$R^T_{MI}$', labelpad=5,fontsize=15)
#ax1.tick_params(axis='both', which='minor', labelsize=15) 
ax1.legend()

ax1 = fig.add_subplot(222)
y_tick_locations=np.array([.0001,.001, 0.01, 0.1, 1])
x_tick_locations=np.array([.001, .01, .1, 1, 10.,100.])
ax1.set_xscale('log')
plt.title('Vector mode')

ax1.loglog(x,tqrv,color='k',label=r'$Analytical$')
ax1.loglog(x,tqrva,":r",label=r'$Approximation$')


ax1.set_xticks(x_tick_locations)
ax1.set_yticks(y_tick_locations)
ax1.set_xlabel(r'$u=2\pi fL/c$', labelpad=5,fontsize=15)
ax1.set_ylabel(r'$R^V_{MI}$', labelpad=5,fontsize=15)
#ax1.tick_params(axis='both', which='minor', labelsize=15) 
ax1.legend()

ax1 = fig.add_subplot(223)
y_tick_locations=np.array([.0001,.001, 0.01, 0.1])
x_tick_locations=np.array([.001, .01, .1, 1, 10.,100.])
ax1.set_xscale('log')
plt.title('Breathing mode')

ax1.loglog(x,tqrb,color='k',label=r'$Analytical$')
ax1.loglog(x,tqrba,":r",label=r'$Approximation$')


ax1.set_xticks(x_tick_locations)
ax1.set_yticks(y_tick_locations)
ax1.set_xlabel(r'$u=2\pi fL/c$', labelpad=5,fontsize=15)
ax1.set_ylabel(r'$R^b_{MI}$', labelpad=5,fontsize=15)
#ax1.tick_params(axis='both', which='minor', labelsize=15) 
ax1.legend()

ax1 = fig.add_subplot(224)
y_tick_locations=np.array([.001, 0.01, 0.1])
x_tick_locations=np.array([.001, .01, .1, 1, 10.,100.])
ax1.set_xscale('log')
plt.title('Longitudinal mode')

ax1.loglog(x,tqrl,color='k',label=r'$Analytical$')
ax1.loglog(x,tqrla,":r",label=r'$Approximation$')


ax1.set_xticks(x_tick_locations)
ax1.set_yticks(y_tick_locations)
ax1.set_xlabel(r'$u=2\pi fL/c$', labelpad=5,fontsize=15)
ax1.set_ylabel(r'$R^l_{MI}$', labelpad=5,fontsize=15)
#ax1.tick_params(axis='both', which='minor', labelsize=15) 
ax1.legend()

plt.tight_layout()
plt.subplots_adjust(wspace=0.5,hspace=0.8)  
plt.savefig('MIfig.jpeg',dpi=300)
plt.savefig('MIfig.pdf')
plt.show()



#plot TDI R_A(u)
fig=plt.figure(figsize=(8,6))
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('font', family='calibri')

ax1 = fig.add_subplot(221)
y_tick_locations=np.array([.0001,.001, 0.01, 0.1, 1])
x_tick_locations=np.array([.001, .01, .1, 1, 10.,100.])
plt.title('Tensor mode')

ax1.set_xscale('log')

ax1.loglog(x,tqrt*16.*np.sin(x)**2.,color='k',label=r'$Analytical$')
ax1.loglog(x,tqrta*16.*np.sin(x)**2.,":r",label=r'$Approximation$')

ax1.set_xticks(x_tick_locations)
ax1.set_yticks(y_tick_locations)
ax1.set_xlabel(r'$u=2\pi fL/c$', labelpad=5,fontsize=15)
ax1.set_ylabel(r'$R^T_{MC}$', labelpad=5,fontsize=15)
#ax1.tick_params(axis='both', which='minor', labelsize=15) 
ax1.legend()

ax1 = fig.add_subplot(222)
y_tick_locations=np.array([.0001,.001, 0.01, 0.1, 1])
x_tick_locations=np.array([.001, .01, .1, 1, 10.,100.])
ax1.set_xscale('log')
plt.title('Vector mode')

ax1.loglog(x,tqrv*16.*np.sin(x)**2.,color='k',label=r'$Analytical$')
ax1.loglog(x,tqrva*16.*np.sin(x)**2.,":r",label=r'$Approximation$')


ax1.set_xticks(x_tick_locations)
ax1.set_yticks(y_tick_locations)
ax1.set_xlabel(r'$u=2\pi fL/c$', labelpad=5,fontsize=15)
ax1.set_ylabel(r'$R^V_{MC}$', labelpad=5,fontsize=15)
#ax1.tick_params(axis='both', which='minor', labelsize=15) 
ax1.legend()

ax1 = fig.add_subplot(223)
y_tick_locations=np.array([.0001,.001, 0.01, 0.1])
x_tick_locations=np.array([.001, .01, .1, 1, 10.,100.])
ax1.set_xscale('log')
plt.title('Breathing mode')

ax1.loglog(x,tqrb*16.*np.sin(x)**2.,color='k',label=r'$Analytical$')
ax1.loglog(x,tqrba*16.*np.sin(x)**2.,":r",label=r'$Approximation$')


ax1.set_xticks(x_tick_locations)
ax1.set_yticks(y_tick_locations)
ax1.set_xlabel(r'$u=2\pi fL/c$', labelpad=5,fontsize=15)
ax1.set_ylabel(r'$R^b_{MC}$', labelpad=5,fontsize=15)
#ax1.tick_params(axis='both', which='minor', labelsize=15) 
ax1.legend()

ax1 = fig.add_subplot(224)
y_tick_locations=np.array([.001, 0.01, 0.1])
x_tick_locations=np.array([.001, .01, .1, 1, 10.,100.])
ax1.set_xscale('log')
plt.title('Longitudinal mode')

ax1.loglog(x,tqrl*16.*np.sin(x)**2.,color='k',label=r'$Analytical$')
ax1.loglog(x,tqrla*16.*np.sin(x)**2.,":r",label=r'$Approximation$')


ax1.set_xticks(x_tick_locations)
ax1.set_yticks(y_tick_locations)
ax1.set_xlabel(r'$u=2\pi fL/c$', labelpad=5,fontsize=15)
ax1.set_ylabel(r'$R^l_{MC}$', labelpad=5,fontsize=15)
#ax1.tick_params(axis='both', which='minor', labelsize=15) 
ax1.legend()

plt.tight_layout()
plt.subplots_adjust(wspace=0.5,hspace=0.8)  
plt.savefig('MCfig.jpeg',dpi=300)
plt.savefig('MCfig.pdf')
plt.show()