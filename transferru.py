# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:14:48 2019
plot the response function R_A(u), R_A(f) (A=+,x,vx,vy,b,l)
and sensitivity curves h_{eff}=sqrt(P_n/R_B) (B=t,v,b,l)
@author: Gong
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from scipy.integrate import quad
from scipy.special import sici
from scipy.interpolate import interp1d

# constants
pi=np.pi
Mpc=3.085677581*1.0e22
H0 = 6.79*1.0e4/Mpc
yr = 365.*24.*3600.

#TianQin parameters
tql=np.sqrt(3)*1.0e8
Sx = 1.0e-12
Sa = 1.0e-15  
rho = 1.
frefTQ = 0.01

#Tensor response function
def frt1(u):
    gamma=pi/3.
    a1=np.sin(u)**2.+2.*np.sin(2.*u)/u**3.
    b1=(1.+np.cos(u)**2.)*(1./3.-2./u**2)
    c1=nquad(frt,[[0.,pi],[0.,2.*pi]],args=(u,gamma))
    z=(a1+b1-c1[0])/4./u**2.
    return z

#Breathing response function
def frb1(x):
    gamma=pi/3.
    a1=np.sin(x)**2.+2.*np.sin(2.*x)/x**3.
    b1=(1.+np.cos(x)**2.)*(1./3.-2./x**2)
    c1=nquad(frb,[[0.,pi],[0.,2.*pi]],args=(x,gamma))
    z=(a1+b1-c1[0])/2./x**2.
    return z

#Vector response function
def frv1(x):
    gamma=pi/3.
    a1=-5.+2.*np.log(2.)+2.*np.euler_gamma+2.*np.log(x)-np.cos(2.*x)/3.-2.*sici(2.*x)[1]
    b1=(-4.*np.sin(2.*x)+4.*(1+np.cos(x)**2.)*x+2.*np.sin(2.*x)*x*x)/x**3.
    c1=nquad(frv,[[0.,pi],[0.,2.*pi]],args=(x,gamma))
    z=(a1+b1-c1[0])/x/x/2.
    return z

#Longitudinal response function
def frl1(x):
    gamma=pi/3.
    a2=(11./3.-np.log(2)-np.euler_gamma-np.log(x))*np.cos(2.*x)+(9.+np.cos(2.*x))*sici(2.*x)[1]
    a1=15.-9.*np.log(2.)-9.*np.euler_gamma-9.*np.log(x)+(2.*x+np.sin(2.*x))*sici(2.*x)[0]
    b1=8.*(np.sin(2.*x)-(1+np.cos(x)**2.)*x-np.sin(2.*x)*x*x)/x**3.
    c1=nquad(frl,[[0.,pi],[0.,2.*pi]],args=(x,gamma))
    z=(a2+a1+b1-c1[0])/8./x/x
    return z    

def eta(a,b,x,c):
    mu1=np.cos(a)
    mu2=np.cos(c)*np.cos(a)+np.sin(c)*np.sin(a)*np.cos(b)
    z1=(np.cos(x)-np.cos(mu1*x))*(np.cos(x)-np.cos(mu2*x))*mu1*mu2
    z2=(np.sin(x)-mu1*np.sin(mu1*x))*(np.sin(x)-mu2*np.sin(mu2*x))
    z=z1+z2
    return z

def frt(a,b,x,c):
    mu2=np.cos(c)*np.cos(a)+np.sin(c)*np.sin(a)*np.cos(b)
    z=np.sin(a)/4./pi*(1.-2.*np.sin(c)**2.*np.sin(b)**2/(1-mu2**2.))*eta(a,b,x,c)
    return z

def frb(a,b,x,c):
    z=np.sin(a)/4./pi*eta(a,b,x,c)
    return z

def frl(a,b,x,c):
    mu2=np.cos(c)*np.cos(a)+np.sin(c)*np.sin(a)*np.cos(b)
    z=np.cos(a)**2.*mu2**2.*eta(a,b,x,c)/pi/np.sin(a)/(1-mu2**2.)
    return z

def frv(a,b,x,c):
    mu1=np.cos(c)*np.sin(a)-np.sin(c)*np.cos(a)*np.cos(b)
    mu2=np.cos(c)*np.cos(a)+np.sin(c)*np.sin(a)*np.cos(b)
    z=np.cos(a)*mu1*mu2*eta(a,b,x,c)/(1-mu2**2.)/2./pi
    return z

def get_Pn(f, L, S_lp, S_ac):
    """
    Get the Power Spectral Density
    """   
    # single-link optical metrology noise (Hz^{-1}), Equation (10)
    P_oms = S_lp**2     
    # single test mass acceleration noise, Equation (11)
    P_acc = S_ac**2*(1. + 0.1e-3/f)    
    # total noise in Michelson-style LISA data channel, Equation (12)
    Pn = (P_oms + 4.*P_acc/(2.*pi*f)**4.)/L**2.    
    return Pn

rttrack=[]
rbtrack=[]
rvtrack=[]
rltrack=[]

x0=np.arange(-3.,-1.,0.1)
x2=np.arange(-1.,2.,0.01)
x3=np.concatenate((x0,x2),axis=0)
#x=np.arange(-1.,2.2,0.01) 
x3=10.**x3
x4=np.arange(94.,160.,1.)
x=np.concatenate((x3,x4),axis=0)
ft=2.99792458*x/2./pi/np.sqrt(3.)
tqsn=get_Pn(ft,tql,Sx,Sa)

x1=np.arange(1.,200,1.)
y=np.sin(pi/3.)/5/x1/x1 
y1=np.sin(pi/3.)/x1/2.5
y2=1.8/np.log(10)*np.log(x1)/x1/x1

for u in x:
    rt=frt1(u)
    rb=frb1(u)
    rv=frv1(u)
    rl=frl1(u)
    rttrack.append(rt)
    rbtrack.append(rb)
    rvtrack.append(rv)
    rltrack.append(rl)

hefft=np.sqrt(tqsn/rttrack/2.)
heffb=np.sqrt(tqsn/rbtrack)
heffv=np.sqrt(tqsn/rvtrack/2.)
heffl=np.sqrt(tqsn/rltrack)

file1='transfer.txt'
data=np.array([x,ft,rttrack,rvtrack,rbtrack,rltrack,hefft,heffv,heffb,heffl])
data=data.T
with open(file1,'w') as file_object:
    file_object.write('x ft rt rb rv rl ht hv hb hl \n')
    np.savetxt(file_object,data)
file_object.close() 

omt_eff=hefft*hefft*2.*pi*pi*ft**3./3./H0/H0
omv_eff=heffv*heffv*2.*pi*pi*ft**3./3./H0/H0
omb_eff=heffb*heffb*2.*pi*pi*ft**3./3./H0/H0
oml_eff=heffl*heffl*2.*pi*pi*ft**3./3./H0/H0

omtf=interp1d(ft,omt_eff)
omvf=interp1d(ft,omv_eff)
ombf=interp1d(ft,omb_eff)
omlf=interp1d(ft,oml_eff)

def fpow(f,b):
    return (f/frefTQ)**b

def omfbt(f,b):
    z=fpow(f,b)**2./omtf(f)**2.
    return z

def ombt(b):
    a=quad(omfbt,ft[0],30.,args=(b,))
    z=rho/np.sqrt(2.*yr)*a[0]**(-0.5)
    return z

def omfbv(f,b):
    z=fpow(f,b)**2./omvf(f)**2.
    return z

def ombv(b):
    z=rho/np.sqrt(2.*yr)*(quad(omfbv,ft[0],30.,args=(b,))[0])**(-0.5)
    return z

def omfbb(f,b):
    z=fpow(f,b)**2./ombf(f)**2.
    return z

def ombb(b):
    z=rho/np.sqrt(2.*yr)*(quad(omfbb,ft[0],30.,args=(b,))[0])**(-0.5)
    return z

def omfbl(f,b):
    z=fpow(f,b)**2./omlf(f)**2.
    return z

def ombl(b):
    z=rho/np.sqrt(2.*yr)*(quad(omfbl,ft[0],30.,args=(b,))[0])**(-0.5)
    return z

omttrack=[]
omvtrack=[]
ombtrack=[]
omltrack=[]
omt2=[]
omv2=[]
omb2=[]
oml2=[]
omt2a=[]
omv2a=[]
omb2a=[]
oml2a=[]

beta=np.arange(-10.,10.,0.5)
for i in range(len(beta)):
    omt2.append(ombt(beta[i]))
    omv2.append(ombv(beta[i]))
    omb2.append(ombb(beta[i]))
    oml2.append(ombl(beta[i]))
    omt2a.append(0)
    omv2a.append(0)
    omb2a.append(0)
    oml2a.append(0)
    
ft1=np.arange(-3.5,0,0.035)
ft1=10.**ft1
for f1 in ft1:
    for i in range(len(beta)):
      omt2a[i]=omt2[i]*fpow(f1,beta[i])
      omv2a[i]=omv2[i]*fpow(f1,beta[i])
      omb2a[i]=omb2[i]*fpow(f1,beta[i])
      oml2a[i]=oml2[i]*fpow(f1,beta[i])
      
    omttrack.append(max(omt2a))
    omvtrack.append(max(omv2a))
    ombtrack.append(max(omb2a))
    omltrack.append(max(oml2a))

file2='PIdata.txt'
data=np.array([ft1,omttrack,omvtrack,ombtrack,omltrack])
data=data.T
with open(file2,'w') as file_object:
    np.savetxt(file_object,data)
file_object.close() 

#plot R_A(u)
plt.figure(figsize=(8,6))
#plt.rcParams['text.usetex'] = True
#plt.rc('text', usetex=True)
#plt.rc('font', family='calibri')
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.set_xscale('log')
ax2.set_xscale('log')
#plt.rcParams['text.usetex'] = True
#plt.rc('text', usetex=True)
#plt.rc('font', family='calibri')
ax1.loglog(x,rttrack,'b',label=r'Tensor $R_{+/\times}$')
ax1.loglog(x,rbtrack,'k',label='Breathing $R_b$')
ax1.loglog(x,rltrack,'r',label='Longitudinal $R_l$')
ax1.loglog(x,rvtrack,color='lightgreen',label='Vector $R_{x/y}$')
ax1.loglog(x1,y,'--b',label='$1/f^2$')
ax1.loglog(x1,y1,'--r',label='$1/f$')
ax1.loglog(x1,y2,linestyle='--',color='lightgreen',linewidth=2,label='$\ln(f)/f^2$')
ax1.set_xlabel(r'$a$', labelpad=5,fontsize=15)
ax1.set_ylabel(r'$R_A$', labelpad=5,fontsize=15)
ax1.tick_params(axis='both', which='major', labelsize=15)

new_tick_locations =np.sqrt(3.)*pi*2.*np.array([.001, 0.01, 0.1,1.,10.])/2.99792458
V1=np.array(['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^0$','$10^1$'])
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(V1)
ax2.set_xlabel(r'$f$/Hz', labelpad=8,fontsize=15)
ax2.tick_params(labelsize=15)

"""
 alternative ways
def tick_function(x):
    V = 2.99792458*x/2./pi/np.sqrt(3.)
    return ["%2.1e" % z for z in V]
#        return ["%1.1e" % z for z in V]

ax2.set_xticklabels(tick_function(new_tick_locations))


#ax1Xs = ax1.get_xticks()
#ax1Xs = np.array([.001, 0.01, 0.1,1.,10.,100.])
#ax2Xs = []
#for x in ax1Xs:
#    ax2Xs.append(2.99792458*x/2./pi/np.sqrt(3.))

#ax2.set_xticks(ax1Xs)
#ax2.set_xbound(ax1.get_xbound())
#ax2.set_xticklabels(ax2Xs)
"""

plt.tight_layout()
ax1.legend()
plt.savefig('raf.pdf')
plt.show()

"""
#plot R_A(f)
plt.figure(figsize=(8,6))
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('font', family='calibri')
plt.loglog(ft,rttrack,'b',label='Tensor $R_t/2$')
plt.loglog(ft,rbtrack,'k',label='Breathing $R_b$')
plt.loglog(ft,rltrack,'r',label='Longitudinal $R_l$')
plt.loglog(ft,rvtrack,color='lightgreen',label='Vector $R_v/2$')
plt.xlabel(r'$f$/Hz', labelpad=5,fontsize=15)
plt.ylabel(r'$R_A(f)$', labelpad=5,fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)  
plt.tight_layout()  
plt.legend()
plt.savefig('tqrf.pdf')
plt.show()
"""

#plot h_{eff}^A(f)
plt.figure(figsize=(8,6))
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('font', family='calibri')
plt.loglog(ft,hefft,'b',label='Combined Tensor')
plt.loglog(ft,heffb,'k',label='Breathing')
plt.loglog(ft,heffl,'r',label='Longitudinal')
plt.loglog(ft,heffv,color='lightgreen',label='Combined Vector')
plt.xlabel(r'$f$/Hz', labelpad=5,fontsize=15)
plt.ylabel(r'$h_{eff}^A(f)$/Hz$^{-1/2}$', labelpad=5,fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15) 
plt.tight_layout()   
plt.legend()
plt.savefig('tqheff.pdf')
plt.show()

# plot PI curve
plt.figure(figsize=(8,6))
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('font', family='calibri')
plt.loglog(ft1,omttrack,'b',label='Combined Tensor')
plt.loglog(ft1,ombtrack,'k',label='Breathing')
plt.loglog(ft1,omltrack,'r',label='Longitudinal')
plt.loglog(ft1,omvtrack,color='lightgreen',label='Combined Vector')
#plt.xlim(0.0001,1.)
plt.xlabel(r'$f$(Hz)/Hz', labelpad=5,fontsize=15)
plt.ylabel(r'$\Omega_A(f)$', labelpad=5,fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15) 
plt.tight_layout()   
plt.legend()
plt.savefig('TQPIcurve.pdf')
plt.show()
