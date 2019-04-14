# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 20:15:41 2019
use the saved data to plot graphs
@author: Gong
"""

import numpy as np
import matplotlib.pyplot as plt

pi=np.pi

transfer_data = np.genfromtxt('transfer.txt')
x=transfer_data[1:,0]
ft=transfer_data[1:,1]
rttrack=transfer_data[1:,2]
rvtrack=transfer_data[1:,3]
rbtrack=transfer_data[1:,4]
rltrack=transfer_data[1:,5]
hefft=transfer_data[1:,6]
heffv=transfer_data[1:,7]
heffb=transfer_data[1:,8]
heffl=transfer_data[1:,9]


PI_data=np.genfromtxt('PIdata.txt')
ft1=PI_data[:,0]
omttrack=PI_data[:,1]
omvtrack=PI_data[:,2]
ombtrack=PI_data[:,3]
omltrack=PI_data[:,4]


x1=np.arange(1.,200,1.)
y=np.sin(pi/3.)/5/x1/x1 
y1=np.sin(pi/3.)/x1/2.5
y2=1.8/np.log(10)*np.log(x1)/x1/x1

#plot R_A(u)
fig=plt.figure(figsize=(8,6))
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
#V1=np.array([r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^0$',r'$10^1$'])
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
plt.xlabel(r'$f$/Hz', labelpad=5,fontsize=15)
plt.ylabel(r'$\Omega_A(f)$', labelpad=5,fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15) 
plt.tight_layout()   
plt.legend()
plt.savefig('TQPIcurve.pdf')
plt.show()