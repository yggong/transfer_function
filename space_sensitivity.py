# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:52:30 2019
Sensitivity curves for space-based gw detector LISA/TQ's
@author: ygong
Based on the code from https://github.com/eXtremeGravityInstitute/LISA_Sensitivity
pleas cite 1803.01944
"""

import numpy as np
import matplotlib.pyplot as plt
import LISA_tools as LISA

#%config InlineBackend.figure_format = 'retina'

#Set physical and LISA constants

""" Cosmological values """
H0      = 69.6      # Hubble parameter today
Omega_m = 0.286     # density parameter of matter

""" Observation Period """
Tobs = 4.*LISA.YEAR

""" Number of Michelson Data Channels """
NC = 2

constants = np.array([H0, Omega_m, Tobs, NC])

#Construct, Plot, and save to dat file, 
#the characteristic noise strain for specified observation period

plt.figure(figsize=(8,6))
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('font', family='calibri')

#Calculate, Plot, and save the noise curve
# h_{eff} with unit 1/sqrt(Hz)
#h_c=sqrt(f)*h_e

f, Sn = LISA.get_Sn(constants,'lisa')

#Snls = LISA.get_Snls(f)  #original lisa result with longer arm
#plt.loglog(f, np.sqrt(f*Snls),'b', label='Simulated') # plot the simulated characteristic strain

L, f_star, S_lp, S_ac = LISA.get_detector('lisa') 
Snlsapprox = LISA.get_Sn_approx(f, f_star, L, NC, S_lp, S_ac)

# plot tq sensitivity curve
ftq, Sntq = LISA.get_Sn(constants,'tq')

'''
plt.loglog(f, np.sqrt(Sn),'k',label='LISA With pattern') # plot the strain
plt.loglog(f, np.sqrt(Snlsapprox),'r',label='LISA Approximated') # plot the simulated strain
plt.loglog(ftq, np.sqrt(Sntq),'c',label='TQ') # plot TQ's strain
plt.ylabel(r'$h_{eff}^T(f)$ [Hz$^{-1/2}$]', fontsize=20, labelpad=10)
'''

# plot characteristic strain
plt.loglog(f, np.sqrt(f*Sn),'k',label='LISA with pattern') # plot the characteristic strain
plt.loglog(f, np.sqrt(f*Snlsapprox),'r',label='LISA Approximated') # plot the simulated characteristic strain
plt.loglog(ftq, np.sqrt(ftq*Sntq),'c',label='TQ') # plot the characteristic strain for TQ
plt.ylabel(r'Characteristic Strain', fontsize=20, labelpad=10)


plt.xlabel(r'$f$ [Hz]', fontsize=20, labelpad=10)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlim(1.0e-4, 10.0)
plt.ylim(3.0e-22, 1.0e-15)
plt.tick_params(labelsize=20)
plt.tight_layout()
plt.legend()
plt.show()

'''
plt.savefig('Sens_he.jpeg',dpi=400)
plt.savefig('Sens_he.pdf',dpi=400)
'''

#plt.savefig('Sens_hc.jpeg',dpi=400)
#plt.savefig('Sens_hc.pdf',dpi=400)


#out_file = 'characteristic_noise_strain.dat'
#np.savetxt(out_file,(np.vstack((f, np.sqrt(f*Sn))).T), delimiter=' ')

#Calculate  Strain, plot it appropriately, and save to dat file
#Massive Binary Black Hole

#m1 = 1.0e6*LISA.TSUN # leading coefficient is the 
#m2 = 1.0e6*LISA.TSUN #   mass in terms of solar mass

D_lum    = None         # Luminosity Distance, meters
z        = 3.           # Redshift
T_merger = 1.*LISA.YEAR # time to merger
f_start  = None         # start frequency

#LISA.calculate_plot_source(m1, m2, constants, Dl=D_lum, z=z, T_merger=T_merger, f_start=f_start)

plt.figure(figsize=(12,8))
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('font', family='calibri')

# get sensitivity curve for lisa
f, Sn = LISA.get_Sn(constants,'lisa')

ftq, Sntq = LISA.get_Sn(constants,'tq')

fa = np.arange(1,1001,0.5)
Sna = LISA.get_Snaligo(fa)

plt.loglog(f, np.sqrt(Sn),'k') # plot the strain
plt.loglog(ftq, np.sqrt(Sntq),'c') # plot the strain
plt.loglog(fa, np.sqrt(Sna),'k') # plot the simulated aLIGO noise

'''
plt.loglog(f, np.sqrt(f*Sn),'k') # plot the characteristic strain
plt.loglog(ftq, np.sqrt(ftq*Sntq),'c') # plot the characteristic strain
plt.loglog(fa, np.sqrt(fa*Sna),'k') # plot the simulated aLIGO noise
'''

detector = "tq"

for m in [1.0e6, 1.0e5, 1.0e4, 1.0e3]:
    m1=m*LISA.TSUN
    m2=m*LISA.TSUN
    f, h_c_arr, SNR, SNR1 = LISA.calculate_source(detector, m1, m2, constants, 
                                                           Dl=D_lum, z=z, T_merger=T_merger, f_start=f_start)
 #   plt.loglog(f, h_c_arr,'b',lw=2, label='SNR: ' + str("%4.1f" % SNR)) #h_c 
    plt.loglog(f, h_c_arr/np.sqrt(f),'b',lw=2, label='SNR: ' + str("%4.1f" % SNR)) #h_e


# GW150914, need to redefine f
m7       = 36.0*LISA.TSUN # leading coefficient is the 
m8       = 29.0*LISA.TSUN #   mass in terms of solar mass
D_lum    = 410.*LISA.MPC # Luminosity Distance, meters
z        = None          # Redshift
T_merger = 1.*LISA.YEAR  # time to merger

f, h_c_arr, SNR, SNR1 = LISA.calculate_source(detector, m7, m8, constants, Dl=D_lum, z=z, T_merger=T_merger, f_start=f_start)
plt.loglog(f, h_c_arr/np.sqrt(f),'m',lw=2, label='SNR: '+str("%4.1f" % SNR) + '/'+str("%4.1f" % SNR1)) 
#plt.loglog(f, h_c_arr,'m',lw=2, label='SNR: ' + str("%4.1f" % SNR) + '/'+str("%4.1f" % SNR1)) #h_c


m3 = 0.50*LISA.TSUN 
m4 = 0.25*LISA.TSUN
D_lum = 1.0e-3*LISA.MPC
f_start  = 2.6e-3
z= None
#plot h_e, if you want to plot h_c, you need to uncomment the line 362 in LISA_tools.py
f, h_c_arr, SNR, SNR1 = LISA.calculate_source(detector, m3, m4, constants, Dl=D_lum, z=z, T_merger=T_merger, f_start=f_start)


font1 = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 20,
        }

font2 = {'family': 'serif',
        'color':  'magenta',
        'weight': 'normal',
        'size': 20,
        }

font3 = {'family': 'serif',
        'color':  'cyan',
        'weight': 'normal',
        'size': 20,
        }

plt.xlim(1.0e-6, 1.0e3)
plt.ylim(1.0e-24, 1.0e-14)
plt.title('SNR for '+detector, fontdict=font1)
plt.text(5.e-5, 1.0e-15, r'$M=10^6M_\odot @ z=3$', fontdict=font1)
plt.text(1.e-4, 6.0e-17, r'$M=10^5M_\odot$', fontdict=font1)
plt.text(2.e-3, 1.5e-18, r'$M=10^4M_\odot$', fontdict=font1)
plt.text(4.e-3, 1.5e-19, r'$M=10^3M_\odot$', fontdict=font1)
plt.text(9., 2.0e-21, r'GW150914', fontdict=font2)
plt.text(3., 1.0e-20, r'aLIGO', fontsize=20)
plt.text(0.2, 6.0e-19, r'LISA', fontsize=20)
plt.text(0.9, 1.0e-19, r'TQ', fontdict=font3)
plt.legend()
plt.xlabel(r'$f$ [Hz]', fontsize=20, labelpad=10)
#plt.ylabel(r'Characteristic Strain', fontsize=20, labelpad=10)
plt.ylabel(r'$h_{eff}(f)$ [Hz$^{-1/2}$]', fontsize=20, labelpad=10)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
#plt.savefig('{}_he.pdf'.format(detector),dpi=600)
#plt.savefig('{}_he.jpeg'.format(detector),dpi=600)
plt.show()

