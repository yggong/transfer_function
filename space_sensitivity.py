# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:52:30 2019
Space-based gw detector LISA/Tianqin/Taiji's Sensitivity
based on 1803.01944
detectors: lisa,tq,tj
@author: ygong
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

#detectors: 'lisa', 'tq', 'tj'
#Construct, Plot, and save to dat file, 
#the characteristic noise strain for specified observation period
'''
plt.figure(figsize=(8,6))
plt.rcParams['text.usetex'] = True
plt.ion()
plt.rc('text', usetex=True)
plt.rc('font', family='calibri')
plt.xlabel(r'$f$ [Hz]', fontsize=20, labelpad=10)
plt.ylabel(r'Characteristic Strain', fontsize=20, labelpad=10)
#plt.ylabel(r'$h_{eff}(f)$ [Hz$^{-1/2}$]', fontsize=20, labelpad=10)
plt.tick_params(axis='both', which='major', labelsize=20)



#Calculate, Plot, and save the noise curve

f, Snls = LISA.get_Sn('lisa', constants)
plt.loglog(f, np.sqrt(f*Snls),'b',label='LISA') # plot the characteristic strain
#plt.loglog(f, np.sqrt(Snls),'b',label='LISA') # plot the effective strain
#plt.loglog(f, np.sqrt(f*Sn),'k',label='With pattern') # plot the characteristic strain
#plt.loglog(f, 2*np.pi**2*f**3*Sn/3/(H0*1000/3.08568025e22)**2,'k',label='LISA With pattern')

ftj, Sntj = LISA.get_Sn('tj', constants)
plt.loglog(ftj, np.sqrt(ftj*Sntj),'c',label='Taiji') # plot the characteristic strain
#plt.loglog(ftj, np.sqrt(Sntj),'c',label='Taiji') # plot the strain

ftq, Sntq = LISA.get_Sn('tq', constants)
plt.loglog(ftq, np.sqrt(ftq*Sntq),'r',label='TianQin') # plot the characteristic strain
#plt.loglog(ftq, np.sqrt(Sntq),'r',label='TianQin') # plot the strain

fa = np.arange(1,1001,0.5)
Sna = LISA.get_Snaligo(fa)
#plt.loglog(fa, np.sqrt(Sna),'k',label="aLIGO") # plot the simulated aLIGO noise
plt.loglog(fa, np.sqrt(fa*Sna),'k',label="aLIGO") # plot the simulated aLIGO noise

#Snls = LISA.get_Snls(f)  #original lisa result with longer arm
#plt.loglog(f, np.sqrt(f*Snls),'b', label='Simulated') # plot the simulated characteristic strain

plt.xlim(1.0e-6, 1.0e3)
plt.ylim(1.0e-24, 1.0e-12)
plt.tight_layout()
plt.legend()
plt.savefig('Space_Curveshc.jpeg',dpi=400)
plt.show()


L, f_star, S_lp, S_ac = LISA.get_detector('lisa') 
Snlsapprox = LISA.get_Sn_approx(f, f_star, L, NC, S_lp, S_ac)
plt.loglog(f, np.sqrt(Snlsapprox),'r',label='LISA Approximated') # plot the simulated strain
#plt.loglog(f, np.sqrt(f*Snlsapprox),'r',label='Approximated') # plot the simulated characteristic strain
'''
'''
ftq, Sntq = LISA.get_Sn('tq', constants)
#plt.loglog(ftq, np.sqrt(Sntq),'c',label='TQ') # plot TQ's strain
plt.loglog(ftq, np.sqrt(ftq*Sntq),'c',label='TQ') # plot the characteristic strain for TQ
#plt.loglog(ftq, 2*np.pi**2*ftq**3*Sntq/3/(H0*1000/3.08568025e22)**2,'k',label='TQ')

fa = np.arange(1,1001,0.5)
Sna = LISA.get_Snaligo(fa)
plt.loglog(fa,2*np.pi**2*fa**3*Sna/3/(H0*1000/3.08568025e22)**2,'k',label='aLIGO') # plot the simulated aLIGO noise

plt.xlim(1.0e-4, 1000.0)
plt.ylim(1.0e-12, 1.0e-2)
plt.tick_params(labelsize=20)

plt.tight_layout()
plt.legend()
plt.savefig('Sens_Curves.jpeg',dpi=400)

plt.show()
'''

'''
# test the transfer function Get_Ru(u)
transfer_data = np.genfromtxt('R.txt')
fu=transfer_data[:,0]
Rd=transfer_data[:,1]
Ra = LISA.Get_Ru(fu)
# due to numerical accuracy for ci(x) as x->0, set the theoretical value 
f_c = 8.5e-4  #set the minimum frequency 
n_e = np.where(fu>=f_c)[0][0]
Ra[0:n_e] = 1./5.*np.sin(np.pi/3.)**2.  

plt.figure(figsize=(8,6))
plt.rcParams['text.usetex'] = True
plt.ion()
plt.rc('text', usetex=True)
plt.rc('font', family='calibri')
plt.xlabel(r'$f$ [Hz]', fontsize=20, labelpad=10)
plt.ylabel(r'$R(u)$', fontsize=20, labelpad=10)
plt.tick_params(axis='both', which='major', labelsize=20)

plt.loglog(fu,Rd,'b',label='Rd')
plt.loglog(fu,Ra,'c',label='Ra')
plt.show()

#save the transfer function to transfer.txt

file1='t_ru.txt'
data=np.array([fu,Rd,Ra])
data=data.T
with open(file1,'w') as file_object:
    file_object.write('f Rd Ra \n')
    np.savetxt(file_object,data)
file_object.close() 
'''

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
#plt.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rcParams["legend.framealpha"] = 0.1
#plt.ion()
#plt.rc('text', usetex=True)
plt.rc('font', family='calibri',size=20,weight='bold')

f, Sn = LISA.get_Sn('lisa', constants)
plt.loglog(f, np.sqrt(f*Sn),color='cyan',lw=2) # plot the characteristic strain
#plt.loglog(f, np.sqrt(Sn),'k') # plot the strain

detector = "tq"

ftq, Sntq = LISA.get_Sn(detector, constants)
plt.loglog(ftq, np.sqrt(ftq*Sntq),color='magenta',lw=2) # plot the characteristic strain
#plt.loglog(ftq, np.sqrt(Sntq),'c') # plot the strain

ftj, Sntj = LISA.get_Sn('tj', constants)
plt.loglog(ftj, np.sqrt(ftj*Sntj),'lightcoral',lw=2) # plot the characteristic strain

fa = np.arange(1,1001,0.5)
Sna = LISA.get_Snaligo(fa)
#plt.loglog(fa, np.sqrt(Sna),'k') # plot the simulated aLIGO noise
plt.loglog(fa, np.sqrt(fa*Sna),'brown',lw=2) # plot the simulated aLIGO noise

linesty=['dashed', 'dotted', 'dashdot', 'solid']
li=0
for m in [1.0e6, 1.0e5, 1.0e4, 1.0e3]:
    m1=m*LISA.TSUN
    m2=m*LISA.TSUN
    f, h_e_arr, h_c_arr, SNR, SNR1 = LISA.calculate_source(detector, m1, m2, constants, 
                                                           Dl=D_lum, z=z, T_merger=T_merger, f_start=f_start)
    plt.loglog(f, h_c_arr,'b',lw=2, linestyle=linesty[li], label='SNR: ' + str("%4.1f" % SNR)) 
    li=li+1
#    plt.loglog(f, h_e_arr,'b',lw=2, label='SNR: ' + str("%4.1f" % SNR)) 
    #print SNR

# GW150914, need to redefine f
m7       = 36.0*LISA.TSUN # leading coefficient is the 
m8       = 29.0*LISA.TSUN #   mass in terms of solar mass
D_lum    = 410.*LISA.MPC # Luminosity Distance, meters
z        = None          # Redshift
T_merger = 1.*LISA.YEAR  # time to merger

f, h_e_arr, h_c_arr, SNR, SNR1 = LISA.calculate_source(detector, m7, m8, constants, Dl=D_lum, z=z, T_merger=T_merger, f_start=f_start)
#plt.loglog(f, h_e_arr,'m',lw=2, label='SNR: '+str("%4.2f" % SNR) + '/'+str("%4.2f" % SNR1)) 
plt.loglog(f, h_c_arr,'y',lw=2, label=str("%4.1f" % SNR) + '/'+str("%4.1f" % SNR1)) 
#print SNR


#ZTF J1539+5027
m3 = 0.61*LISA.TSUN 
m4 = 0.21*LISA.TSUN
D_lum = 2.34e-3*LISA.MPC
f_start  = 2.0/414.7915404
z= None
f, h_e_ztf, h_c_ztf, ZTF_SNR, SNR1 = LISA.calculate_source(detector, m3, m4, constants, Dl=D_lum, z=z, T_merger=T_merger, f_start=f_start)

m3u = (0.61+0.017)*LISA.TSUN 
m4u = (0.21+0.014)*LISA.TSUN
D_lum_d = (2.34-0.14)*1e-3*LISA.MPC
f, h_e_ztf_u, h_c_ztf_u, SNR_u, SNR1 = LISA.calculate_source(detector, m3u, m4u, constants, Dl=D_lum_d, z=z, T_merger=T_merger, f_start=f_start)

m3d = (0.61-0.022)*LISA.TSUN 
m4d = (0.21-0.015)*LISA.TSUN
D_lum_u = (2.34+0.14)*1e-3*LISA.MPC
f, h_e_ztf_l, h_c_ztf_l, SNR_d, SNR1 = LISA.calculate_source(detector, m3d, m4d, constants, Dl=D_lum_u, z=z, T_merger=T_merger, f_start=f_start)

ZTF_erroru=h_c_ztf_u -h_c_ztf
ZTF_errorl=h_c_ztf-h_c_ztf_l
#plt.scatter(f_start,h_c_ztf,c='r',label='ZTF J1539+5027(6.9 min)'+' SNR: '+str("%4.1f" % ZTF_SNR))
plt.scatter(f_start,h_c_ztf,c='k',label='ZTF J1539+5027 SNR: 22.5')
plt.errorbar([f_start],[h_c_ztf],ZTF_erroru+ZTF_errorl,ecolor='k',lolims=False,uplims=False)

#plt.scatter(f_start,h_e_ztf,c='r',label='ZTF J1539+5027(6.9 min)'+' SNR: '+str("%4.1f" % ZTF_SNR))
#plt.errorbar([f_start],[h_e_ztf],ZTF_erroru+ZTF_errorl,ecolor='r',lolims=False,uplims=False)

#SDSS J0651+2844
m5 = 0.49*LISA.TSUN 
m6 = 0.247*LISA.TSUN
D_lum = 1.e-3*LISA.MPC
f_start  = 2./765.206543
f, h_e_sdss, h_c_sdss, SDSS_SNR, SNR1 = LISA.calculate_source(detector, m5, m6, constants, Dl=D_lum, z=z, T_merger=T_merger, f_start=f_start)

m5u = (0.49+0.02)*LISA.TSUN 
m6u = (0.247+0.015)*LISA.TSUN
D_lum_l = (1.0-0.1)*1e-3*LISA.MPC
f, h_e_sdss_u, h_c_sdss_u, SNR_u, SNR1 = LISA.calculate_source(detector, m5u, m6u, constants, Dl=D_lum_l, z=z, T_merger=T_merger, f_start=f_start)

m5l = (0.49-0.02)*LISA.TSUN 
m6l = (0.247-0.015)*LISA.TSUN
D_lum_u =(1.0+0.1)* 1e-3*LISA.MPC
f, h_e_sdss_l, h_c_sdss_l, SNR_l, SNR1 = LISA.calculate_source(detector, m5l, m6l, constants, Dl=D_lum_u, z=z, T_merger=T_merger, f_start=f_start)

SDSS_erroru=h_c_sdss_u -h_c_sdss
SDSS_errorl=h_c_sdss-h_c_sdss_l
#plt.scatter(f_start,h_c_sdss,c='b',label='SDSS J0651+2844(12.8 min)'+' SNR: '+str("%4.1f" % SDSS_SNR))
plt.scatter(f_start,h_c_sdss,c='r',label='SDSS J0651+2844 SNR: 9.3')
plt.errorbar([f_start],[h_c_sdss],SDSS_erroru+SDSS_errorl,ecolor='r',lolims=False,uplims=False)

#RX J0806.3+1527
mhm1 = 0.55*LISA.TSUN 
mhm2 = 0.27*LISA.TSUN
D_l_hm = 5e-3*LISA.MPC
f_start  = 2./321.5
f, h_e_hm, h_c_hm, hm_SNR, SNR1 = LISA.calculate_source(detector, mhm1, mhm2, constants, Dl=D_l_hm, z=z, T_merger=T_merger, f_start=f_start)
#plt.plot(f_start,h_c_hm,'r*',label='RX J0806.3+1527(5.4 min)'+' SNR: '+str("%4.1f" % hm_SNR))
plt.plot(f_start,h_c_hm,'c*',label='RX J0806.3+1527 SNR: 78.0')
#plt.plot(f_start,h_e_hm,'r*',label='RX J0806.3+1527(5.4 min)'+' SNR: '+str("%4.1f" % hm_SNR))

#plot PTA, use h_c\sim f
#IPTA: 2.1e-9Hz, 3.54e-16
#SKA: 1.58e-9, 1.35e-17
fipta=2.1e-9
hipta=3.54e-16
fska=1.58e-9
hska=1.35e-17
fpta1=10.**np.arange(np.log10(fipta),-6.,0.1)
plt.loglog(fpta1,fpta1*hipta/fipta,'--',color='purple',lw=2)
fpta2=10.**np.arange(np.log10(fska),-6.,0.1)
plt.loglog(fpta2,fpta2*hska/fska,'--',color='gray',lw=2)
plt.vlines(fipta,hipta,ymax=1.e-12,colors='purple',lw=2,linestyle='dashed')
plt.vlines(fska,hska,ymax=1.e-12,colors='gray',lw=2,linestyle='dashed')
fd=10.**np.arange(-3.,1.,0.1)
fdp=7.36
sdecigo=7.05e-48*(1.+(fd/fdp)**2)+4.8e-51/fd**4/(1.+(fd/fdp)**2)+5.33e-52/fd**4
plt.loglog(fd,np.sqrt(fd*sdecigo),'k',lw=2)


font1 = {'family': 'serif',
        'color':  'blue',
        'weight': 'bold',
        'size': 20,
        }

font2 = {'family': 'serif',
        'color':  'y',
        'weight': 'bold',
        'size': 18,
        }

font3 = {'family': 'serif',
        'color':  'lightcoral',
        'weight': 'bold',
        'size': 18,
        }

font4 = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 18,
        }

font5 = {'family': 'serif',
        'weight': 'bold',
        'size': 18,
        }

font6 = {'family': 'serif',
        'color':  'magenta',
        'weight': 'bold',
        'size': 18,
        }

#plt.xlim(1.0e-6, 1.0e3)
#plt.ylim(1.0e-23, 1.0e-16)
plt.xlim(1.0e-9, 1.0e3)
plt.ylim(1.0e-25, 1.0e-12)
#plt.title('SNR for '+detector, fontdict=font1)
plt.tight_layout()
plt.tick_params(labelsize=20)
plt.text(3.e-5, 1.0e-17, r'$M=10^6M_\odot @ z=3$', fontdict=font1)
plt.text(1.e-4, 1.0e-18, r'$M=10^5M_\odot$', fontdict=font1)
plt.text(4.e-4, 1.5e-19, r'$M=10^4M_\odot$', fontdict=font1)
plt.text(2.e-3, 1.5e-20, r'$M=10^3M_\odot$', fontdict=font1)
plt.xlabel(r'$f$\textbf{/Hz}', fontsize=25, labelpad=10)
plt.ylabel(r'\textbf{Characteristic Strain}', fontsize=25, labelpad=10)
#plt.ylabel(r'$h_{eff}(f)$ [Hz$^{-1/2}$]', fontsize=20, labelpad=10)
plt.tick_params(axis='both', which='major',labelsize=20)
#plt.legend(loc="lower left",prop=dict(size=20,weight='bold'))
plt.rcParams.update({"text.usetex": False})
plt.text(5., 5.0e-21, 'GW150914', fontdict=font2)
plt.text(40., 9.5e-23, 'aLIGO', fontsize=18,weight='bold',family='serif',color='brown')
plt.text(0.2, 6.0e-19, 'LISA', fontsize=18,weight='bold',family='serif',color='cyan')
plt.text(0.8, 1.0e-19, 'TJ', fontdict=font3)
#plt.text(0.8, 2.0e-20, r'{}'.format(detector), fontdict=font4)
plt.text(0.8, 2.0e-20, 'TQ', fontdict=font6)
plt.text(4.e-9, 1.0e-14, 'IPTA', fontsize=18,weight='bold',family='serif',color='purple')
plt.text(4.e-9, 2.0e-16, 'SKA', fontsize=18,weight='bold',family='serif',color='gray')
plt.text(0.1, 6.0e-24, 'DECIGO', fontsize=18,weight='bold',family='serif',color='black')
plt.legend(loc="lower left",prop=font5)
plt.tight_layout()
plt.savefig('snr_{}.pdf'.format(detector),dpi=600)
plt.savefig('snr_{}.png'.format(detector),dpi=600)
plt.show()

