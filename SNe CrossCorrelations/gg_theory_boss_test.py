'''

cltheory_kappa_kappa_march15.py

Ross Cawthon
3/25/15

from cltheory_kappacross_kappakappa_v5.py and others

5/1/18
Chicago

magnification_theory.py
Ross Cawthon
3/4/20
Madison

gg_theory_boss_test.py
8/18/20
Madison

'''

import numpy as np
import scipy as sp
from scipy import integrate
from scipy import interpolate
from scipy.interpolate import interp1d, UnivariateSpline,splrep
from scipy.integrate import quad, dblquad
from scipy.special import j0
from scipy.misc import derivative
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import datetime
import fitsio

def H(a):
	H0=100
	om=0.3089
	ol=1.-om
	result=H0*((om/(1.*a**3))+ol)**0.5
	return result

#Solve chi(z), since H(a) is in km/sec/mpc, using c=3*10^5 km/sec leaves a result in mpc
#Equation 2.42 pg 34 Dodelson
def chi(z):
	athen=1/(1+z)
	c=3.*10**5
	chiint=integrate.quad(lambda a: 1/((a**2)*H(a)), athen, 1)
	result=c*chiint[0]
	return result
	
zs201=np.arange(201)*0.01
chiarray=np.zeros(201)
for i in range(0,201):
	chiarray[i]=chi(zs201[i])

zchiinterp=sp.interpolate.InterpolatedUnivariateSpline(chiarray,zs201,k=5)


h=0.6774
def zchi(chi):
	result=zchiinterp(chi)
	return result

#check with ned wright site
print zchi(100)
print zchi(1000)
print zchi(1317)  #my chis are true chi*h....so this is 1945 real mpc
print zchi(1945)  #this is 2871 real mpc
print zchi(2871)

derivarray=np.zeros(201)
for j in range(0,201):
	derivarray[j]=derivative(zchi,chiarray[j])
	
print chiarray
print derivarray
plt.plot(chiarray[1:200]/h,derivarray[1:200]*h)
plt.xlabel('chi')
plt.ylabel('deriv z(chi)')
plt.show()



#Full calculation Choi et al. 2016
desnz=fitsio.read('galaxy_DR12v5_CMASSLOWZ_South.fits')
zs1=np.arange(101)*0.01+0.
zs=np.arange(100)*0.01+0.005
b=plt.hist(desnz['Z'],bins=zs1)
pz=b[0]
pz=pz*100/np.sum(b[0])
plt.show()

W2 = sp.interpolate.InterpolatedUnivariateSpline(zs,pz,k=5)

w2array=np.zeros(len(zs))
for j in range(0,len(w2array)):
	w2array[j]=W2(zs[j])

plt.plot(zs,w2array)
normal=integrate.quad(lambda z: W2(z),0.0,1.0)
print 'normal',normal
plt.show()


normal2=integrate.quad(lambda chi: W2(zchi(chi))*derivative(zchi,chi),chi(0.0),chi(1.0))
print 'normal2',normal2

newchiarray=np.zeros(len(zs))
for p in range(0,len(newchiarray)):
	newchiarray[p]=chi(zs[p])

w22array=np.zeros(len(zs))
for jj in range(0,len(w22array)):
	w22array[jj]=W2(zchi(newchiarray[jj]))*derivative(zchi,newchiarray[jj])
plt.plot(newchiarray,w22array)
plt.title('n(z(chi))*dz/dchi')
plt.show()


pkall=np.genfromtxt('pk_none_z0-7_k200cosmosis_kmax100_2019.txt')
newpk=np.reshape(pkall,((701,200)))
klist=np.genfromtxt('k_h_planck2015ext.txt')
kmin=np.min(klist)
kmax=np.max(klist)
zlist=np.arange(701)*0.01
Pk = sp.interpolate.RectBivariateSpline(zlist,klist,newpk,kx=5,ky=5)
	
def newPk3(k,z):
	result=Pk(z,k)
	return result

plt.plot(klist,newpk[0],label='z=0')
plt.plot(klist,newpk[100],label='z=1')
plt.xlabel('k (h units)')
plt.ylabel('p(k) (h units)')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-4,1)
plt.ylim(2*10,3*10**4)
plt.legend()
plt.show()
	

def powerchi(chis,theta):
	h=0.6774
	powint=integrate.quad(lambda k: k*newPk3(k,zchi(chis))*j0(chis*k*theta), kmin, kmax,epsrel=10**-12,epsabs=10**-12,limit=5000)
	return powint[0]/(2*np.pi)


def g2(chis):
	nn=W2(zchi(chis))*derivative(zchi,chis)
	return nn
	
def ggtheta(theta):
	#chimin=chi(zref-0.01)
	#chimax=chi(zref+0.01)
	chimin=chi(0.01)
	chimax=chi(0.9)
	b1=1
	b2=1
	n2=50.
	result=b1*b2*integrate.quad(lambda chi: g2(chi)**2*powerchi(chi,theta), chimin,chimax)
	return result[0]
	
kk=0.1
print 'j0',j0(chi(0.5)*kk*0.1*180/np.pi)
print 'g2',g2(chi(0.5))
#print 'powerchi', powerchi(chi(0.5),0.1*180/np.pi)
	
#print 'theta=0.1 deg',ggtheta(0.1*np.pi/180.)
#print 'theta=0.5 deg',ggtheta(0.5*np.pi/180.)
#print 'theta=1.0 deg',ggtheta(1.0*np.pi/180.)

thetaarray=[0.02,0.06,0.1,0.3,0.5,0.7,1.0]
ggarray=np.zeros((2,len(thetaarray)))
ggarray[0]=1*thetaarray
for q in range(0,len(thetaarray)):
	print q
	ggarray[1][q]=ggtheta(thetaarray[q]*np.pi/180.)

np.savetxt('gg_boss_theory_check1.txt',ggarray)

print 'gg done'