from __future__ import print_function
import treecorr
import fitsio
import numpy
import time
import pprint
import matplotlib
import matplotlib.pyplot as plt

file_name = '/afs/hep.wisc.edu/home/tabbutt/public/projects/Jan2020/RedMagic/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits'
data = fitsio.read(file_name)
print(data.dtype)  # Includes: ID, RA, DEC, ZREDMAGIC, ZREDMAGIC_E, weight
print(data.shape)  # 653691 objects

file_name_rand = '/afs/hep.wisc.edu/home/tabbutt/public/projects/Jan2020/RedMagic/DES_Y1A1_3x2pt_redMaGiC_RANDOMS.fits'
data_rand = fitsio.read(file_name)
print(data_rand.dtype)  # Includes: ID, RA, DEC, ZREDMAGIC, ZREDMAGIC_E, weight
print(data_rand.shape)  # 653691 objects

cat = treecorr.Catalog(file_name, ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg')
cat_rand = treecorr.Catalog(file_name_rand, ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg')

dd = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
dd.process(cat)

rr = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rr.process(cat_rand)

ra_min = numpy.min(cat.ra)
ra_max = numpy.max(cat.ra)
dec_min = numpy.min(cat.dec)
dec_max = numpy.max(cat.dec)
print('ra range = %f .. %f' % (ra_min, ra_max))
print('dec range = %f .. %f' % (dec_min, dec_max))

ra_min_rand = numpy.min(cat_rand.ra)
ra_max_rand = numpy.max(cat_rand.ra)
dec_min_rand = numpy.min(cat_rand.dec)
dec_max_rand = numpy.max(cat_rand.dec)
print('ra range = %f .. %f' % (ra_min_rand, ra_max_rand))
print('dec range = %f .. %f' % (dec_min_rand, dec_max_rand))


# Check that the randoms cover the same space as the data
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
ax1.scatter(cat.ra * 180/numpy.pi, cat.dec * 180/numpy.pi, color='blue', s=0.1)
ax1.scatter(cat_rand.ra * 180/numpy.pi, cat_rand.dec * 180/numpy.pi, color='green', s=0.1)
ax1.set_xlabel('RA (degrees)')
ax1.set_ylabel('Dec (degrees)')
ax1.set_title('Randoms on top of data')

# Repeat in the opposite order
ax2.scatter(cat_rand.ra * 180/numpy.pi, cat_rand.dec * 180/numpy.pi, color='green', s=0.1)
ax2.scatter(cat.ra * 180/numpy.pi, cat.dec * 180/numpy.pi, color='blue', s=0.1)
ax2.set_xlabel('RA (degrees)')
ax2.set_ylabel('Dec (degrees)')
ax2.set_title('Data on top of randoms')

plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
ax1.scatter(cat_rand.ra * 180/numpy.pi * numpy.cos(cat_rand.dec), cat_rand.dec * 180/numpy.pi, color='green', s=0.1)
ax1.set_xlabel('RA * cos(Dec)')
ax1.set_ylabel('Dec')
ax1.set_xlim(3,8)
ax1.set_ylim(-50,-55)
ax1.set_title('Randoms')

ax2.scatter(cat.ra * 180/numpy.pi * numpy.cos(cat.dec), cat.dec * 180/numpy.pi, color='blue', s=0.1)
ax2.set_xlabel('RA * cos(Dec)')
ax2.set_ylabel('Dec')
ax2.set_xlim(3,8)
ax2.set_ylim(-50,-55)
ax2.set_title('Data')

plt.show()

# Why are there so many more points for the randoms?

xi, varxi = dd.calculateXi(rr)

r = numpy.exp(dd.meanlogr)
sig = numpy.sqrt(varxi)


xi, varxi = dd.calculateXi(rr)

r = numpy.exp(dd.meanlogr)
sig = numpy.sqrt(varxi)

plt.plot(r, xi, color='blue')
plt.plot(r, -xi, color='blue', ls=':')
plt.errorbar(r[xi>0], xi[xi>0], yerr=sig[xi>0], color='blue', lw=0.1, ls='')
plt.errorbar(r[xi<0], -xi[xi<0], yerr=sig[xi<0], color='blue', lw=0.1, ls='')
leg = plt.errorbar(-r, xi, yerr=sig, color='blue')

plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
#plt.xlim([0.01,10])
plt.show()

dr = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
dr.process(cat, cat_rand)

xi, varxi = dd.calculateXi(rr, dr)
sig = numpy.sqrt(varxi)

xi, varxi = dd.calculateXi(rr, dr)
sig = numpy.sqrt(varxi)

plt.plot(r, xi, color='blue')
plt.plot(r, -xi, color='blue', ls=':')
plt.errorbar(r[xi>0], xi[xi>0], yerr=sig[xi>0], color='blue', lw=0.1, ls='')
plt.errorbar(r[xi<0], -xi[xi<0], yerr=sig[xi<0], color='blue', lw=0.1, ls='')
leg = plt.errorbar(-r, xi, yerr=sig, color='blue')

plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
plt.xlim([0.01,10])
plt.show()

xi, varxi = dd.calculateXi(rr, dr)
sig = numpy.sqrt(varxi)

xi, varxi = dd.calculateXi(rr, dr)
sig = numpy.sqrt(varxi)

plt.plot(r, r*xi, color='blue')
plt.plot(r, -r*xi, color='blue', ls=':')
plt.errorbar(r[xi>0], xi[xi>0], yerr=sig[xi>0], color='blue', lw=0.1, ls='')
plt.errorbar(r[xi<0], -xi[xi<0], yerr=sig[xi<0], color='blue', lw=0.1, ls='')
leg = plt.errorbar(-r, xi, yerr=sig, color='blue')

plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
plt.xlim([0.05, 3.5])
plt.show()


xi, varxi = dd.calculateXi(rr, dr)
sig = numpy.sqrt(varxi)

plt.plot(r*60, r*60*xi, color='blue')
plt.plot(r*60, -r*60*xi, color='blue', ls=':')
plt.errorbar(r[xi>0], xi[xi>0], yerr=sig[xi>0], color='blue', lw=0.1, ls='')
plt.errorbar(r[xi<0], -xi[xi<0], yerr=sig[xi<0], color='blue', lw=0.1, ls='')
leg = plt.errorbar(-r, xi, yerr=sig, color='blue')

plt.xscale('log')
#plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='upper left')
plt.xlim([2.5, 200])
plt.ylim([0, 2.5])
plt.show()

file_name_z1 = '/Users/megantabbutt/Cosmology/RedMagic/y1_redshift_distributions_v1.fits'
data__z1 = fitsio.read(file_name_z1)
print(data__z1.dtype)  #
print(data__z1.shape)  #



file_name_z2 = '/Users/megantabbutt/Cosmology/RedMagic/y1_source_redshift_binning_v1.fits'
data__z2 = fitsio.read(file_name_z2)
print(data__z2.dtype)  #
print(data__z2.shape)  #



