#!/usr/bin/env python
# coding: utf-8

# This notebook will calculate the Count-Count Auto-Correlations for PanSTARRS and SDSS EBOSS CMASS/LOWZ data sets
# This notebook is a conversion of the jupyter notebook - computation is the same but the code is tweaked for Python
# V0    MT      4/29/20

# FIXES:
#
# Pull in the database files and parse, don't need to remake the .db files, have a parsing notebook just for hadelling
# the data and then run once on a new machine. -- SHORT TEM FIX: Assume databases are in the folder, still need to
# make the parsing notebook that actually puts them there if they aren't already...

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Imports:

import time
import treecorr
#import fitsio
import numpy
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import astropy
from astropy.coordinates import SkyCoord
import astropy.units as u
import sqlite3
from astropy.table import Table
from matplotlib.patches import Circle

# If need to import basemaps, going to be a nightmare... thread lightly
# CCL could also be an issue... Need to resolve eventually, works on HEP in pyCCL conda environment and on LSST_Stack


#//////////////////////////////////////////// THINGS TO CHANGE EACH TIME ///////////////////////////////////////////////

#dataPath = '/Users/megantabbutt/CosmologyDataProducts/'
dataPath = '/afs/hep.wisc.edu/home/tabbutt/private/CosmologyDataProducts/'

# Change this for HEP for more rands
randsLength = 10**7

# Figures generated will go to this file
dateName = 'Apr_30_20_6pm/' # need to make this...

#saveFigFolder = '/Users/megantabbutt/Cosmology/Cosmology/SNe CrossCorrelations/figures/' + dateName
saveFigFolder =  '/afs/hep.wisc.edu/home/tabbutt/public/Cosmology/SNe CrossCorrelations/figures/' + dateName


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Define notebook wide functions and data paths to use:

'''
Convert from PanSTARRS data where RA is in "u.hourangle" and DEC is in "u.deg" to just degrees
 @Param Dataframe     is the panstarrs dataframe to convert
 @Param newDataFrame  is the new data frame you will fill in 
 --- FIX: could be improved to be more versitile
'''
def getRADecFromHourAngles(Dataframe, newDataFrame):
    for i, row in Dataframe.iterrows():
        Coords = SkyCoord(PanSTARRS['RA'][i], PanSTARRS['Dec'][i], unit=(u.hourangle, u.deg))
        newDataFrame['ID'][i] = row['ID']
        newDataFrame['RA'][i] = Coords.ra.degree
        newDataFrame['DEC'][i] = Coords.dec.degree 
        newDataFrame['zSN'][i] = row['zSN']
        newDataFrame['zHost'][i] = row['zHost']


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 1. Parse db files and query to get good objects:


# Open a SQL Connection and pull out SNe data that has a good z for itsself or its host

connPAN = sqlite3.connect(dataPath + 'PanSTARRS.db')

#PanSTARRSNEW.to_sql("PanSTARRSNEW", con=connPAN) # ADD TO PARSING NOTEBOOK -- FIX
qry = "SELECT ID, DEC, RA, zSN, zHost FROM PanSTARRSNEW WHERE (zSN > -999) || (zHost > -999)"
PanSTARRSNEW_GoodZ = pd.read_sql(qry, con=connPAN)
print("Good z PanSTARRS data: 1129 objects")
print(PanSTARRSNEW_GoodZ.head(3)) # 1129 objects over 10 pointings


# Open a SQL connection to union the CMASS/LOWZ data sets together:

connBOSS = sqlite3.connect(dataPath + 'CMASS_and_LOWZ.db')
#CMASSLOWZTOT_South_DF.to_sql("CMASSLOWZTOT_South", con=connBOSS) # Execute these if .db doesn't exist yet
#CMASSLOWZTOT_North_DF.to_sql("CMASSLOWZTOT_North", con=connBOSS) # Do one at a time to make sure all is good
qry = "SELECT * FROM CMASSLOWZTOT_South UNION SELECT * FROM CMASSLOWZTOT_North"
CMASSLOWZTOT_DF = pd.read_sql(qry, con=connBOSS)
print("CMASS adn LOWZ data set unioned: 1.3M objects")
print(CMASSLOWZTOT_DF.head(3)) # 1.3 million objects


# Open a SQL connection to union the CMASS/LOWZ RANDOMS data sets together:

print("starting to pull in CMASS randoms data with query 10,000")
startTime = time.time()
connBOSSRands = sqlite3.connect(dataPath + 'CMASS_and_LOWZ_rands.db')
#CMASSLOWZTOT_South_rand_DF.to_sql("CMASSLOWZTOT_South_rands", con=connBOSSRands) # ADD TO PARSING NOTEBOOK
#CMASSLOWZTOT_North_rand_DF.to_sql("CMASSLOWZTOT_North_rands", con=connBOSSRands) # ADD TO PARSING NOTEBOOK
# NOTE: index is a SQL keyword... BLAH
randSampleQry = "SELECT * FROM CMASSLOWZTOT_South_rands WHERE `index` IN (SELECT `index` FROM CMASSLOWZTOT_South_rands ORDER BY RANDOM() LIMIT 500000) UNION SELECT * FROM CMASSLOWZTOT_North_rands WHERE `index` IN (SELECT `index` FROM CMASSLOWZTOT_North_rands ORDER BY RANDOM() LIMIT 500000)"
randQry = "SELECT * FROM CMASSLOWZTOT_South_rands UNION SELECT * FROM CMASSLOWZTOT_North_rands"
CMASSLOWZTOT_DF_rands = pd.read_sql(randSampleQry, con=connBOSSRands)
CMASSLOWZTOT_DF_rands.to_json(dataPath + "CMASSLOWZTOT_DF_rands")
print("CMASS adn LOWZ randoms data set opened: 20k/___ objects")
print(CMASSLOWZTOT_DF_rands.head(3))
print("--- %s seconds ---" % (time.time() - startTime))


# CLOSE THE CONNECTIONS ASAP:

connBOSS.close()
connBOSSRands.close()
print("completed sqlite and closed connections. ")



#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 2. Create the TreeCorr Catalogs:


catPanSTARRS = treecorr.Catalog(ra=PanSTARRSNEW_GoodZ['RA'], dec=PanSTARRSNEW_GoodZ['DEC'], ra_units='degrees', dec_units='degrees')
print("TreeCorr PanSTARRS Catalog:")
print(catPanSTARRS)


# Count-Count Correlation Function:
# Data Auto-correlation: (dd)
ddPanSTARRS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
ddPanSTARRS.process(catPanSTARRS)


# Create the randoms with PanSTARRS since no mask yet
# Include all ten pointings for now, can redo when we are going to crossCorr with CMASS

ra_min_PanSTARRS = numpy.min(catPanSTARRS.ra)
ra_max_PanSTARRS = numpy.max(catPanSTARRS.ra)
dec_min_PanSTARRS = numpy.min(catPanSTARRS.dec)
dec_max_PanSTARRS = numpy.max(catPanSTARRS.dec)
print('PanSTARRS ra range = %f .. %f' % (ra_min_PanSTARRS, ra_max_PanSTARRS))
print('PanSTARRS dec range = %f .. %f' % (dec_min_PanSTARRS, dec_max_PanSTARRS))

print("making the randoms in the sky")
startTime = time.time()
rand_ra_PanSTARRS = numpy.random.uniform(ra_min_PanSTARRS, ra_max_PanSTARRS, randsLength)
rand_sindec_PanSTARRS = numpy.random.uniform(numpy.sin(dec_min_PanSTARRS), numpy.sin(dec_max_PanSTARRS), randsLength)
rand_dec_PanSTARRS = numpy.arcsin(rand_sindec_PanSTARRS)
print("--- %s seconds ---" % (time.time() - startTime))

# MD02 is the one that needs to be eliminated, not in CMASS footprint
pointings = {"MD01": [035.875, -04.250], "MD03": [130.592, 44.317], "MD04": [150.000, 02.200], 
             "MD05": [161.917, 58.083], "MD06": [185.000, 47.117], "MD07": [213.704, 53.083], 
             "MD08": [242.787, 54.950], "MD09": [334.188, 00.283], "MD10": [352.312, -00.433], "MD02": [053.100, -27.800],}


# Check that the randoms cover the same space as the data
f1, (ax1a, ax2a, ax3a) = plt.subplots(1, 3, figsize=(20, 5))

ax1a.scatter(catPanSTARRS.ra * 180/numpy.pi, catPanSTARRS.dec * 180/numpy.pi, color='red', s=0.1, marker='x')
ax1a.scatter(rand_ra_PanSTARRS * 180/numpy.pi, rand_dec_PanSTARRS * 180/numpy.pi, color='blue', s=0.1)
ax1a.set_xlabel('RA (degrees)')
ax1a.set_ylabel('Dec (degrees)')
ax1a.set_title('Randoms on top of data')

# Repeat in the opposite order
ax2a.scatter(rand_ra_PanSTARRS * 180/numpy.pi, rand_dec_PanSTARRS * 180/numpy.pi, color='blue', s=0.1, marker='x')
ax2a.scatter(catPanSTARRS.ra * 180/numpy.pi, catPanSTARRS.dec * 180/numpy.pi, color='red', s=0.1)
ax2a.set_xlabel('RA (degrees)')
ax2a.set_ylabel('Dec (degrees)')
ax2a.set_title('Data on top of randoms')

# Zoom to look at coverage of randoms and reals
ax3a.scatter(rand_ra_PanSTARRS * 180/numpy.pi, rand_dec_PanSTARRS * 180/numpy.pi, color='blue', s=1, marker='x', label='rands')
ax3a.scatter(catPanSTARRS.ra * 180/numpy.pi, catPanSTARRS.dec * 180/numpy.pi, color='red', s=1, label='data')
ax3a.set_xlabel('RA (degrees)')
ax3a.set_ylabel('Dec (degrees)')
ax3a.set_title('Data on top of randoms_Zoom')
ax3a.legend(loc = "upper right")
ax3a.set_xlim(129, 133)
ax3a.set_ylim(42, 46)

plt.savefig(saveFigFolder + 'PanSTARRS_data_rands_overlap')
plt.close()


# Need to make the mask for the randoms in PanSTARRS:
maskRA = []
maskDEC = []

print("sorting the randoms into pointings:")
startTime = time.time()
for pointing in pointings: 
    maskRAprevious = len(maskRA)
    X0 = pointings[pointing][0]
    Y0 = pointings[pointing][1]
    rad = 3.3/2
    print("Pointing coords:")
    print(pointings[pointing])
    
    for i in range(len(rand_ra_PanSTARRS)):
        #print(rand_ra_PanSTARRS[i], rand_dec_PanSTARRS[i])
        X = rand_ra_PanSTARRS[i] * 180 / numpy.pi
        Y = rand_dec_PanSTARRS[i] * 180 / numpy.pi
        
        if ((X - X0)**2 + (Y - Y0)**2 < rad**2):
            maskRA.append(X)
            maskDEC.append(Y)
    print("Number of randoms in Pointing: ")
    print(len(maskRA) - maskRAprevious)
    print(" ")

print("--- %s seconds ---" % (time.time() - startTime))


# Check that the randoms cover the same space as the data
f2, (ax1b, ax2b, ax3b) = plt.subplots(1, 3, figsize=(20,5))

ax1b.scatter(catPanSTARRS.ra * 180/numpy.pi, catPanSTARRS.dec * 180/numpy.pi, color='red', s=0.1, marker='x')
ax1b.scatter(maskRA, maskDEC, color='blue', s=0.1)
ax1b.set_xlabel('RA (degrees)')
ax1b.set_ylabel('Dec (degrees)')
ax1b.set_title('Randoms on top of data with Mask')

# Repeat in the opposite order
ax2b.scatter(maskRA, maskDEC, color='blue', s=0.1)
ax2b.scatter(catPanSTARRS.ra * 180/numpy.pi, catPanSTARRS.dec * 180/numpy.pi, color='red', s=0.1, marker='x')
ax2b.set_xlabel('RA (degrees)')
ax2b.set_ylabel('Dec (degrees)')
ax2b.set_title('Data on top of randoms with Mask')

# Zoom to look at coverage of randoms and reals
ax3b.scatter(maskRA, maskDEC, color='blue', s=1, marker='x', label='rands_mask')
ax3b.scatter(catPanSTARRS.ra * 180/numpy.pi, catPanSTARRS.dec * 180/numpy.pi, color='red', s=1, label='data')
ax3b.set_xlabel('RA (degrees)')
ax3b.set_ylabel('Dec (degrees)')
ax3b.set_title('Data on top of randoms with mask_Zoom')
ax3b.legend(loc = "upper right")
ax3b.set_xlim(128, 133)
ax3b.set_ylim(42, 47)

plt.savefig(saveFigFolder + 'PanSTARRS_data_rands_overlap_mask')
#plt.show()
plt.close()

# make Random catalog with mask
rand = treecorr.Catalog(ra=maskRA, dec=maskDEC, ra_units='degrees', dec_units='degrees')
rr = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rr.process(rand)

dr = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
dr.process(catPanSTARRS, rand)


# Plot Landy-Sca
r = numpy.exp(ddPanSTARRS.meanlogr)
xi, varxi = ddPanSTARRS.calculateXi(rr, dr)
sig = numpy.sqrt(varxi)

plt.plot(r, xi, color='blue')
plt.plot(r, -xi, color='blue', ls=':')
plt.errorbar(r[xi>0], xi[xi>0], yerr=sig[xi>0], color='green', lw=0.5, ls='')
plt.errorbar(r[xi<0], -xi[xi<0], yerr=sig[xi<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-r, xi, yerr=sig, color='blue')

plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
plt.title("Count-Count Auto Corr Function for PanSTARRS")
plt.xlim([0.01,10])
plt.savefig(saveFigFolder + 'PanSTARRS_NNautoCorr')
#plt.show()
plt.close()


# ## 2.  CMASS Count-Count Auto Correlation Function:

catCMASS = treecorr.Catalog(ra=CMASSLOWZTOT_DF['RA'], dec=CMASSLOWZTOT_DF['DEC'], ra_units='degrees', dec_units='degrees')
print("TreeCorr CMASS/LOWZ Catalog:")
print(catCMASS)

# Data Auto-correlation: (dd)
ddCMASS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
ddCMASS.process(catCMASS)
CMASSLOWZTOT_DF_rands.head(3)

# Check that the randoms cover the same space as the data
f3, (ax1c, ax2c) = plt.subplots(1, 2, figsize=(20, 5))

ax1c.scatter(catCMASS.ra * 180/numpy.pi, catCMASS.dec * 180/numpy.pi, color='red', s=0.1)
ax1c.set_xlabel('RA (degrees)')
ax1c.set_ylabel('Dec (degrees)')
ax1c.set_title('CMASS/LOWZ Data')

# Repeat in the opposite order
ax2c.scatter(CMASSLOWZTOT_DF_rands['RA'], CMASSLOWZTOT_DF_rands['DEC'], color='blue', s=0.1)
ax2c.set_xlabel('RA (degrees)')
ax2c.set_ylabel('Dec (degrees)')
ax2c.set_title('CMASS/LOWZ Randoms')
plt.savefig(saveFigFolder + 'CMASS_data_rands')
#plt.show()
plt.close()


randCMASS = treecorr.Catalog(ra=CMASSLOWZTOT_DF_rands['RA'], dec=CMASSLOWZTOT_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')
rrCMASS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rrCMASS.process(randCMASS)

drCMASS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
drCMASS.process(catCMASS, randCMASS)

rCMASS = numpy.exp(ddCMASS.meanlogr)
xiCMASS, varxiCMASS = ddCMASS.calculateXi(rrCMASS, drCMASS)
sigCMASS = numpy.sqrt(varxiCMASS)

# plot the LS count-count auto corr function for CMASS and rands:
plt.plot(rCMASS, xiCMASS, color='blue')
plt.plot(rCMASS, -xiCMASS, color='blue', ls=':')
plt.errorbar(rCMASS[xiCMASS>0], xiCMASS[xiCMASS>0], yerr=sigCMASS[xiCMASS>0], color='green', lw=0.5, ls='')
plt.errorbar(rCMASS[xiCMASS<0], -xiCMASS[xiCMASS<0], yerr=sigCMASS[xiCMASS<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-rCMASS, xiCMASS, yerr=sigCMASS, color='blue')

plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
plt.xlim([0.01,10])
plt.title("Count-Count Auto Corr Function for CMASS")
plt.savefig(saveFigFolder + 'CMASS_NNautoCorr')
#plt.show()
plt.close()

print("Program is done.")
