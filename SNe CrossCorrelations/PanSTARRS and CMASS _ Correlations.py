#!/usr/bin/env python
# coding: utf-8

# I DONT WORK ON WINDOWS... REWRITE WITH .os.join things if doing that...

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         Imports and formatting:

#plt.switch_backend('agg') #For HEP, matplotlib x windows issues see python version for more usage 
import treecorr
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import os
import datetime



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         Define notebook wide functions and data paths to use:

# Define the paths for local and HEP machines:
DATA_PATH = '/Users/megantabbutt/CosmologyDataProducts/'
#dataPath = '/afs/hep.wisc.edu/home/tabbutt/private/CosmologyDataProducts/'

TESTING_PRODUCTS_PATH = "/Users/megantabbutt/Cosmology/Cosmology/SNe CrossCorrelations/VerificationTestingProducts/"
# Add HEP testing path

# Create the directory to save to and a file with info about this run:
DATE = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
CURRENT_DIRECTORY = DATE
TESTING_PRODUCTS_PATH = TESTING_PRODUCTS_PATH + CURRENT_DIRECTORY

os.mkdir(TESTING_PRODUCTS_PATH)

NOTES_NAME = "/RUNNING_NOTES_" + DATE + ".txt"
NOTES_PATH = TESTING_PRODUCTS_PATH + NOTES_NAME

# Write an opening note in the file:
NOTES = open(NOTES_PATH, "a")
NOTES.write("Created Running notes file for tracking details about this run and products produced/saved")
NOTES.write("\n")
NOTES.write("\n")
NOTES.close()


"""
# 
# ## 0. Pull in and parse data:
# 
# note: There are 10 pointings for the PanSTARRS data, we will use all 10 for the Auto Correlation, but when we correlated to CMASS, we need to only use the 9 overlap with CMASS. --- IMPORTANT

# #### PanSTARRS:

connPAN = sqlite3.connect(dataPath + 'PanSTARRS.db')

qry = "SELECT ID, DEC, RA, zSN, zHost FROM PanSTARRSNEW WHERE (zSN > -999) || (zHost > -999)"

PanSTARRSNEW_GoodZ = pd.read_sql(qry, con=connPAN)
PanSTARRSNEW_GoodZ.head(3) # 1129 objects over 10 pointings 


# #### CMASS/LOWZ:

# In[5]:


connBOSS = sqlite3.connect(dataPath + 'CMASS_and_LOWZ.db')

qry = "SELECT * FROM CMASSLOWZTOT_South UNION SELECT * FROM CMASSLOWZTOT_North"

CMASSLOWZTOT_DF = pd.read_sql(qry, con=connBOSS)
CMASSLOWZTOT_DF.head(3) # 1.3 million objects


# #### Pull in the Randoms provided by CMASS:

# In[6]:


connBOSSRands = sqlite3.connect(dataPath + 'CMASS_and_LOWZ_rands.db')
randSampleQry = "SELECT * FROM CMASSLOWZTOT_South_rands WHERE `index` IN (SELECT `index` FROM CMASSLOWZTOT_South_rands ORDER BY RANDOM() LIMIT 500000) UNION SELECT * FROM CMASSLOWZTOT_North_rands WHERE `index` IN (SELECT `index` FROM CMASSLOWZTOT_North_rands ORDER BY RANDOM() LIMIT 500000)"
randQry = "SELECT * FROM CMASSLOWZTOT_South_rands UNION SELECT * FROM CMASSLOWZTOT_North_rands"

CMASSLOWZTOT_DF_rands = pd.read_sql(randSampleQry, con=connBOSSRands)
CMASSLOWZTOT_DF_rands.to_json(dataPath + "CMASSLOWZTOT_DF_rands")
CMASSLOWZTOT_DF_rands.head(3)


# In[7]:


connBOSS.close()
connBOSSRands.close()


#  <hr style="height:3px"> 

# ## 1. Create the TreeCorr Catalogs of Data:

# A set of input data (positions and other quantities) to be correlated.
# 
# A Catalog object keeps track of the relevant information for a number of objects to be correlated. The objects each have some kind of position (for instance (x,y), (ra,dec), (x,y,z), etc.), and possibly some extra information such as weights (w), shear values (g1,g2), or kappa values (k).
# 
# The simplest way to build a Catalog is to simply pass in numpy arrays for each piece of information you want included. 
# 
# > cat = treecorr.Catalog(ra=ra, dec=dec, g1=g1, g2=g2, ra_units='hour', dec_units='deg')
# 
# Other options for reading in from a file, using a config file, etc

# In[8]:


catPanSTARRS = treecorr.Catalog(ra=PanSTARRSNEW_GoodZ['RA'], dec=PanSTARRSNEW_GoodZ['DEC'], ra_units='degrees', dec_units='degrees')
catPanSTARRS


# ## 2. Create the randoms for PanSTARRS
# Include all ten pointings for now, can just exclude the pointing that isn't in CMASS when doing the CrossCorr </br>
# 
# Possibility to ask for mask eventually if we think that it is a limitation </br>

# In[9]:


# Change this for more and less, 10E5 good for personal laptop ~5min run time
randsLength = 10**6


# In[10]:


ra_min_PanSTARRS = numpy.min(catPanSTARRS.ra)
ra_max_PanSTARRS = numpy.max(catPanSTARRS.ra)
dec_min_PanSTARRS = numpy.min(catPanSTARRS.dec)
dec_max_PanSTARRS = numpy.max(catPanSTARRS.dec)
print('PanSTARRS ra range = %f .. %f' % (ra_min_PanSTARRS, ra_max_PanSTARRS))
print('PanSTARRS dec range = %f .. %f' % (dec_min_PanSTARRS, dec_max_PanSTARRS))

rand_ra_PanSTARRS = numpy.random.uniform(ra_min_PanSTARRS, ra_max_PanSTARRS, randsLength)
rand_sindec_PanSTARRS = numpy.random.uniform(numpy.sin(dec_min_PanSTARRS), numpy.sin(dec_max_PanSTARRS), randsLength)
rand_dec_PanSTARRS = numpy.arcsin(rand_sindec_PanSTARRS)


# #### Note: MD02 is the one that needs to be eliminated, not in CMASS footprint 

# In[11]:


# Got from a paper, need to cite it here:  https://arxiv.org/pdf/1612.05560.pdf

pointings = {"MD01": [035.875, -04.250], "MD03": [130.592, 44.317], "MD04": [150.000, 02.200], 
             "MD05": [161.917, 58.083], "MD06": [185.000, 47.117], "MD07": [213.704, 53.083], 
             "MD08": [242.787, 54.950], "MD09": [334.188, 00.283], "MD10": [352.312, -00.433], "MD02": [053.100, -27.800],}


# In[12]:


# Check how well the randoms cover the same space as the data

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

plt.show()


# "The telescope illuminates a diameter of 3.3 degrees,  with low distortion, and mild vignetting at the edge of this illuminated region. The field of view is approximately 7 square degrees. The 8  meter  focal  length  atf/4.4  gives  an  approximate  10micron pixel scale of 0.258 arcsec/pixel."
# 
# 7 square degrees --> r = 1.49 deg

# In[13]:


# Make a mask that consists of the ten pointings populated with the randoms that are in it...
# Def a better way to do this? 

maskRA = []
maskDEC = []

for pointing in pointings: 
    maskRAprevious = len(maskRA)
    X0 = pointings[pointing][0]
    Y0 = pointings[pointing][1]
    rad = 3.3/2
    print(pointings[pointing])
    
    for i in range(len(rand_ra_PanSTARRS)):
        #print(rand_ra_PanSTARRS[i], rand_dec_PanSTARRS[i])
        X = rand_ra_PanSTARRS[i] * 180 / numpy.pi
        Y = rand_dec_PanSTARRS[i] * 180 / numpy.pi
        
        if ((X - X0)**2 + (Y - Y0)**2 < rad**2):
            maskRA.append(X)
            maskDEC.append(Y)
    print(len(maskRA) - maskRAprevious)


# In[14]:


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

plt.show()


# ## 3. Make PanSTARRS Count-Count Auto Correlation Functions:
# 
# Typical Usage Pattern:
# 
# > nn = treecorr.NNCorrelation(config) 
# <br>
# nn.process(cat)     # For auto-correlation.
# <br>
# nn.process(cat1,cat2)   # For cross-correlation.
# <br>
# rr.process...           # Likewise for random-random correlations
# <br>
# dr.process...        # If desired, also do data-random correlations
# <br>
# rd.process...    # For cross-correlations, also do the reverse.
# <br>
# nn.write(file_name,rr,dr,rd)  # Write out to a file.
# <br>
# xi,varxi = nn.calculateXi(rr,dr,rd)  # Or get the correlation function directly.

# In[15]:


# Data Auto-correlation: (dd)
ddPanSTARRS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
ddPanSTARRS.process(catPanSTARRS)


# In[16]:


rand = treecorr.Catalog(ra=maskRA, dec=maskDEC, ra_units='degrees', dec_units='degrees')
rr = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rr.process(rand)


# In[17]:


dr = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
dr.process(catPanSTARRS, rand)


# In[18]:


r = numpy.exp(ddPanSTARRS.meanlogr)
xi, varxi = ddPanSTARRS.calculateXi(rr, dr)
sig = numpy.sqrt(varxi)


# In[19]:


# Plot the Correlation function:

plt.plot(r, xi, color='blue')
plt.plot(r, -xi, color='blue', ls=':')
plt.errorbar(r[xi>0], xi[xi>0], yerr=sig[xi>0], color='green', lw=0.5, ls='')
plt.errorbar(r[xi<0], -xi[xi<0], yerr=sig[xi<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-r, xi, yerr=sig, color='blue')

plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
plt.xlim([0.01,10])
plt.show()


# ## 4. Make CMASS Count-Count Auto Correlation Functions:
# 
# Typical Usage Pattern:
# 
# > nn = treecorr.NNCorrelation(config) 
# <br>
# nn.process(cat)     # For auto-correlation.
# <br>
# nn.process(cat1,cat2)   # For cross-correlation.
# <br>
# rr.process...           # Likewise for random-random correlations
# <br>
# dr.process...        # If desired, also do data-random correlations
# <br>
# rd.process...    # For cross-correlations, also do the reverse.
# <br>
# nn.write(file_name,rr,dr,rd)  # Write out to a file.
# <br>
# xi,varxi = nn.calculateXi(rr,dr,rd)  # Or get the correlation function directly.

# In[20]:


catCMASS = treecorr.Catalog(ra=CMASSLOWZTOT_DF['RA'], dec=CMASSLOWZTOT_DF['DEC'], 
                                ra_units='degrees', dec_units='degrees')
catCMASS


# In[21]:


# Data Auto-correlation: (dd)
ddCMASS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
ddCMASS.process(catCMASS)


# In[22]:


randCMASS = treecorr.Catalog(ra=CMASSLOWZTOT_DF_rands['RA'], dec=CMASSLOWZTOT_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')
rrCMASS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rrCMASS.process(randCMASS)


# In[23]:


drCMASS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
drCMASS.process(catCMASS, randCMASS)


# In[24]:


rCMASS = numpy.exp(ddCMASS.meanlogr)
xiCMASS, varxiCMASS = ddCMASS.calculateXi(rrCMASS, drCMASS)
sigCMASS = numpy.sqrt(varxiCMASS)


# In[25]:


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

plt.show()


# In[26]:


# Plot the autocorrelation function: 

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
plt.show()


# ## 5. Analyze if the plots are correct: 
# 
# The CMASS plot should look like this paper's figure 1: </br>
# 
# https://arxiv.org/pdf/1607.03144.pdf
# 
# 

# In[27]:


rCMASS = numpy.exp(ddCMASS.meanlogr)
xiCMASS, varxiCMASS = ddCMASS.calculateXi(rrCMASS, drCMASS)
sigCMASS = numpy.sqrt(varxiCMASS)

plt.plot(rCMASS, xiCMASS, color='blue')
plt.plot(rCMASS, -xiCMASS, color='blue', ls=':')
#plt.errorbar(rCMASS[xiCMASS>0], xiCMASS[xiCMASS>0], yerr=sigCMASS[xiCMASS>0], color='green', lw=0.5, ls='')
#plt.errorbar(rCMASS[xiCMASS<0], -xiCMASS[xiCMASS<0], yerr=sigCMASS[xiCMASS<0], color='green', lw=0.5, ls='')
#leg = plt.errorbar(-rCMASS, xiCMASS, yerr=sigCMASS, color='blue')

#plt.xscale('log')
#plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

#plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
plt.xlim([0.0, 7.5])
plt.ylim([0.0, .4])
plt.show()


# My plot vs Theirs:

# In[28]:


Image("/Users/megantabbutt/Desktop/ObservationalCosmology/SNeProject/Data_pictures_informal/5_14_20/plot1.jpeg")


# ### Cross Corr the CMASS and LOWZ randoms:
# 
# - Do autoCorr function and choose CMASS to be the data, and LOWZ to be the rands

# In[31]:


connCMASSRands = sqlite3.connect(dataPath + 'CMASS_rands.db')
randSampleQry = "SELECT * FROM CMASS_South_rands WHERE `index` IN (SELECT `index` FROM CMASS_South_rands ORDER BY RANDOM() LIMIT 500000) UNION SELECT * FROM CMASS_North_rands WHERE `index` IN (SELECT `index` FROM CMASS_North_rands ORDER BY RANDOM() LIMIT 500000)"
randQry = "SELECT * FROM CMASS_South_rands UNION SELECT * FROM CMASS_North_rands"

CMASS_DF_rands = pd.read_sql(randSampleQry, con=connCMASSRands)
CMASS_DF_rands.to_json(dataPath + "CMASS_DF_rands")
CMASS_DF_rands.head(3)


# In[32]:


connLOWZRands = sqlite3.connect(dataPath + 'LOWZ_rands.db')
randSampleQry = "SELECT * FROM LOWZ_South_rands WHERE `index` IN (SELECT `index` FROM LOWZ_South_rands ORDER BY RANDOM() LIMIT 500000) UNION SELECT * FROM LOWZ_North_rands WHERE `index` IN (SELECT `index` FROM LOWZ_North_rands ORDER BY RANDOM() LIMIT 500000)"
randQry = "SELECT * FROM LOWZ_South_rands UNION SELECT * FROM LOWZ_North_rands"

LOWZ_DF_rands = pd.read_sql(randSampleQry, con=connLOWZRands)
LOWZ_DF_rands.to_json(dataPath + "LOWZ_DF_rands")
LOWZ_DF_rands.head(3)


# In[33]:


catCMASSrands = treecorr.Catalog(ra=CMASS_DF_rands['RA'], dec=CMASS_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')
catCMASSrands


# In[34]:


ddCMASSrands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
ddCMASSrands.process(catCMASSrands)
ddCMASSrands


# In[35]:


randCMASSrands = treecorr.Catalog(ra=LOWZ_DF_rands['RA'], dec=LOWZ_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')
rrCMASSrands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rrCMASSrands.process(randCMASSrands)
rrCMASSrands


# In[36]:


drCMASSrands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
drCMASSrands.process(catCMASSrands, randCMASSrands)
drCMASSrands


# In[37]:


rCMASSrands = numpy.exp(ddCMASSrands.meanlogr)
xiCMASSrands, varxiCMASSrands = ddCMASSrands.calculateXi(rrCMASSrands, drCMASSrands)
sigCMASSrands = numpy.sqrt(varxiCMASSrands)
xiCMASSrands
#varxiCMASSrands


# In[38]:


# Check that the randoms cover the same space as the data
f4, (ax1d, ax2d) = plt.subplots(1, 2, figsize=(20, 5))

ax1d.scatter(catCMASSrands.ra * 180/numpy.pi, catCMASSrands.dec * 180/numpy.pi, color='blue', s=0.1)
ax1d.set_xlabel('RA (degrees)')
ax1d.set_ylabel('Dec (degrees)')
ax1d.set_title('CMASS Randoms')

# Repeat in the opposite order
ax2d.scatter(randCMASSrands.ra * 180/numpy.pi, randCMASSrands.dec * 180/numpy.pi, color='green', s=0.1)
ax2d.set_xlabel('RA (degrees)')
ax2d.set_ylabel('Dec (degrees)')
ax2d.set_title('LOWZ Randoms')

#plt.savefig(testingProductsPath + 'CMASS_LOWZ_randoms_skyplot')
plt.show()


# In[39]:


# Plot the autocorrelation function: 

plt.plot(rCMASSrands, xiCMASSrands, color='blue')
plt.plot(rCMASSrands, -xiCMASSrands, color='blue', ls=':')
plt.errorbar(rCMASSrands[xiCMASSrands>0], xiCMASSrands[xiCMASSrands>0], yerr=sigCMASSrands[xiCMASSrands>0], color='green', lw=0.5, ls='')
plt.errorbar(rCMASSrands[xiCMASSrands<0], -xiCMASSrands[xiCMASSrands<0], yerr=sigCMASSrands[xiCMASSrands<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-rCMASSrands, xiCMASSrands, yerr=sigCMASSrands, color='blue')

plt.xscale('log')
#plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
#plt.xlim([0.01,10])
#plt.ylim([0.0, .00001])
plt.title("Count-Count Auto Corr Function for CMASS rands as data and LOWZ rands as rands")

#plt.savefig(testingProductsPath + 'NN_Auto_CMASS_Rands')
plt.show()


# In[40]:


# Plot the autocorrelation function: 

plt.plot(rCMASSrands, xiCMASSrands, color='blue')
plt.plot(rCMASSrands, -xiCMASSrands, color='blue', ls=':')
plt.errorbar(rCMASSrands[xiCMASSrands>0], xiCMASSrands[xiCMASSrands>0], yerr=sigCMASSrands[xiCMASSrands>0], color='green', lw=0.5, ls='')
plt.errorbar(rCMASSrands[xiCMASSrands<0], -xiCMASSrands[xiCMASSrands<0], yerr=sigCMASSrands[xiCMASSrands<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-rCMASSrands, xiCMASSrands, yerr=sigCMASSrands, color='blue')

plt.xscale('log')
#plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
#plt.xlim([0.01,10])
plt.ylim([-.4, .4])
plt.title("Count-Count Auto Corr Function for CMASS rands as data and LOWZ rands as rands")
plt.savefig(testingProductsPath + 'NN_Auto_CMASS_Rands_zoom')
plt.show()


# ### AutoCorr the CMASS and LOWZ randoms:
# 
# - Do autoCorr function and choose LOWZ to be the data, and CMASS to be the rands

# In[41]:


catLOWZrands = treecorr.Catalog(ra=LOWZ_DF_rands['RA'], dec=LOWZ_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')
catLOWZrands


# In[42]:


ddLOWZrands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
ddLOWZrands.process(catLOWZrands)
ddLOWZrands


# In[43]:


randLOWZrands = treecorr.Catalog(ra=CMASS_DF_rands['RA'], dec=CMASS_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')
rrLOWZrands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rrLOWZrands.process(randLOWZrands)
rrLOWZrands


# In[44]:


drLOWZrands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
drLOWZrands.process(catLOWZrands, randLOWZrands)
drLOWZrands


# In[45]:


rLOWZrands = numpy.exp(ddLOWZrands.meanlogr)
xiLOWZrands, varxiLOWZrands = ddLOWZrands.calculateXi(rrLOWZrands, drLOWZrands)
sigLOWZrands = numpy.sqrt(varxiLOWZrands)
xiLOWZrands
#varxiCMASSrands


# In[46]:


# Plot the autocorrelation function: 

plt.plot(rLOWZrands, xiLOWZrands, color='blue')
plt.plot(rLOWZrands, -xiLOWZrands, color='blue', ls=':')
plt.errorbar(rLOWZrands[xiLOWZrands>0], xiLOWZrands[xiLOWZrands>0], yerr=sigLOWZrands[xiLOWZrands>0], color='green', lw=0.5, ls='')
plt.errorbar(rLOWZrands[xiLOWZrands<0], -xiLOWZrands[xiLOWZrands<0], yerr=sigLOWZrands[xiLOWZrands<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-rLOWZrands, xiLOWZrands, yerr=sigLOWZrands, color='blue')

plt.xscale('log')
#plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
#plt.xlim([0.01,10])
#plt.ylim([0.0, .00001])
plt.title("Count-Count Auto Corr Function for LOWZ rands as data and CMASS rands as rands")
#plt.savefig(testingProductsPath + 'NN_Auto_LOWZ_Rands')
plt.show()


# In[47]:


# Plot the autocorrelation function: 

plt.plot(rLOWZrands, xiLOWZrands, color='blue')
plt.plot(rLOWZrands, -xiLOWZrands, color='blue', ls=':')
plt.errorbar(rLOWZrands[xiLOWZrands>0], xiLOWZrands[xiLOWZrands>0], yerr=sigLOWZrands[xiLOWZrands>0], color='green', lw=0.5, ls='')
plt.errorbar(rLOWZrands[xiLOWZrands<0], -xiLOWZrands[xiLOWZrands<0], yerr=sigLOWZrands[xiLOWZrands<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-rLOWZrands, xiLOWZrands, yerr=sigLOWZrands, color='blue')

plt.xscale('log')
#plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
#plt.xlim([0.01,10])
plt.ylim([-0.4, .4])
plt.title("Count-Count Auto Corr Function for LOWZ rands as data and CMASS rands as rands")
#plt.savefig(testingProductsPath + 'NN_Auto_LOWZ_Rands_zoom')
plt.show()


# ### AutoCorr the CMASS and LOWZ data sets seperatly with their individual randoms to check for asymptote
# 

# In[48]:


# come back and do this later


# ## 6. Cross Correlate the eBOSS and PanSTARRS sets
# 

# In[49]:


# Need to get just 9 pointings from PanSTARRS: 

maskRA_overlap = []
maskDEC_overlap = []

for pointing in pointings:
    if(pointing == "MD02"):
        continue
    else:
        maskRAprevious = len(maskRA_overlap)
        X0 = pointings[pointing][0]
        Y0 = pointings[pointing][1]
        rad = 3.3/2
        print(pointings[pointing])

        for i in range(len(rand_ra_PanSTARRS)):
            #print(rand_ra_PanSTARRS[i], rand_dec_PanSTARRS[i])
            X = rand_ra_PanSTARRS[i] * 180 / numpy.pi
            Y = rand_dec_PanSTARRS[i] * 180 / numpy.pi

            if ((X - X0)**2 + (Y - Y0)**2 < rad**2):
                maskRA_overlap.append(X)
                maskDEC_overlap.append(Y)
        print(len(maskRA_overlap) - maskRAprevious)


# In[50]:


f2, (ax1b, ax2b) = plt.subplots(1, 2, figsize=(20,5))

ax1b.scatter(catPanSTARRS.ra * 180/numpy.pi, catPanSTARRS.dec * 180/numpy.pi, color='blue', s=0.1, marker='x')
ax1b.scatter(catCMASS.ra * 180/numpy.pi, catCMASS.dec * 180/numpy.pi, color='green', s=0.1, marker='x')
ax1b.set_xlabel('RA (degrees)')
ax1b.set_ylabel('Dec (degrees)')
#ax1b.set_title('')

# Repeat in the opposite order
ax2b.scatter(catCMASS.ra * 180/numpy.pi, catCMASS.dec * 180/numpy.pi, color='blue', s=0.1, marker='x')
ax2b.scatter(catPanSTARRS.ra * 180/numpy.pi, catPanSTARRS.dec * 180/numpy.pi, color='green', s=0.1, marker='x')
ax2b.set_xlabel('RA (degrees)')
ax2b.set_ylabel('Dec (degrees)')
#ax2b.set_title('')

plt.show()


# In[51]:


connPAN = sqlite3.connect(dataPath + 'PanSTARRS.db')

qry = "SELECT ID, DEC, RA, zSN, zHost FROM PanSTARRSNEW WHERE (DEC > -20) AND ((zSN > -999) OR (zHost > -999))"

PanSTARRSNEW_GoodZ_ovelap = pd.read_sql(qry, con=connPAN)
PanSTARRSNEW_GoodZ_ovelap.head(3) # 1058 objects


# In[52]:


connBOSS.close()
connBOSSRands.close()


# In[ ]:





# In[53]:


catPanSTARRS_overlap = treecorr.Catalog(ra=PanSTARRSNEW_GoodZ_ovelap['RA'], dec=PanSTARRSNEW_GoodZ_ovelap['DEC'], ra_units='degrees', dec_units='degrees')
catPanSTARRS


# In[54]:


catCMASS = treecorr.Catalog(ra=CMASSLOWZTOT_DF['RA'], dec=CMASSLOWZTOT_DF['DEC'], 
                                ra_units='degrees', dec_units='degrees')
catCMASS


# In[55]:


nnXCorr = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
nnXCorr.process(catPanSTARRS_overlap, catCMASS)
nnXCorr


# In[56]:


RandsPAN = treecorr.Catalog(ra=maskRA_overlap, dec=maskDEC_overlap, ra_units='degrees', dec_units='degrees')
randCMASS = treecorr.Catalog(ra=CMASSLOWZTOT_DF_rands['RA'], dec=CMASSLOWZTOT_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')


# In[57]:


rrXCorr = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rrXCorr.process(RandsPAN, randCMASS)


# In[58]:


drXCorr = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
drXCorr.process(catPanSTARRS_overlap, randCMASS)


# In[59]:


rdXCorr = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rdXCorr.process(catCMASS, RandsPAN)


# In[60]:


rXCorr = numpy.exp(nnXCorr.meanlogr)
xiXCorr, varxiXCorr = nnXCorr.calculateXi(rrXCorr, drXCorr, rdXCorr)
sigXCorr = numpy.sqrt(varxiXCorr)


# In[61]:


# Plot the Cross Correlation function: 

plt.plot(rXCorr, xiXCorr, color='blue')
plt.plot(rXCorr, -xiXCorr, color='blue', ls=':')
plt.errorbar(rXCorr[xiXCorr>0], xiXCorr[xiXCorr>0], yerr=sigXCorr[xiXCorr>0], color='green', lw=0.5, ls='')
plt.errorbar(rXCorr[xiXCorr<0], -xiXCorr[xiXCorr<0], yerr=sigXCorr[xiXCorr<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-rXCorr, xiXCorr, yerr=sigXCorr, color='blue')

plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
#plt.xlim([0.01,10])
#plt.ylim([0.0, .00001])
plt.title("Cross Correlation")
plt.savefig(testingProductsPath + 'Cross-Corr PanSTARRS and eBOSS with error' + date)
plt.show()


# In[62]:


# Plot the Cross Correlation function: 

plt.plot(rXCorr, xiXCorr, color='blue')
plt.plot(rXCorr, -xiXCorr, color='blue', ls=':')
#plt.errorbar(rXCorr[xiXCorr>0], xiXCorr[xiXCorr>0], yerr=sigXCorr[xiXCorr>0], color='green', lw=0.5, ls='')
#plt.errorbar(rXCorr[xiXCorr<0], -xiXCorr[xiXCorr<0], yerr=sigXCorr[xiXCorr<0], color='green', lw=0.5, ls='')
#leg = plt.errorbar(-rXCorr, xiXCorr, yerr=sigXCorr, color='blue')

plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')

plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
#plt.xlim([0.01,10])
#plt.ylim([0.0, .00001])
plt.title("Cross Correlation")
plt.savefig(testingProductsPath + 'Cross-Corr PanSTARRS and eBOSS' + date)
plt.show()


# In[63]:


f2, (ax1b, ax2b) = plt.subplots(1, 2, figsize=(20,5))

ax1b.scatter(catPanSTARRS_overlap.ra * 180/numpy.pi, catPanSTARRS_overlap.dec * 180/numpy.pi, color='blue', s=0.1, marker='x')
ax1b.scatter(catCMASS.ra * 180/numpy.pi, catCMASS.dec * 180/numpy.pi, color='green', s=0.1, marker='x')
ax1b.set_xlabel('RA (degrees)')
ax1b.set_ylabel('Dec (degrees)')
#ax1b.set_title('')

# Repeat in the opposite order
ax2b.scatter(catCMASS.ra * 180/numpy.pi, catCMASS.dec * 180/numpy.pi, color='blue', s=0.1, marker='x')
ax2b.scatter(catPanSTARRS_overlap.ra * 180/numpy.pi, catPanSTARRS_overlap.dec * 180/numpy.pi, color='green', s=0.1, marker='x')
ax2b.set_xlabel('RA (degrees)')
ax2b.set_ylabel('Dec (degrees)')
#ax2b.set_title('')

plt.show()


# In[64]:


f2, (ax1b, ax2b) = plt.subplots(1, 2, figsize=(20,5))

ax1b.scatter(RandsPAN.ra * 180/numpy.pi, RandsPAN.dec * 180/numpy.pi, color='blue', s=0.1, marker='x')
ax1b.scatter(randCMASS.ra * 180/numpy.pi, randCMASS.dec * 180/numpy.pi, color='green', s=0.1, marker='x')
ax1b.set_xlabel('RA (degrees)')
ax1b.set_ylabel('Dec (degrees)')
#ax1b.set_title('')

# Repeat in the opposite order
ax2b.scatter(randCMASS.ra * 180/numpy.pi, randCMASS.dec * 180/numpy.pi, color='blue', s=0.1, marker='x')
ax2b.scatter(RandsPAN.ra * 180/numpy.pi, RandsPAN.dec * 180/numpy.pi, color='green', s=0.1, marker='x')
ax2b.set_xlabel('RA (degrees)')
ax2b.set_ylabel('Dec (degrees)')
#ax2b.set_title('')

plt.show()




"""