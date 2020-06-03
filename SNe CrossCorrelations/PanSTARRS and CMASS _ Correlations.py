#!/usr/bin/env python
# coding: utf-8

# I DONT WORK ON WINDOWS... REWRITE WITH .os.join things if doing that...

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         Imports and formatting:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#plt.switch_backend('agg') #For HEP, matplotlib x windows issues see python version for more usage 
import treecorr
import numpy
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import sqlite3
import os
import datetime



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         Define notebook wide functions and data paths to use:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Define the paths for local and HEP machines:
#DATA_PATH = '/Users/megantabbutt/CosmologyDataProducts/'
DATA_PATH = '/afs/hep.wisc.edu/home/tabbutt/private/CosmologyDataProducts/'

#TESTING_PRODUCTS_PATH = "/Users/megantabbutt/Cosmology/Cosmology/SNe CrossCorrelations/VerificationTestingProducts/"
TESTING_PRODUCTS_PATH = "/afs/hep.wisc.edu/home/tabbutt/public/Cosmology/SNe CrossCorrelations/VerificationTestingProducts"

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
NOTES.write("\n \n")
NOTES.close()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         0. Pull in and parse data:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NOTES = open(NOTES_PATH, "a")
NOTES.write("0. Pull in and parse data:")
NOTES.write("\n \n")
NOTES.close()


# note: There are 10 pointings for the PanSTARRS data, we will use all 10 for the Auto Correlation,
# # but when we correlated to CMASS, we need to only use the 9 overlap with CMASS. --- IMPORTANT

# PanSTARRS:
connPAN = sqlite3.connect(DATA_PATH + 'PanSTARRS.db')
qry = "SELECT ID, DEC, RA, zSN, zHost FROM PanSTARRSNEW WHERE (zSN > -999) || (zHost > -999)"
PanSTARRSNEW_GoodZ = pd.read_sql(qry, con=connPAN)
print("PanSTARRSNEW_GoodZ: ") # 1129 objects over 10 pointings
print(PanSTARRSNEW_GoodZ.head(3)) # 1129 objects over 10 pointings
NOTES = open(NOTES_PATH, "a")
NOTES.write("PanSTARRSNEW_GoodZ Database (with 10 pointings) objects: " + str(len(PanSTARRSNEW_GoodZ)))
NOTES.write("\n \n")
NOTES.close()
connPAN.close()


# CMASS/LOWZ:
connBOSS = sqlite3.connect(DATA_PATH + 'CMASS_and_LOWZ.db')
qry = "SELECT * FROM CMASSLOWZTOT_South UNION SELECT * FROM CMASSLOWZTOT_North"
CMASSLOWZTOT_DF = pd.read_sql(qry, con=connBOSS)
print("CMASSLOWZTOT_DF: ")
print(CMASSLOWZTOT_DF.head(3)) # 1.3 million objects
NOTES = open(NOTES_PATH, "a")
NOTES.write("CMASSLOWZTOT_DF Database objects: " + str(len(CMASSLOWZTOT_DF)))
NOTES.write("\n \n")
NOTES.close()
connBOSS.close()


# Randoms provided by CMASS:
connBOSSRands = sqlite3.connect(DATA_PATH + 'CMASS_and_LOWZ_rands.db')
randSampleQry = "SELECT * FROM CMASSLOWZTOT_South_rands WHERE `index` IN (SELECT `index` FROM CMASSLOWZTOT_South_rands ORDER BY RANDOM() LIMIT 500) UNION SELECT * FROM CMASSLOWZTOT_North_rands WHERE `index` IN (SELECT `index` FROM CMASSLOWZTOT_North_rands ORDER BY RANDOM() LIMIT 500)"
#randQry = "SELECT * FROM CMASSLOWZTOT_South_rands UNION SELECT * FROM CMASSLOWZTOT_North_rands"
CMASSLOWZTOT_DF_rands = pd.read_sql(randSampleQry, con=connBOSSRands)
CMASSLOWZTOT_DF_rands.to_json(DATA_PATH + "CMASSLOWZTOT_DF_rands")
print("CMASSLOWZTOT_DF_rands: ")
print(CMASSLOWZTOT_DF_rands.head(3))
NOTES = open(NOTES_PATH, "a")
NOTES.write("CMASSLOWZTOT_DF_rands Database objects: " + str(len(CMASSLOWZTOT_DF_rands)))
NOTES.write("\n \n")
NOTES.close()
connBOSSRands.close()




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         1. Create the TreeCorr Catalogs of Data:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NOTES = open(NOTES_PATH, "a")
NOTES.write("1. Create the TreeCorr Catalogs of Data:")
NOTES.write("\n \n")
NOTES.close()


cat_PanSTARRS_Full = treecorr.Catalog(ra=PanSTARRSNEW_GoodZ['RA'], dec=PanSTARRSNEW_GoodZ['DEC'], ra_units='degrees', dec_units='degrees')
print("cat_PanSTARRS_Full:")
print(cat_PanSTARRS_Full)
NOTES = open(NOTES_PATH, "a")
NOTES.write("Created cat_PanSTARRS_Full.")
NOTES.write("\n \n")
NOTES.close()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         2. Create the randoms for PanSTARRS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NOTES = open(NOTES_PATH, "a")
NOTES.write("2. Create the randoms for PanSTARRS. Include all 10 pointings, delete MD02 later.")
NOTES.write("\n \n")
NOTES.close()


# Change this for more and less, 10E5 good for personal laptop ~5min run time
randsLength = 10**4
NOTES = open(NOTES_PATH, "a")
NOTES.write("randsLength: " + str(randsLength))
NOTES.write("\n \n")
NOTES.close()


# Create the random points in RA and Sin(DEC)
ra_min_PanSTARRS = numpy.min(cat_PanSTARRS_Full.ra)
ra_max_PanSTARRS = numpy.max(cat_PanSTARRS_Full.ra)
dec_min_PanSTARRS = numpy.min(cat_PanSTARRS_Full.dec)
dec_max_PanSTARRS = numpy.max(cat_PanSTARRS_Full.dec)
print('PanSTARRS ra range = %f .. %f' % (ra_min_PanSTARRS, ra_max_PanSTARRS))
print('PanSTARRS dec range = %f .. %f' % (dec_min_PanSTARRS, dec_max_PanSTARRS))

rand_ra_PanSTARRS = numpy.random.uniform(ra_min_PanSTARRS, ra_max_PanSTARRS, randsLength)
rand_sindec_PanSTARRS = numpy.random.uniform(numpy.sin(dec_min_PanSTARRS), numpy.sin(dec_max_PanSTARRS), randsLength)
rand_dec_PanSTARRS = numpy.arcsin(rand_sindec_PanSTARRS)


# Got from a paper:  https://arxiv.org/pdf/1612.05560.pdf
pointings = {"MD01": [035.875, -04.250], "MD03": [130.592, 44.317], "MD04": [150.000, 02.200], 
             "MD05": [161.917, 58.083], "MD06": [185.000, 47.117], "MD07": [213.704, 53.083], 
             "MD08": [242.787, 54.950], "MD09": [334.188, 00.283], "MD10": [352.312, -00.433], "MD02": [053.100, -27.800],}

# Check how well the randoms cover the same space as the data
f1, (ax1a, ax2a, ax3a) = plt.subplots(1, 3, figsize=(20, 5))
ax1a.scatter(cat_PanSTARRS_Full.ra * 180/numpy.pi, cat_PanSTARRS_Full.dec * 180/numpy.pi, color='red', s=0.1, marker='x')
ax1a.scatter(rand_ra_PanSTARRS * 180/numpy.pi, rand_dec_PanSTARRS * 180/numpy.pi, color='blue', s=0.1)
ax1a.set_xlabel('RA (degrees)')
ax1a.set_ylabel('Dec (degrees)')
ax1a.set_title('Randoms on top of data')
# Repeat in the opposite order
ax2a.scatter(rand_ra_PanSTARRS * 180/numpy.pi, rand_dec_PanSTARRS * 180/numpy.pi, color='blue', s=0.1, marker='x')
ax2a.scatter(cat_PanSTARRS_Full.ra * 180/numpy.pi, cat_PanSTARRS_Full.dec * 180/numpy.pi, color='red', s=0.1)
ax2a.set_xlabel('RA (degrees)')
ax2a.set_ylabel('Dec (degrees)')
ax2a.set_title('Data on top of randoms')
# Zoom to look at coverage of randoms and reals
ax3a.scatter(rand_ra_PanSTARRS * 180/numpy.pi, rand_dec_PanSTARRS * 180/numpy.pi, color='blue', s=1, marker='x', label='rands')
ax3a.scatter(cat_PanSTARRS_Full.ra * 180/numpy.pi, cat_PanSTARRS_Full.dec * 180/numpy.pi, color='red', s=1, label='data')
ax3a.set_xlabel('RA (degrees)')
ax3a.set_ylabel('Dec (degrees)')
ax3a.set_title('Data on top of randoms_Zoom')
ax3a.legend(loc = "upper right")
ax3a.set_xlim(128, 133)
ax3a.set_ylim(42, 47)
plt.savefig(TESTING_PRODUCTS_PATH + "/PanSTARRS data and randoms")
plt.show()
NOTES = open(NOTES_PATH, "a")
NOTES.write("Plotted: PanSTARRS data and randoms")
NOTES.write("\n \n")
NOTES.close()


# Make a mask that consists of the ten pointings populated with the randoms that are in it...
# Def a better way to do this?
maskRA = []
maskDEC = []
randoms_Lengths = []

print("Populating pointings with randoms: ")

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
    randoms_Lengths.append(len(maskRA) - maskRAprevious)

NOTES = open(NOTES_PATH, "a")
NOTES.write("Populated pointings with randoms. Randoms per pointing: (1, 3-10, 2):")
NOTES.write(str(randoms_Lengths))
NOTES.write("\n \n")
NOTES.close()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         3. Make PanSTARRS Count-Count Auto Correlation Functions:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NOTES = open(NOTES_PATH, "a")
NOTES.write("3. Make PanSTARRS Count-Count Auto Correlation Functions:")
NOTES.write("\n \n")
NOTES.close()


# Data Auto-correlation: (dd)
nn_PanSTARRS_Auto_self = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
nn_PanSTARRS_Auto_self.process(cat_PanSTARRS_Full)

cat_rand_PanSTARRS_Full = treecorr.Catalog(ra=maskRA, dec=maskDEC, ra_units='degrees', dec_units='degrees')
rr_PanSTARRS_Full = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rr_PanSTARRS_Full.process(cat_rand_PanSTARRS_Full)

dr_PanSTARRS_Full = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
dr_PanSTARRS_Full.process(cat_PanSTARRS_Full, cat_rand_PanSTARRS_Full)

r_PanSTARRS_Full = numpy.exp(nn_PanSTARRS_Auto_self.meanlogr)
xi_PanSTARRS_Full, varxi_PanSTARRS_Full = nn_PanSTARRS_Auto_self.calculateXi(rr_PanSTARRS_Full, dr_PanSTARRS_Full)
sig_PanSTARRS_Full = numpy.sqrt(varxi_PanSTARRS_Full)


# Plot the Correlation function:
plt.plot(r_PanSTARRS_Full, xi_PanSTARRS_Full, color='blue')
plt.plot(r_PanSTARRS_Full, -xi_PanSTARRS_Full, color='blue', ls=':')
plt.errorbar(r_PanSTARRS_Full[xi_PanSTARRS_Full>0], xi_PanSTARRS_Full[xi_PanSTARRS_Full>0], yerr=sig_PanSTARRS_Full[xi_PanSTARRS_Full>0], color='green', lw=0.5, ls='')
plt.errorbar(r_PanSTARRS_Full[xi_PanSTARRS_Full<0], -xi_PanSTARRS_Full[xi_PanSTARRS_Full<0], yerr=sig_PanSTARRS_Full[xi_PanSTARRS_Full<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-r_PanSTARRS_Full, xi_PanSTARRS_Full, yerr=sig_PanSTARRS_Full, color='blue')
plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')
plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
plt.xlim([0.01,10])
plt.savefig(TESTING_PRODUCTS_PATH + "/PanSTARRS Auto-Corr with PanSTARRS randoms")
plt.show()
NOTES = open(NOTES_PATH, "a")
NOTES.write("Plotted: PanSTARRS Auto-Corr with PanSTARRS randoms")
NOTES.write("\n \n")
NOTES.close()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         4. Make CMASS Count-Count Auto Correlation Functions:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NOTES = open(NOTES_PATH, "a")
NOTES.write("4. Make CMASS Count-Count Auto Correlation Functions:")
NOTES.write("\n \n")
NOTES.close()


cat_eBOSS = treecorr.Catalog(ra=CMASSLOWZTOT_DF['RA'], dec=CMASSLOWZTOT_DF['DEC'], ra_units='degrees', dec_units='degrees')
print("cat_eBOSS:")
print(cat_eBOSS)
NOTES = open(NOTES_PATH, "a")
NOTES.write("Created cat_eBOSS.")
NOTES.write("\n \n")
NOTES.close()


nn_eBOSS_Auto_self = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
nn_eBOSS_Auto_self.process(cat_eBOSS)

cat_rand_eBOSS = treecorr.Catalog(ra=CMASSLOWZTOT_DF_rands['RA'], dec=CMASSLOWZTOT_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')
rr_eBOSS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rr_eBOSS.process(cat_rand_eBOSS)

dr_eBOSS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
dr_eBOSS.process(cat_eBOSS, cat_rand_eBOSS)

r_eBOSS = numpy.exp(nn_eBOSS_Auto_self.meanlogr)
xi_eBOSS, varxi_eBOSS = nn_eBOSS_Auto_self.calculateXi(rr_eBOSS, dr_eBOSS)
sig_eBOSS = numpy.sqrt(varxi_eBOSS)


# Check that the randoms cover the same space as the data
f3, (ax1c, ax2c) = plt.subplots(1, 2, figsize=(20, 5))
ax1c.scatter(cat_eBOSS.ra * 180/numpy.pi, cat_eBOSS.dec * 180/numpy.pi, color='red', s=0.1)
ax1c.set_xlabel('RA (degrees)')
ax1c.set_ylabel('Dec (degrees)')
ax1c.set_title('CMASS/LOWZ Data')
# Repeat in the opposite order
ax2c.scatter(CMASSLOWZTOT_DF_rands['RA'], CMASSLOWZTOT_DF_rands['DEC'], color='blue', s=0.1)
ax2c.set_xlabel('RA (degrees)')
ax2c.set_ylabel('Dec (degrees)')
ax2c.set_title('CMASS/LOWZ Randoms')
plt.savefig(TESTING_PRODUCTS_PATH + "/CMASS_LOWZ Randoms")
plt.show()
NOTES = open(NOTES_PATH, "a")
NOTES.write("Plotted: CMASS_LOWZ Randoms")
NOTES.write("\n \n")
NOTES.close()


# Plot the autocorrelation function:
plt.plot(r_eBOSS, xi_eBOSS, color='blue')
plt.plot(r_eBOSS, -xi_eBOSS, color='blue', ls=':')
plt.errorbar(r_eBOSS[xi_eBOSS>0], xi_eBOSS[xi_eBOSS>0], yerr=sig_eBOSS[xi_eBOSS>0], color='green', lw=0.5, ls='')
plt.errorbar(r_eBOSS[xi_eBOSS<0], -xi_eBOSS[xi_eBOSS<0], yerr=sig_eBOSS[xi_eBOSS<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-r_eBOSS, xi_eBOSS, yerr=sig_eBOSS, color='blue')
plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')
plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
plt.xlim([0.01,10])
plt.title("eBOSS Auto Corr with eBOSS randoms")
plt.savefig(TESTING_PRODUCTS_PATH + "/eBOSS Auto Corr with eBOSS randoms")
plt.show()
NOTES = open(NOTES_PATH, "a")
NOTES.write("Plotted: eBOSS Auto Corr with eBOSS randoms")
NOTES.write("\n \n")
NOTES.close()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         5. Analyze if the plots are correct:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NOTES = open(NOTES_PATH, "a")
NOTES.write("5. Analyze if the plots are correct:")
NOTES.write("\n \n")
NOTES.close()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         5.1 Auto Correlate the CMASS and LOWZ randoms:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NOTES = open(NOTES_PATH, "a")
NOTES.write("5.1 Auto Correlate the CMASS and LOWZ randoms:")
NOTES.write("\n \n")
NOTES.close()


connCMASSRands = sqlite3.connect(DATA_PATH + 'CMASS_rands.db')
randSampleQry = "SELECT * FROM CMASS_South_rands WHERE `index` IN (SELECT `index` FROM CMASS_South_rands ORDER BY RANDOM() LIMIT 5000) UNION SELECT * FROM CMASS_North_rands WHERE `index` IN (SELECT `index` FROM CMASS_North_rands ORDER BY RANDOM() LIMIT 5000)"
CMASS_DF_rands = pd.read_sql(randSampleQry, con=connCMASSRands)
CMASS_DF_rands.to_json(DATA_PATH + "CMASS_DF_rands")
print("CMASS_DF_rands: ")
print(CMASS_DF_rands.head(3)) # 1.3 million objects
NOTES = open(NOTES_PATH, "a")
NOTES.write("CMASS_DF_rands Database objects: " + str(len(CMASS_DF_rands)))
NOTES.write("\n \n")
NOTES.close()
connCMASSRands.close()

connLOWZRands = sqlite3.connect(DATA_PATH + 'LOWZ_rands.db')
randSampleQry = "SELECT * FROM LOWZ_South_rands WHERE `index` IN (SELECT `index` FROM LOWZ_South_rands ORDER BY RANDOM() LIMIT 5000) UNION SELECT * FROM LOWZ_North_rands WHERE `index` IN (SELECT `index` FROM LOWZ_North_rands ORDER BY RANDOM() LIMIT 5000)"
LOWZ_DF_rands = pd.read_sql(randSampleQry, con=connLOWZRands)
LOWZ_DF_rands.to_json(DATA_PATH + "LOWZ_DF_rands")
print("LOWZ_DF_rands: ")
print(LOWZ_DF_rands.head(3)) # 1.3 million objects
NOTES = open(NOTES_PATH, "a")
NOTES.write("LOWZ_DF_rands Database objects: " + str(len(LOWZ_DF_rands)))
NOTES.write("\n \n")
NOTES.close()
connLOWZRands.close()


cat_CMASS_rands = treecorr.Catalog(ra=CMASS_DF_rands['RA'], dec=CMASS_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')
cat_CMASS_rands

nn_CMASS_Auto_LOWZRands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
nn_CMASS_Auto_LOWZRands.process(cat_CMASS_rands)

cat_rand_LOWZ = treecorr.Catalog(ra=LOWZ_DF_rands['RA'], dec=LOWZ_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')
rr_LOWZ_rands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rr_LOWZ_rands.process(cat_rand_LOWZ)

dr_CMASS_LOWZrands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
dr_CMASS_LOWZrands.process(cat_CMASS_rands, cat_rand_LOWZ)

r_CMASS_LOWZrands = numpy.exp(nn_CMASS_Auto_LOWZRands.meanlogr)
xi_CMASS_LOWZrands, varxi_CMASS_LOWZrands = nn_CMASS_Auto_LOWZRands.calculateXi(rr_LOWZ_rands, dr_CMASS_LOWZrands)
sig_CMASS_LOWZrands = numpy.sqrt(varxi_CMASS_LOWZrands)


# Check that the randoms cover the same space as the data
f4, (ax1d, ax2d) = plt.subplots(1, 2, figsize=(20, 5))
ax1d.scatter(cat_CMASS_rands.ra * 180/numpy.pi, cat_CMASS_rands.dec * 180/numpy.pi, color='blue', s=0.1)
ax1d.set_xlabel('RA (degrees)')
ax1d.set_ylabel('Dec (degrees)')
ax1d.set_title('CMASS Randoms')
# Repeat in the opposite order
ax2d.scatter(cat_CMASS_rands.ra * 180/numpy.pi, cat_CMASS_rands.dec * 180/numpy.pi, color='green', s=0.1)
ax2d.set_xlabel('RA (degrees)')
ax2d.set_ylabel('Dec (degrees)')
ax2d.set_title('LOWZ Randoms')
plt.title("CMASS and LOWZ randoms")
plt.savefig(TESTING_PRODUCTS_PATH + "/CMASS and LOWZ randoms")
plt.show()
NOTES = open(NOTES_PATH, "a")
NOTES.write("Plotted: CMASS and LOWZ randoms")
NOTES.write("\n \n")
NOTES.close()


# Plot the autocorrelation function:
plt.plot(r_CMASS_LOWZrands, xi_CMASS_LOWZrands, color='blue')
plt.plot(r_CMASS_LOWZrands, -xi_CMASS_LOWZrands, color='blue', ls=':')
plt.errorbar(r_CMASS_LOWZrands[xi_CMASS_LOWZrands>0], xi_CMASS_LOWZrands[xi_CMASS_LOWZrands>0], yerr=sig_CMASS_LOWZrands[xi_CMASS_LOWZrands>0], color='green', lw=0.5, ls='')
plt.errorbar(r_CMASS_LOWZrands[xi_CMASS_LOWZrands<0], -xi_CMASS_LOWZrands[xi_CMASS_LOWZrands<0], yerr=sig_CMASS_LOWZrands[xi_CMASS_LOWZrands<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-r_CMASS_LOWZrands, xi_CMASS_LOWZrands, yerr=sig_CMASS_LOWZrands, color='blue')
plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')
plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
#plt.xlim([0.01,10])
#plt.ylim([0.0, .00001])
plt.title("CMASS_rands Auto Corr with LOWZ_rands as randoms")
plt.savefig(TESTING_PRODUCTS_PATH + "/CMASS_rands Auto Corr with LOWZ_rands as randoms")
plt.show()
NOTES = open(NOTES_PATH, "a")
NOTES.write("Plotted: CMASS_rands Auto Corr with LOWZ_rands as randoms")
NOTES.write("\n \n")
NOTES.close()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         5.2 Auto Correlate the CMASS and LOWZ randoms - Reversed:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NOTES = open(NOTES_PATH, "a")
NOTES.write("5.2 Auto Correlate the CMASS and LOWZ randoms - Reversed:")
NOTES.write("\n \n")
NOTES.close()

cat_LOWZ_rands = treecorr.Catalog(ra=LOWZ_DF_rands['RA'], dec=LOWZ_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')
cat_LOWZ_rands

nn_LOWZ_Auto_CMASSRands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
nn_LOWZ_Auto_CMASSRands.process(cat_LOWZ_rands)

cat_rand_CMASS = treecorr.Catalog(ra=CMASS_DF_rands['RA'], dec=CMASS_DF_rands['DEC'], ra_units='degrees', dec_units='degrees')
rr_CMASS_rands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rr_CMASS_rands.process(cat_rand_CMASS)

dr_LOWZ_CMASSrands = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
dr_LOWZ_CMASSrands.process(cat_rand_LOWZ, cat_CMASS_rands)

r_LOWZ_CMASSrands = numpy.exp(nn_LOWZ_Auto_CMASSRands.meanlogr)
xi_LOWZ_CMASSrands, varxi_LOWZ_CMASSrands = nn_LOWZ_Auto_CMASSRands.calculateXi(rr_CMASS_rands, dr_LOWZ_CMASSrands)
sig_LOWZ_CMASSrands = numpy.sqrt(varxi_LOWZ_CMASSrands)


# Plot the autocorrelation function:
plt.plot(r_LOWZ_CMASSrands, xi_LOWZ_CMASSrands, color='blue')
plt.plot(r_LOWZ_CMASSrands, -xi_LOWZ_CMASSrands, color='blue', ls=':')
plt.errorbar(r_LOWZ_CMASSrands[xi_LOWZ_CMASSrands>0], xi_LOWZ_CMASSrands[xi_LOWZ_CMASSrands>0], yerr=sig_LOWZ_CMASSrands[xi_LOWZ_CMASSrands>0], color='green', lw=0.5, ls='')
plt.errorbar(r_LOWZ_CMASSrands[xi_LOWZ_CMASSrands<0], -xi_LOWZ_CMASSrands[xi_LOWZ_CMASSrands<0], yerr=sig_LOWZ_CMASSrands[xi_LOWZ_CMASSrands<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-r_LOWZ_CMASSrands, xi_LOWZ_CMASSrands, yerr=sig_LOWZ_CMASSrands, color='blue')
plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')
plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
#plt.xlim([0.01,10])
#plt.ylim([0.0, .00001])
plt.title("LOWZ_rands Auto Corr with CMASS_rands as randoms")
plt.savefig(TESTING_PRODUCTS_PATH + "/LOWZ_rands Auto Corr with CMASS_rands as randoms")
plt.show()
NOTES = open(NOTES_PATH, "a")
NOTES.write("Plotted: LOWZ_rands Auto Corr with CMASS_rands as randoms")
NOTES.write("\n \n")
NOTES.close()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#         6. Cross Correlate the eBOSS and PanSTARRS sets
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NOTES = open(NOTES_PATH, "a")
NOTES.write("6. Cross Correlate the eBOSS and PanSTARRS sets")
NOTES.write("\n \n")
NOTES.close()

print("Populating pointings with randoms: ")
# Need to get just 9 pointings from PanSTARRS:
maskRA_overlap = []
maskDEC_overlap = []
randoms_Lengths_overlap = []

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
        randoms_Lengths_overlap.append(len(maskRA_overlap) - maskRAprevious)

NOTES = open(NOTES_PATH, "a")
NOTES.write("Populated pointings with randoms. Randoms per pointing: (1, 3-10, 2):")
NOTES.write(str(randoms_Lengths_overlap))
NOTES.write("\n \n")
NOTES.close()


connPANoverlap = sqlite3.connect(DATA_PATH + 'PanSTARRS.db')
qry = "SELECT ID, DEC, RA, zSN, zHost FROM PanSTARRSNEW WHERE (DEC > -20) AND ((zSN > -999) OR (zHost > -999))"
PanSTARRSNEW_GoodZ_ovelap = pd.read_sql(qry, con=connPANoverlap)
print("PanSTARRSNEW_GoodZ_ovelap: ") # 1129 objects over 10 pointings
print(PanSTARRSNEW_GoodZ_ovelap.head(3)) # 1129 objects over 10 pointings
NOTES = open(NOTES_PATH, "a")
NOTES.write("PanSTARRSNEW_GoodZ_ovelap Database (with 9 pointings) objects: " + str(len(PanSTARRSNEW_GoodZ_ovelap)))
NOTES.write("\n \n")
NOTES.close()
connPANoverlap.close()



cat_PanSTARRS_overlap = treecorr.Catalog(ra=PanSTARRSNEW_GoodZ_ovelap['RA'], dec=PanSTARRSNEW_GoodZ_ovelap['DEC'], ra_units='degrees', dec_units='degrees')

nn_Pan_xCorr_eBOSS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
nn_Pan_xCorr_eBOSS.process(cat_PanSTARRS_overlap, cat_eBOSS)

cat_rand_Pan_Overlap = treecorr.Catalog(ra=maskRA_overlap, dec=maskDEC_overlap, ra_units='degrees', dec_units='degrees')

rr_Pan_xCorr_eBOSS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rr_Pan_xCorr_eBOSS.process(cat_rand_Pan_Overlap, cat_rand_eBOSS)

dr_Pan_xCorr_eBOSS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
dr_Pan_xCorr_eBOSS.process(cat_PanSTARRS_overlap, cat_rand_eBOSS)

rd_Pan_xCorr_eBOSS = treecorr.NNCorrelation(min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees')
rd_Pan_xCorr_eBOSS.process(cat_rand_eBOSS, cat_PanSTARRS_overlap)

r_Pan_xCorr_eBOSS = numpy.exp(nn_Pan_xCorr_eBOSS.meanlogr)
xi_Pan_xCorr_eBOSS, varxi_Pan_xCorr_eBOSS = nn_Pan_xCorr_eBOSS.calculateXi(rr_Pan_xCorr_eBOSS, dr_Pan_xCorr_eBOSS, rd_Pan_xCorr_eBOSS)
sig_Pan_xCorr_eBOSS = numpy.sqrt(varxi_Pan_xCorr_eBOSS)


# Plot the Cross Correlation function:
plt.plot(r_Pan_xCorr_eBOSS, xi_Pan_xCorr_eBOSS, color='blue')
plt.plot(r_Pan_xCorr_eBOSS, -xi_Pan_xCorr_eBOSS, color='blue', ls=':')
plt.errorbar(r_Pan_xCorr_eBOSS[xi_Pan_xCorr_eBOSS>0], xi_Pan_xCorr_eBOSS[xi_Pan_xCorr_eBOSS>0], yerr=sig_Pan_xCorr_eBOSS[xi_Pan_xCorr_eBOSS>0], color='green', lw=0.5, ls='')
plt.errorbar(r_Pan_xCorr_eBOSS[xi_Pan_xCorr_eBOSS<0], -xi_Pan_xCorr_eBOSS[xi_Pan_xCorr_eBOSS<0], yerr=sig_Pan_xCorr_eBOSS[xi_Pan_xCorr_eBOSS<0], color='green', lw=0.5, ls='')
leg = plt.errorbar(-r_Pan_xCorr_eBOSS, xi_Pan_xCorr_eBOSS, yerr=sig_Pan_xCorr_eBOSS, color='blue')
plt.xscale('log')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$\theta$ (degrees)')
plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
#plt.xlim([0.01,10])
#plt.ylim([0.0, .00001])
plt.title("PanSTARRS cross Correlation with eBOSS")
plt.savefig(TESTING_PRODUCTS_PATH + "/PanSTARRS cross Correlation with eBOSS")
plt.show()
NOTES = open(NOTES_PATH, "a")
NOTES.write("Plotted: PanSTARRS cross Correlation with eBOSS")
NOTES.write("\n \n")
NOTES.close()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NOTES = open(NOTES_PATH, "a")
NOTES.write("Program is done.")
NOTES.write("\n \n")
NOTES.close()
