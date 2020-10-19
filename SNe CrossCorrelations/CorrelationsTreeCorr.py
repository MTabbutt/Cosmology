import sqlite3
import pandas as pd
import treecorr
import numpy
import matplotlib.pyplot as plt

class Correlation_Data:
    
    def __init__(self, DATA_PATH, TESTING_PRODUCTS_PATH):
        self.DATA_PATH = DATA_PATH
        self.TESTING_PRODUCTS_PATH = TESTING_PRODUCTS_PATH
                

    def makeDataCatalogs(self, databaseFile, query, printHead=True):
        """ Query the database, print head, and return pandas DF object.
        
        databaseFile - String: database path
        query - String: query for the database
        printHead=True - prints the first three rows of the returned database
        """   
        connection = sqlite3.connect(self.DATA_PATH + databaseFile)
        try:
            dataFrame = pd.read_sql(query, con=connection)
            if printHead == True:
                print("dataFrame: \n" + str(dataFrame.head(3)))            
            return dataFrame
        except Error: 
            print("Error with pd.read_sql on database: " + databaseFile)
        else:
            connection.close()
  
            
    def autoCorrelation(self, dataRADEC, randRADEC, ra_units='degrees', dec_units='degrees', min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees'):
        """ Use TreeCorr to make the autoCorrelation of two dataFrames passed in. 
        
        dataFrame1 - obj: pandas dataFrame object made with makeDataCatalogs()
        dataFrame1 - obj: pandas dataFrame object made with makeDataCatalogs()
        
        Return - r, xi, varxi, sig
        """
        self.dataRA = dataRADEC[0]
        self.dataDEC = dataRADEC[1]
        self.randRA = randRADEC[0]
        self.randDEC = randRADEC[1]
        
        dataCatalog = treecorr.Catalog(ra=self.dataRA, dec=self.dataDEC, ra_units=ra_units, dec_units=dec_units)
        randCatalog = treecorr.Catalog(ra=self.randRA, dec=self.randDEC, ra_units=ra_units, dec_units=dec_units)
        
        nn = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units=sep_units)
        nn.process(dataCatalog)
        rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units=sep_units)
        rr.process(randCatalog)
        dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units=sep_units)
        dr.process(dataCatalog, randCatalog)

        r = numpy.exp(nn.meanlogr)
        xi, varxi = nn.calculateXi(rr, dr)
        sig = numpy.sqrt(varxi)
    
        return r, xi, varxi, sig
    
    
    def crossCorrelation(self, dataRADEC_1, randRADEC_1, dataRADEC_2, randRADEC_2, ra_units='degrees', dec_units='degrees', min_sep=0.01, max_sep=10, bin_size=0.2, sep_units='degrees'):
        """ Use TreeCorr to make the autoCorrelation of two dataFrames passed in. 
        
        dataFrame1 - obj: pandas dataFrame object made with makeDataCatalogs()
        dataFrame1 - obj: pandas dataFrame object made with makeDataCatalogs()
        
        Return - r, xi, varxi, sig
        """
        self.dataRA_1 = dataRADEC_1[0]
        self.dataDEC_1 = dataRADEC_1[1]
        self.randRA_1 = randRADEC_1[0]
        self.randDEC_1 = randRADEC_1[1]
        self.dataRA_2 = dataRADEC_2[0]
        self.dataDEC_2 = dataRADEC_2[1]
        self.randRA_2 = randRADEC_2[0]
        self.randDEC_2 = randRADEC_2[1]
        
        dataCatalog_1 = treecorr.Catalog(ra=self.dataRA_1, dec=self.dataDEC_1, ra_units=ra_units, dec_units=dec_units)
        randCatalog_1 = treecorr.Catalog(ra=self.randRA_1, dec=self.randDEC_1, ra_units=ra_units, dec_units=dec_units)
        dataCatalog_2 = treecorr.Catalog(ra=self.dataRA_2, dec=self.dataDEC_2, ra_units=ra_units, dec_units=dec_units)
        randCatalog_2 = treecorr.Catalog(ra=self.randRA_2, dec=self.randDEC_2, ra_units=ra_units, dec_units=dec_units)
        
        nn = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units=sep_units)
        nn.process(dataCatalog_1, dataCatalog_2)
        
        rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units=sep_units)
        rr.process(randCatalog_1, randCatalog_2)
        
        dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units=sep_units)
        dr.process(dataCatalog_1, randCatalog_2)
        
        rd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units=sep_units)
        rd.process(randCatalog_1, dataCatalog_2)

        r = numpy.exp(nn.meanlogr)
        xi, varxi = nn.calculateXi(rr, dr, rd)
        sig = numpy.sqrt(varxi)
    
        return r, xi, varxi, sig
    
    
    def plotCorrelationFunction(self, r, xi, varxi, sig, plotTitle, saveName=None, save=False, loc='lower left'):
        self.saveName = saveName
        self.r = r
        self.xi = xi
        self.varxi = varxi
        self.sig = sig
        
        plt.figure(figsize=(12, 8))
        plt.plot(self.r, self.xi, color='blue')
        plt.plot(self.r, -self.xi, color='blue', ls=':')
        plt.errorbar(self.r[self.xi>0], self.xi[self.xi>0], yerr=self.sig[self.xi>0], color='green', lw=0.5, ls='')
        plt.errorbar(self.r[self.xi<0], -self.xi[self.xi<0], yerr=self.sig[self.xi<0], color='green', lw=0.5, ls='')
        leg = plt.errorbar(-self.r, self.xi, yerr=self.sig, color='blue')
        plt.xscale('log')
        plt.yscale('log', nonpositive='clip')
        plt.xlabel(r'$\theta$ (degrees)', fontsize = 12)
        plt.ylabel(r'$w(\theta)$', fontsize = 12)
        #plt.legend([leg], [r'$w(\theta)$'], loc=loc)
        plt.xlim([0.01,10])
        plt.title(plotTitle, fontsize = 16)

        if save == True:
            plt.savefig(self.TESTING_PRODUCTS_PATH + self.saveName)
            print("Saved figure to: \t" + self.TESTING_PRODUCTS_PATH + self.saveName)

        plt.show()
    
    
    
    
    def __str__(self):
        return 
        