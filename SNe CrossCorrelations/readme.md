# SNe CrossCorrelations Overview:


<Strong> <em> Project Summary: </Strong> </em> We are investigating the Mukherjee and Wandelt (2018) Paper: <em> Beyond the classical distance-redshift test: cross-correlating redshift-free standard candles and sirens with redshift surveys </em>. The proposal of the paper is to use cross correlations of Supernova (without red shift information) and galaxy surveys (with redshift information) to constrain cosmological parameters. The article suggests a greatly improved constraint with this new method. Our first goal is to use the method proposed on data from BOSS and PanSTARRS. Ultimatly we want to verify the improved constraint ourselves and explore applications to future surveys such as LSST. 

### Directories: 
- <Strong> Cross Correlations Other Versions </Strong>: Contains other versions of the code. For example there is a basemaps version where some of the PanSTARRS data is plotted using basemaps to trouble shoot some issues with the randoms and data not overlapping. there is a "Monkey version" where it produces a very specific graph to complare to someone else's results. 

- <Strong> Mukherjee_2018_Explanation </Strong>: Latex files and PDF of the explanation of the paper this analysis is based off. 

- <Strong> Old Code </Strong>: Not in active use anymore. Might need to reference later. 

- <Strong> VerificationTestingProducts </Strong>: Folder full of the products from individula code runs. Each sub-directory is the date and time of the run and contains a text doc with details of the run as well as graphs/products produced. 


### Files: 

- <Strong> CMASS and LOWZ rands _ Parsing Notebook.ipynb </Strong>:  

- <Strong> Mukherjee_2018_CrossCorr.pdf </Strong>: Article paper that we are basing this porject on. 

- <Strong> PANSTARRS and BOSS _ Raw notebook.ipynb </Strong>: Messing around with the data, plotting masks, distributions in redshift bins, etc. <em> Work in Progress. </em>

- <Strong> PanSTARRS and CMASS _ Correlations _ V2.ipynb </Strong>: Current version. Correlations for SDSS (BOSS) and PanSTARRS data using TreeCorr. Prelim signal verification for porject. <em> Work in Progress. </em>

- <Strong> PanSTARRS and CMASS _ Correlations.ipynb </Strong>: Most Recent Previous Version

- <Strong> PanSTARRS and CMASS _ Correlations.py </Strong>: Current version. Same as jupyter but written in Python. (cleaned up differently). 

- <Strong> PanSTARRS and CMASS _ Parsing Notebook.ipynb </Strong>: Most Recent Previous Version

- <Strong> TreeCorr Documentation Summary.ipynb </Strong> : A bridge between Jarvis Treecorr Documentation and my actual CrossCorr code. Cheat sheet of need to know functions to understand the PanSTARRS and CMASS _ Correlations.ipynb. <em> Work in Progress. </em>

