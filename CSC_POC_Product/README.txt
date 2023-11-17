This is the first iteration of my Proof of Concept (POC).

The POC is housed in the script "Centralised_SC_POC".

The aim of this POC is to facilitate fictious blue carbon credit distribution through a centralised smart contract (CSS). 

The focus of this POC is mangrove land cover. Mangrove percentage change cover over a monitoring period will serve as the threshold for credit distribution.

The only required input for this POC is a GeoDataFrame. This repository provides one and is called "Client_GeoDataFrame.geojson"

The user must also have access to Google Earth Engine (GEE). You will be required to sign into your GEE account in order for the script to function correctly.


To start this script, simply run and answer the prompts when asked. 


Using a CSS, the user is asked to input the following:
	1 - GeoDataFrame 
 	2 - Appropriate columns for the GeoDataFrame
	3 - The token holder who will have their geometry column monitored
	4 - Valid start and end monitoring period. This script creates annual mosaics so only years are required 
	5 - Valid mangrove land cover change percentag
	6 - A valid amount of tokens to distribute to the monitored user if the CSC is contract compliant

Then the script will calculate mangrove land cover change for the monitoring area and provide validation metrics for the CSC participants to inspect

The user will then be prompted to say if they want to download the outputs and create additional plots with map elements

The classified maps used to calculate mangrove cover percentage change will then be saved and opened for inspection

Finally, the user will be asked to answer if all participants including the Trusted Third Party (TTP) agree that the outputs are valid and contract compliant. 

Tokens will only be distributed if the output is contract compliant. 


Please feel free to explore the script and update the parameters and code to fit your needs when neccessary. 


