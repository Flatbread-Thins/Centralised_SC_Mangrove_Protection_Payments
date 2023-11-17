
#GEE Cloud Project name = GEE-Mangrove-SC

#Libraries related to GEE and processing GEE outputs
import ee
import google
import geemap


#Libraries for setting directory paths
import os

#Math related libararies
import math
import numpy as np

#Libraries to process geodataframes, excel files and change geodataframe geometries
import geopandas as gpd
import pandas as pd

from shapely.geometry import mapping

#Libraries to visualise the web outputs
import webbrowser


#Libraries to open and edit raster data types 
import rasterio

#Libraries to add map elements to offline map outputs
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib_scalebar.scalebar import ScaleBar


'''
Initialise the "GEE_Authenticate" function:
    - If you are using the Python API, ee. Authenticate() will create and save credentials that will automatically be used by 
    ee.Initialize()
    - Thus, you only need to run ee.Authenticate once on your system. 
'''

def GEE_Authenticate ():
    
    #Set API call variable to None for conditional block
    COLAB_AUTH_FLOW_CLOUD_PROJECT_FOR_API_CALLS = None
    
    #Checks if GEE authentication has already occured, if it has then itialise GEE with the GEE API (ee library) 
    if COLAB_AUTH_FLOW_CLOUD_PROJECT_FOR_API_CALLS is None:
      print("\nAuthenticating...")
      if os.path.exists(ee.oauth.get_credentials_path()) is False:
        ee.Authenticate()
      else:
        print('\N{check mark} '
              'Previously created authentication credentials were found.')
      ee.Initialize()
      
    #If it cannot find authentication credentials, it will bring the user to GEE where they must sign in and give the relevatnt permissions
    else:
      print('Authenticating using Colab auth...')
      # Authenticate to populate Application Default Credentials in the Colab VM.
      google.colab.auth.authenticate_user()
      # Create credentials needed for accessing Earth Engine.
      credentials, auth_project_id = google.auth.default()
      # Initialize Earth Engine.
      ee.Initialize(credentials, project=COLAB_AUTH_FLOW_CLOUD_PROJECT_FOR_API_CALLS)
    print('\N{check mark} Successfully initialized!')


'''
Initialise the "create_smart_contract" function:
    - Imports the Centralised GeoDataFrame that the user requests. It must be in their current directory for it to be selected. 
    
    - Returns the following from the token user being monitored in the GeoDataFrame chosen:
        1 - The row index of the monitored user
        2 - The column index of the token count for the monitored user 
        3 - The geometry of the monitoring area - this is extracted from the geospatial data frame and converted to the required .geojson format

    - Returns the following information which forms the Smart Contract requirements:
        1 - The amount of tokens that should be given to the monitored user if they meet the requirements
        2 - The percentage change threshold that the monitored area needs to meet. 
    
    - Returns the start and end date of the monitoring period in the correct GEE format (yyyy-mm-dd) for change analysis 
'''
def create_smart_contract():
    
    
    '''
    Specify the Study Area Geometry 
    '''
    
    #Obtain Current Working Directory of the python file
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # List the files in the current directory
    file_list = os.listdir(current_directory)

    # Print the files in the users current directory as a list of numbered options
    print("Files in your current working directory:\n")
    for index, file in enumerate(file_list):
        print(f"{index + 1}: {file}")
    
    
    '''
    Specify the GeoDataFrame file which will be utilised 
    '''
    
    while True:
        try:
            #input() function will turn any user input from the console into a string variable. The int() function then converts this string into an int data type
            user_choice = int(input("\nPlease note that if your GeoDataFrame is not in the '.geojson' format, this script will not work correctly.\n\nPlease ENTER the number of the GeoDataFrame that you wish to ENTER: "))
            
            #If block to ensure that the number chosen is one of the choices listed from the input() function
            if 1 <= user_choice <= len(file_list):
                chosen_file_index = user_choice - 1  # Adjusted for 0-based indexing
                chosen_file = file_list[chosen_file_index]
    
                # Read the chosen file using GeoPandas
                file_path = os.path.join(current_directory, chosen_file)
                GDF = gpd.read_file(file_path)
    
                print(f"\nYou picked '{chosen_file}'")
                
                

                break  # Exit the loop if a valid choice is made
        
        #Else and except conditional blocks continue the while loop until the user has made a valid choice.
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    '''
    Specify the Token Holder and Token Value columns from the previously chosen GeoDataFrame which will be used for parametising the Smart Contract. 
    '''
    
    available_columns = GDF.columns
    
    print("\nThese are the available columns in your GeoDataFrame:\n")
    for index, header in enumerate(available_columns):
        print(f"{index + 1}: {header}")
    
    
    while True:
        try:
            user_choice_TH = int(input("\nPlease ENTER the number of the 'TOKEN HOLDER' column that you wish to ENTER: "))
            
            user_choice_TA = int(input("\nPlease ENTER the number of the 'TOKEN AMOUNT' column that you wish to ENTER: "))

    
            if 1 <= user_choice_TH <= len(available_columns) and 1 <= user_choice_TA <= len(available_columns) :
                ### Extracting the Token Holder column choice
                chosen_column_index_TH = user_choice_TH - 1  # Adjusted for 0-based indexing
                chosen_column_TH = available_columns[chosen_column_index_TH]
    
                # Extract the chosen column as a Series
                chosen_data_TH = GDF[chosen_column_TH]
                
                ### Extracting the Token Amount column choice ###
                chosen_column_index_TA = user_choice_TA - 1  # Adjusted for 0-based indexing
                chosen_column_TA = available_columns[chosen_column_index_TA]
                
                    
                print(f"\nYou picked the following 'TOKEN HOLDER' column: '{chosen_column_TH}'")
                print(f"\nYou picked the column 'TOKEN AMOUNT' column: '{chosen_column_TA}'\n")
    
                break  # Exit the loop if a valid choice is made
        
        #Else and except conditional blocks continue the while loop until the user has made a valid choice.
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    '''
    Specify the specifc Token Holder from the previously chosen column which will be used for parametising the Smart Contract. 
    '''
    
    print("Contents of the chosen column:\n")
    #print(chosen_data_TH)
    for index, header in enumerate(chosen_data_TH):
        print(f"{index + 1}: {header}")
    
    while True:
        try:
            user_choice_FINAL = int(input("\nPlease ENTER the number of the Token Holder that you wish to MONITOR: "))
            
    
            if 1 <= user_choice_FINAL <= len(available_columns) :
                ### Extracting the Token Holder column choice
                Token_Holder_Index = user_choice_FINAL - 1  # Adjusted for 0-based indexing
                #chosen_column_TH = available_columns[chosen_column_index_TH]
                chosen_Token_Holder_print = chosen_data_TH[Token_Holder_Index]
                    
                print(f"\nYou have chosen the following TOKEN HOLDER to MONITOR: '{chosen_Token_Holder_print}'\n")
               
    
                break  # Exit the loop if a valid choice is made

        #Else and except conditional blocks continue the while loop until the user has made a valid choice.                
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    
    
    
    
    geometry = GDF['geometry'].iloc[Token_Holder_Index] 
    

    #Use Shapely's mapping function to convert the Shapely geometry to a format that Earth Engine understands 
    #(GeoJSON format)
    aoi_geo =  ee.Geometry(mapping(geometry))
    aoi_feature = ee.Feature(aoi_geo)
   
    aoi = ee.FeatureCollection([aoi_feature])
   
    

    #users/thomaswalton45/Tanbi_Wetland_National_Park_Study_Area
    #P:/Dissertation/code/Study_Areas/Tanbi_Wetland_National_Park/Tanbi_Wetland_Park_Outline
    #print(aoi)
    
    
    
    
    '''
    Specify the Date 
    '''
    #1 - Smart Contract(SC) monitoring years requested
    #NOTE: it only works between the years 2018 and 2022
    #"Please input the Start year of the monitoring period"

    #"Please input the End year of the monitoring period"
    
    while True:
        INPUT_SC_mon_before =  input( "\nNOTE: Please understand that currently, only the years between 2018 and 2022 work due to satellite data availability\n\nPlease input the 'Start Year' of the monitoring period here: ")
        INPUT_SC_mon_after   =  input("\nPlease input the 'End Year' of the monitoring period: ")
    
        # Split the inputs by whitespace and remove any leading/trailing spaces
        inputs1 =  INPUT_SC_mon_before.strip().split()
        inputs2 = INPUT_SC_mon_after.strip().split()
        
    
        # Check if both inputs have the required values 
        # .strip() and .split() turn values into a single list item
        # This is why both input checks are list of lists
        if inputs1 in [['2018'], ['2019'], ['2020'],['2021'],['2022']] and inputs2 in [['2018'], ['2019'], ['2020'],['2021'],['2022']]:
            break
        else:
            print("Please enter exactly one of the following years for each input: '2018', '2019', '2020','2021','2022'")
    
        # Use value1 and value2 in your code
        print("\nYou entered:", INPUT_SC_mon_before, "and", INPUT_SC_mon_after) 
    
    #Convert the inputs into strings
    SC_mon_before =  str(INPUT_SC_mon_before)
    SC_mon_after   =  str(INPUT_SC_mon_after) 
    
    #Converting the year to the correct format (yyyy-mm-dd) for GEE
    #Currently, the monitoring periods are configured as annual mosaics
    
    #start of monitoring period
    mon_before_SD = SC_mon_before + '-01-01'
    mon_before_ED = SC_mon_before + '-12-31' 

    #end of monitoring period
    mon_after_SD = SC_mon_after + '-01-01'
    mon_after_ED = SC_mon_after + '-12-31' 

   

    '''
    Specify the agreed change percentage 
    ''' 
    while True:
        try:
            user_input = input("\nEnter a valid change percentage over the agreed monitoring period (a whole number that is either positive or negative): ")
            Agreed_Change_Percentage = int(user_input)
    
            if -100 <=  Agreed_Change_Percentage  <= 100:
                # Valid percentage input
                print(f"\nYou entered the following percentage: {Agreed_Change_Percentage }%")
                break  # Exit the loop if a valid input is provided
        
        #Else and except conditional blocks continue the while loop until the user has made a valid choice.                
            else:
                print("\nInvalid input. Please enter a valid percentage between -100 and 100.")
        except ValueError:
            print("\nInvalid input. Please enter a valid whole number.")
    
    '''
    Specify the agreed amount of tokens that would be given to the monitored token holder 
    ''' 
    while True:
        try:
            user_input = input("\nEnter a valid token amount that will be transferred to the monitored token holder if the change percentage is accepted by all parties (a positive whole number): ")
            Token_Value = int(user_input)
    
            if Token_Value > 0:
                # Valid positive whole number input
                print(f"\nYou entered a valid positive whole number: {Token_Value}")
                break  # Exit the loop if a valid input is provided
       
        #Else and except conditional blocks continue the while loop until the user has made a valid choice.                
            else:
                print("\nInvalid input. Please enter a positive whole number.")
        except ValueError:
            print("\nInvalid input. Please enter a valid whole number.")
    
    
    
    return aoi, mon_before_SD, mon_before_ED, mon_after_SD,  mon_after_ED, GDF, Token_Holder_Index, Agreed_Change_Percentage, Token_Value, user_choice_TH, chosen_column_TA, aoi_geo

'''
Initialise the "SAR_MOSAIC_Collection" function:
     - SAR DATA COLLECTION, FILTERING AND MOSAIC CREATION   
     - A median annual composite is produced and clipped to the monitoring area
     - A Lee spckle filter with 3 × 3 kernel size is applied to the mosaic
'''
def SAR_MOSAIC_Collection (aoi,start,end):
    #try/except block to catch and print any GEE errors that may occur. 
    try:
        # Load Sentinel-1 C-band SAR Ground Range collection (Interferometric Wide Swath Mode, VV, ascending orbit pass, 10m resolution, within the monitored area and monitoring period)
        collectionVV = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
            .filterMetadata('resolution_meters', 'equals', 10) \
            .filterBounds(aoi) \
            .filterDate(start, end) \
            .select('VV')
        
       
        #Create a median composite of the filtered areas. I.e., each pixel will be the median taken from every pixel in the same area within the image collection variable "collectionVV"
        VV_median = collectionVV.median()
    
    
        # Apply Lee Filter to reduce speckle
        SARVV_filtered = VV_median.focal_mean(
            # When radius is set to 1, it means the distance from the center to the edge is 1 pixel in all directions. 
            #So, you have a 3x3 square kernel with a center pixel and 8 surrounding pixels. 
            #When radius is set to 2, it would create a 5x5 square kernel, and so on.
            radius=1, 
            kernelType='square', 
            units='pixels'  # Use 'pixels' for kernel size
            )
        
        # Clip the Sentinel-1 mosaic to your study area
        SARVV_filtered = SARVV_filtered.clip(aoi)
        
        return(SARVV_filtered) #add collectionOP later
    
    #Catches and print GEE errors
    except Exception as e:
        print('Error: %s'%e)
        

'''
Initiliase the "OPTICAL_MOSAIC_Collection function":
    - OPTICAL DATA COLLECTION, FILTERING AND MOSAIC CREATION    
    - A median annual composite is produced and clipped to the monitoring area
    - A cloud mask is applied which utilises the Sentinel-2 cloud probability data set
'''
def OPTICAL_MOSAIC_Collection(aoi,start,end):
    
    try:
        
        # Import Harmonized Sentinel-2 Multispectral Instrument, level-2A and corresponding Sentinel-2 cloud probability collections.

        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        s2c = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        
        # Define a function to filter an image collection by bounds and date.
        def filter_bounds_date(img_col, aoi, start, end):
            return img_col.filterBounds(aoi).filterDate(start, end)
        
        # Filter the collection by AOI and date.
        s2 = filter_bounds_date(s2, aoi, start, end)
        s2c = filter_bounds_date(s2c, aoi, start, end)
                
        # Define a function to join two image collections based on system:index.
        def index_join(colA, colB, prop_name):

            # Create a join filter based on the system:index property.
            join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
        
            # Define the join using saveFirst to get the first matching image from colB.
            joined = ee.ImageCollection(ee.Join.saveFirst(prop_name).apply(
                primary=colA,
                secondary=colB,
                condition=join_filter
            ))
        
            # Define a function to add the bands from colB to each image in the joined collection.
            def add_bands(image):
                prop_image = ee.Image(image.get(prop_name))
                return image.addBands(prop_image)
        
            # Map the add_bands function over the joined collection.
            return joined.map(add_bands)
        
        
        
        # Join the cloud probability collection to the Harmonized Sentinel-2 collection.
        with_cloud_probability = index_join(s2, s2c, 'cloud_probability')
        
        # Define a function to create a cloud masking function.
        def build_mask_function(cloud_prob):
            
            def mask_function(img):
                # Define clouds as pixels having greater than the given cloud probability.
                cloud = img.select('probability').gt(ee.Image(cloud_prob))
        
                # Apply the cloud mask to the image and return it.
                return img.updateMask(cloud.Not())
        
            return mask_function
        
        # Map the cloud masking function over the joined collection, select only the reflectance bands.
        #Currently, the cloud masking function is set to 50 
        mask_clouds = build_mask_function(50)
        
        #Create masked image collection using the previously created functions
        s2_masked = ee.ImageCollection(with_cloud_probability.map(mask_clouds)).select(ee.List.sequence(0, 12))
        
        
        # Calculate the median of overlapping pixels per band.
        median = s2_masked.median()
        
        # Calculate the difference between each image and the median.
        def dif_from_median(img):
            dif = img.subtract(median).pow(ee.Image.constant(2))
            return dif.reduce(ee.Reducer.sum()).addBands(img).copyProperties(img, ['system:time_start'])
        
        dif_from_median = s2_masked.map(dif_from_median)
        
        # Generate a composite image by selecting the pixel that is closest to the median.
        band_names = dif_from_median.first().bandNames()
        band_positions = ee.List.sequence(1, band_names.length().subtract(1))
        mosaic = dif_from_median.reduce(ee.Reducer.min(band_names.length())).select(band_positions, band_names.slice(1)).clip(aoi)
        
        return mosaic

    #Catches and print GEE errors
    except Exception as e:
        print('Error: %s'%e)


'''

   ### RF CLASSIFIER CODEEEE ###

'''

'''
Initilalise the "prepare_classification_data" function:
    - Merge the land cover feature collections 
    - Add a random column to the merged land cover collection 
    - Split the data into training and validation points based on the random values - this is currently set to a 70:30 training:validation split 
    - Returns training (70% of sample) and validation (30% of sample) sets
'''

def prepare_LC_classification_data():
   
    #Define the feature collections Water, Bare_ground, Urban, Mangrove_open_can, and Mangrove_closed_can 
    
    water =               ee.FeatureCollection("users/thomaswalton45/Training_Classes/water_NEW_36_Class")
    
    urban =               ee.FeatureCollection("users/thomaswalton45/Training_Classes/urban_NEW_36_Class")
    
    bare_ground =         ee.FeatureCollection("users/thomaswalton45/Training_Classes/bare_ground_NEW_61_Class")
    
    mangrove_open_can =   ee.FeatureCollection("users/thomaswalton45/Training_Classes/mangrove_open_can_NEW_250_Class") 
    
    mangrove_closed_can = ee.FeatureCollection("users/thomaswalton45/Training_Classes/mangrove_closed_can_NEW_250_Class") 
    
    
    # Merge Feature Collections
    newfc = water.merge(bare_ground).merge(urban).merge(mangrove_open_can).merge(mangrove_closed_can)    

    # Add a random column to the feature collection
    trainingData = newfc.randomColumn()
    
    # Filter training points to less than the given value which is currently set to 0.7. 
    # Hence, 70% of the data will be used for training the random forest model
    trainingPts = trainingData.filter(ee.Filter.lt('random', 0.7))
    
    # Filter validation points to more than the given value which is currently set to 0.7. 
    # Hence, 30% of the data will be used for calidating the random forest model
    validationPts = trainingData.filter(ee.Filter.gte('random', 0.7))


    return trainingPts,validationPts


'''
Initialise the "generate_classification_composite" function:
    - generates a composite mosaic from the VV polarisation SAR and Optical mosaics for classification
    - returns the fused composite, training polygons, land cover attribute label and a list of the bands used in the fused composite. 
'''
def generate_classification_composite(VV_Mosaic, Optical_Mosaic,trainingPts):

    # Combine images into a single image, containing all bands from all images
    opt_fus = ee.Image.cat([VV_Mosaic, Optical_Mosaic])
    

    # Initialise the parameters neccessary for creating the Random forest classifer. 
    
    #the label parameter is the name of the value given to each land cover class. 
    label = 'LC'
    
    #the bands parameter is a list containing the bands required for the classification process.
    bands = [
        'VV', 'B2', 'B3', 'B4', 'B8'] 
    
    #Select the required bands from the fused mosaic
    imageCl = opt_fus.select(bands)
    
    # Overlay the training points on the imagery to get a training sample
    training = imageCl.sampleRegions(
        collection=trainingPts,
        properties=['LC'],
        scale=10,
        tileScale=6
    ).filter(ee.Filter.notNull(['LC']))  # Filters out any features that have a null 'LC' value in the trainingPts collection.
    
    return label, imageCl, bands, training
    

'''
Initialise the "train_RF" function:
        - Trains the Random Forest (RF) classifier using the Optical and VV fused mosaic and bands specified
'''

def train_RF(training, bands,imageCl):

    '''
    **A) Training a Random Forest Classifier
    '''


    #Train the RF classifier 
    
    trainedRf = ee.Classifier.smileRandomForest(**{
        'numberOfTrees': 15  # Specify the number of trees as an option
    }).train(
        features=training, # Specify the training data that you wish to use
        classProperty='LC', # Specify the class property. I.e., the name associated with the value of your training data
        inputProperties=bands # Specify the bands which will be used to train this model.
    )
    
    
    # Classify the image with the same bands used for training
    classifiedRf = imageCl.select(bands).classify(trainedRf)
        
    
    '''
    Return model results
    '''
    
    return trainedRf, classifiedRf


'''
Initialise the "validate_RF" function:
        - Valdiates the RF model trained in the previous "train_RF" function
        - A number of validation metrics are calculated and saved in an excel file format
        - A consistency map is created, shoiwng how consistently a pixel is classified over 5 repetitions
        
'''

def validate_RF(validationPts,imageCl, trainedRf, bands):
    
    '''
    **B)ASSESSING reliability of classification outputs 
    '''
    
    # Extract band pixel values for validation rather than training points
    validation = imageCl.sampleRegions(
        collection=validationPts,
        properties=['LC'],
        scale=10,
    )
    
    # Classify the validation data
    validatedRf = validation.classify(trainedRf)
    
    # Calculate the validation error matrix and accuracy for both classifiers
    validation_confusion_matrix = validatedRf.errorMatrix('LC', 'classification')
    
    print('\n##### CALCULATING VALIDATION ACCURACY METRICS. PLEASE WAIT :) #####')
    
    '''
    ### The following Functions are designed to correctly format the error matrices and validation results (E.g, remove the first row and the first column from every other row)
    ### This was done because a land cover value of 0 was recorded but never used. This was to prevent further complications
    '''

    def format_error_matrix(original_matrix):
        
        # Remove the first row 
        modified_matrix = original_matrix[1:]
        
        modified_matrix = [row[1:] for row in modified_matrix]
        
        #Return modified matrix
        return modified_matrix
    
    def format_consumer_accuracy(original_list):
        
        # Use slicing to remove the first item from the sublist
        
        original_list[0] = original_list[0][1:]
        
        
        return original_list
    
    def format_producer_accuracy(original_list):
        # Use slicing to remove the first list
        original_list = original_list[1:]
        
        # Use list comprehension to flatten the list and wrap it inside a new list
        flattened_list = [item for sublist in original_list for item in sublist]

       
        flattened_list = [flattened_list]
        
        return flattened_list
            
    #Initialise the Confusion Matrix variable
    
    confusion_matrix = format_error_matrix(validation_confusion_matrix.getInfo())
    
    #Initialise the other validation metric variables
    
    overall_accuracy = validation_confusion_matrix.accuracy().getInfo()
    
    kappa_statistic = validation_confusion_matrix.kappa().getInfo()
    F_score_old = validation_confusion_matrix.fscore().getInfo()
    F_score = F_score_old[1:]
    consumer_accuracy = format_consumer_accuracy(validation_confusion_matrix.consumersAccuracy().getInfo())
    producer_accuracy = format_producer_accuracy(validation_confusion_matrix.producersAccuracy().getInfo())
   

    '''
    #Export validation outputs to excel
    '''
    
    # Get the current directory of the Python script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Create the new folder variable
    new_folder_path = os.path.join(current_directory, "Outputs")
    
    #Add the new "Validation_Results" to the "Outputs" folder
    file_name = "Classiifer_Validation_Metric_Results.xlsx"
    file_path = os.path.join(new_folder_path, file_name)

    # Add column headings for the validation score sheets
    new_column_names = {
    0: 'Water',
    1: 'Bare Ground',
    2: 'Urban',
    3: 'Open Canopy Mangrove',
    4: 'Closed Canopy Mangrove'
    }
    
    '''
    Format each validation output to have the correct column and row headings where neccessary 
    They are all converted to dataframes using the pandas library for easier table editing and exporting
    All validation metrics are rounded to 2 significant figures
    '''
    
    #Confusion Matrix
    val_df1 = pd.DataFrame(confusion_matrix)
    val_df1.rename(columns=new_column_names, inplace=True)
    
    #Confusion Accuracy
    val_df2 = pd.DataFrame(consumer_accuracy)
    val_df2 = val_df2.round(2)
    val_df2.rename(columns=new_column_names, inplace=True)
    val_df2.index = ['Consumer Accuracy Score (2 sf)']
    
    #Producer Accuracy
    val_df3 = pd.DataFrame(producer_accuracy)
    val_df3 = val_df3.round(2)
    val_df3.rename(columns=new_column_names, inplace=True)
    val_df3.index = ['Producer Accuracy Score (2 sf)']
    
    #F-Score
    F_score = [F_score]
    val_df4 = pd.DataFrame(F_score)
    val_df4 = val_df4.round(2)
    val_df4.rename(columns=new_column_names, inplace=True)
    val_df4.index = ['F-Score (2 sf)']
    
    #Overall Accuracy and Kappa Statistic
    val_oa = {'Error Matrix Overall Accuracy (2 sf)': [round(overall_accuracy, 2)]}
    val_df5 = pd.DataFrame(val_oa)
    
    val_df5['Kappa Statistic (2 sf)'] = [round(kappa_statistic, 2)]
   
    
    # Specify the Excel file 
    excel_file = file_path 
   
    #for XLSX format, sheet titles must be below 32 characters. 
    #If it is longer then get a warning about a corrupted file when attempting to open
    #Specify the sheet names
    sheet_names = ['Correctly Classified Pixels', 'Overall Accuracy_Kappa Score', 'Consumer Accuracy', 'Producer Accuracy', 'F-score']
    
    # Create an ExcelWriter object
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:
        for i, sheet_name in enumerate(sheet_names):
            # Write each DataFrame to the specified sheet
            if i == 0:
                val_df1.to_excel(writer, sheet_name=sheet_name, index=False)
            elif i == 1:
                val_df5.to_excel(writer, sheet_name=sheet_name, index=False)
            elif i == 2:
                val_df2.to_excel(writer, sheet_name=sheet_name, index=True)
            elif i == 3:
                val_df3.to_excel(writer, sheet_name=sheet_name, index=True)
            elif i == 4:
                val_df4.to_excel(writer, sheet_name=sheet_name, index=True)
            
    
    print("\nSCRIPT UPDATE: The Classiifer_Validation_Metric_Results.xlsx file has been saved to the following directory:\n" +"'" +str(new_folder_path) + "'")
    

    '''
    Consistency map creation
    - Produce a map that shows the consistency of pixel classification 
    - Achieve this by running the classifier across 10 repetitions
    
    '''
    
    # Define the number of runs
    num_runs = 5
    classified_images = []
    
    # Run the classifier multiple times
    for i in range(num_runs):
        # Load the input image you want to classify (e.g., Landsat image)
        input_image = imageCl
    
        # Classify the image with the same classifier and bands used for training
        classified_image = input_image.select(bands).classify(trainedRf)
        
        #Add classified images to the empty classiifed_images list
        classified_images.append(classified_image)
    
    # Create an image collection from the classified results
    classified_collection = ee.ImageCollection(classified_images)
    
    # Calculate the mode (most frequent class) for each pixel
    consistency_map = classified_collection.mode()

    
    '''
    ## Visualise the consistency map (i.e., the 'consitency_map' variable)
    '''
    
    #Initiliase map object using the geemap library
    Map = geemap.Map()
    
    '''
    #Convert hexcode colour to RGB tuple format suitable for the geemap library 
    '''
    
    #Initilaise rgb colors
    rgb_colors = [(247,252,245),(229,245,224),(199,233,192),(161,217,155),(116,196,118),(65,171,93),(35,139,69),(0,109,44),(0,68,27), (0,0,0)]
    
    # Convert RGB tuples to hexadecimal color codes as strings
    hex_colors = ['#%02x%02x%02x' % rgb for rgb in rgb_colors]
    
        
    # Define visualization parameters for classification display
    classVis = {
        'min': 0,
        'max': num_runs,
        'palette': hex_colors
    }
        
    #Add the map object, clipping it to the monitoring area and using the visualisation paramters created previously
    Map.addLayer(
        consistency_map.clipToCollection(aoi), classVis, 'Consistency Map')
    
    #Center the map object 
    Map.centerObject(aoi, 12)
    
    #Get the current directory of the file
    current_directory = os.path.dirname(os.path.abspath(__file__))

    #Save the map
    Map.save(os.path.join(current_directory,"Outputs","Consistency_Map_Output.html"))
    
    #End the function, returning the required outputs
    return validatedRf,validation_confusion_matrix, consistency_map


'''

Initialise the "RF_Model_Creation_and_Validation" function:
    -Creates and validates the RF model used for monitoring mangrove cover change within the Smart Contract
    -This model will then be applied to the monitoring start and end period
    -Returns the RF model and validation metrics
    
'''
def RF_Model_Creation_and_Validation(aoi): 

    #Intiialise the start and end date range for mosaic creation to train the RF model 
    training_start_date = '2020-01-01'
    training_end_date = '2020-12-31'
    
    
    '''
    ### CALL BOTH MOSAIC FUNCTIONS HERE ####
    '''

    ### SAR ####
    VV_Mosaic = SAR_MOSAIC_Collection(aoi,training_start_date,training_end_date)
    #print("Band names for SAR MOSAIC:", VV_Mosaic.bandNames().getInfo())
    
    
    ### OPTICAL ####
    Optical_Mosaic = OPTICAL_MOSAIC_Collection(aoi,training_start_date,training_end_date)

    '''
    Visualise RF Classifier Mosaic inputs for training so the user can inspect and validate the results
    It is called here specifically to prevent multiple mosaic generations.
    '''

    ##### OPTICAL #####
    
    #Initialise the map object
    Map = geemap.Map()

    vis_params = {
        'min': 0,
        'max': 2500,
        'bands': ['B4', 'B3', 'B2'],
    }

    #Add the optical mosaic to the map object 
    Map.addLayer(Optical_Mosaic, vis_params, 'Classifier_Training_Optical_Mosaic')

    # Center the map on the area of interest
    Map.centerObject(aoi, 12)
    
    #Get current directory of the file
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Use the join function to extend to the directory to the map output file 
    Map.save(os.path.join(current_directory,"Outputs","Classifier_Training_Optical_Mosaic.html"))
    
    #Print update message for the user 
    print("\nSCRIPT UPDATE: The Classifier_Training_Optical_Mosaic.html file has been saved to the following directory:\n" +"'" +str(current_directory) +"\Outputs" + "'")

    ##### SAR #####

    #Initialise the map object
    Map = geemap.Map()
    
    vis_params = {
        'min': -15,
        'max': 0,
        #'gamma': 2,
        'bands': ['VV'],
    }
    
    #Add the SAR filtered images to the Map
    Map.addLayer(VV_Mosaic, vis_params, "Classifier_Training_SAR_VV_Mosaic")

    # Center the map on the area of interest
    Map.centerObject(aoi, 12)

    #Get current directory of the file
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Use the join function to extend to the directory to the map output file. 
    Map.save(os.path.join(current_directory,"Outputs","Classifier_Training_SAR_VV_Mosaic.html"))
    
    #Print update message for the user
    print("\nSCRIPT UPDATE: The Classifier_Training_SAR_VV_Mosaic.html file has been saved to the following directory:\n" +"'" +str(current_directory) +"\Outputs" + "'")

    '''
    End of Mosaic visualisation
    '''

    
    '''
    ###CALL YOUR FUNCTIONS - Random Forest (RF)
    '''
    
    #Get training and validation points
    trainingPts,validationPts = prepare_LC_classification_data()
    

    #Create the fused classification mosaic
    label, imageCl, bands, training = generate_classification_composite(VV_Mosaic, Optical_Mosaic, trainingPts)
    
    #train the RF model
    trainedRf, classifiedRf = train_RF(training, bands, imageCl) 
    
    #validate the RF model
    validatedRf,validationAccuracyRf, consistency_map = validate_RF(validationPts,imageCl, trainedRf, bands)
    
    
    #Return the neccessary outputs and end the function
    return classifiedRf, trainedRf, trainingPts, consistency_map, VV_Mosaic, Optical_Mosaic


'''
Initialise the "change_detection" function:
    - This function is a combination of the previously created functions
    - It will apply the trained RF model to the start and end monitoring dates
    - These dates were input by the user and will be created as annual mosaics for that year
    - The start and end year SAR and Optical mosaics will be saved and stored for the user to validate
'''

def change_detection(aoi, mon_before_SD, mon_before_ED, mon_after_SD, mon_after_ED, classifiedRf, trainedRf, trainingPts):
    
    #Confirm with the user that the previous input dates will be applied to this function
    print("\nStart of monitoring period:" , mon_before_SD , "-" , mon_before_ED , "\nEnd of monitoring period  :", mon_after_SD,"-", mon_after_ED)
    
    '''
    Create the Optical and SAR mosaics for the start and end monitoring year periods
    '''
    
    #Start of monitoring period
    START_MONITOR_SARVV_mosaic = SAR_MOSAIC_Collection(aoi, mon_before_SD, mon_before_ED)
    
    START_MONITOR_Optical_mosaic = OPTICAL_MOSAIC_Collection (aoi, mon_before_SD, mon_before_ED)
    
    #End of monitoring period 
    END_MONITOR_SARVV_mosaic = SAR_MOSAIC_Collection(aoi, mon_after_SD, mon_after_ED)
    
    END_MONITOR_Optical_mosaic = OPTICAL_MOSAIC_Collection (aoi, mon_after_SD, mon_after_ED)
    
    
    '''
    Visualise SAR VV polarisation and Optical Mosaics for the start and end monitoring periods so the user can inspect and validate the results
    '''
    
    # Get the current directory of the Python script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    '''
    Create a new folder to store 4 mosaics. I.e., the start and end SAR and optical mosaics
    '''
    # Specify the name of the new folder
    new_folder_name_monitor = "Start_and_End_Monitoring_Mosaics"

    # Create the new folder
    new_folder_path_monitor = os.path.join(current_directory,"Outputs", new_folder_name_monitor)

    # Check if the folder already exists, and create it if not
    if not os.path.exists(new_folder_path_monitor):
        os.makedirs(new_folder_path_monitor)
        print(f"\nFolder '{new_folder_name_monitor}' created at {new_folder_path}")
    else:
        print(f"\nFolder '{new_folder_name_monitor}' already exists at {new_folder_path}. No new folder will be created.")
    
    
    
    '''
    ##### OPTICAL #####
    '''
    
    
    vis_params_op = {
        'min': 0,
        'max': 2500,
        'bands': ['B4', 'B3', 'B2'],
    }

    '''
    # Start of Monitoring Period Optical
    '''
    #Intiialise map object
    Map = geemap.Map()
    
    #Add Optical mosaic 
    Map.addLayer(START_MONITOR_Optical_mosaic, vis_params_op, 'Start_of_Monitoring_Period_Optical_Mosaic')

    # Center the map on the area of interest
    Map.centerObject(aoi, 12)

    # Use the join function to extend to the directory to the consistency map output file. 
    Map.save(os.path.join(current_directory,"Outputs","Start_and_End_Monitoring_Mosaics","START_of_Monitoring_Period_OPTICAL_Mosaic.html"))
    
    '''
    #End of Monitoring Period Optical
    '''
    #Intiialise map object
    Map = geemap.Map()
    
    #Add Optical mosaic    
    Map.addLayer(END_MONITOR_Optical_mosaic, vis_params_op, 'End_of_Monitoring_Period_Optical_Mosaic')

    # Center the map on the area of interest
    Map.centerObject(aoi, 12)

    # Use the join function to extend to the directory to the consistency map output file. 
    Map.save(os.path.join(current_directory,"Outputs","Start_and_End_Monitoring_Mosaics","END_of_Monitoring_Period_OPTICAL_Mosaic.html"))
    
    

    '''
    ##### SAR #####
    '''

    vis_params_sar = {
        'min': -15,
        'max': 0,
        'bands': ['VV'],
    }
    
    '''
    #Start of monitoring period SAR
    '''
    #Intiialise map object
    Map = geemap.Map()
    
    #Clip SAR image to study area
    START_MONITOR_SARVV_mosaic = START_MONITOR_SARVV_mosaic.clip(aoi)
    
    #Add SAR mosaic
    Map.addLayer(START_MONITOR_SARVV_mosaic, vis_params_sar, "START_of_Monitoring_Period_SAR_VV_Mosaic")

    # Center the map on the area of interest
    Map.centerObject(aoi, 12)

    # Use the join function to extend to the directory to the consistency map output file. 
    Map.save(os.path.join(current_directory,"Outputs","Start_and_End_Monitoring_Mosaics","START_of_Monitoring_Period_SAR_VV_Mosaic.html"))
    

    '''
    #End of monitoring period SAR
    '''
    #Intiialise map object
    Map = geemap.Map()
    
    #Clip SAR image to study area
    END_MONITOR_SARVV_mosaic = START_MONITOR_SARVV_mosaic.clip(aoi)
    
    #Add SAR mosaic
    Map.addLayer(END_MONITOR_SARVV_mosaic, vis_params_sar, "END_of_Monitoring_Period_SAR_VV_Mosaic")

    # Center the map on the area of interest
    Map.centerObject(aoi, 12)

    # Use the join function to extend to the directory to the consistency map output file. 
    Map.save(os.path.join(current_directory,"Outputs","Start_and_End_Monitoring_Mosaics","END_of_Monitoring_Period_SAR_VV_Mosaic.html"))

    
    print("\nSCRIPT UPDATE: The Start and End Monitoring Period Mosaics have been saved to the following directory:\n" +"'" +str(current_directory) +"\Outputs\Start_and_End_Monitoring_Mosaics" + "'")
    
    
    '''
    Generate the fused classification mosaics for the start and end monitoring year periods
    '''    
    #Start of monitoring period
    START_label, START_imageCl, START_bands, START_training = generate_classification_composite(START_MONITOR_SARVV_mosaic,  START_MONITOR_Optical_mosaic, trainingPts)
    
    #End of monitoring period 
    END_label, END_imageCl, END_bands, END_training = generate_classification_composite(END_MONITOR_SARVV_mosaic, END_MONITOR_Optical_mosaic, trainingPts)

    '''
    Apply the trained RF classifier to the start and end monitoring period annual mosaics 
    '''
    
    # Classify the image with the same bands used for training
    START_classifiedRf = START_imageCl.select(START_bands).classify(trainedRf)
    
    # Classify the image with the same bands used for training
    END_classifiedRf = END_imageCl.select(END_bands).classify(trainedRf)
    
    
    '''
    Creating the mask neccessary for mangrove cover pixel counts
    -I.e., removing all other class values
    '''
    
    # Define the specific class values you're interested in for the mask 
    # Specific classes: 4 = Open Canopy, 5 = Closed Canopy - The client currently only wants to measure mangrove cover percentage change
    specific_class_values = [4,5]
   
    #START of monitoring period mask 
    
    #Creating a mask that only selects a specific class value (I will be using 4 and 5 for open and closed canopy)
    # Create an initial mask to select no pixels (all set to 0)
    START_mask = ee.Image(0)
    
    # Loop through the specific class values and update the mask
    for value in specific_class_values:
        #Check if the current value iteration is equal to the ones initiliased in specific_class_values
        START_class_mask = START_classifiedRf.eq(value)
        # the .bitwiseOr() method creates a composite mask by bitwise OR'ing the individual masks
        # using .bitwiseOr() combines multiple binary masks, each representing a specific class of interest
        #The result will be a mask where a pixel is selected (set to 1) if it belongs to any of the specified classes in the list
        START_mask = START_mask.bitwiseOr(START_class_mask)
        
    #END of monitoring period mask:
        
    END_mask = ee.Image(0)
    
    # Loop through the specific class values and update the mask
    for value in specific_class_values:
        #Check if the current value iteration is equal to the ones initiliased in specific_class_values
        END_class_mask = END_classifiedRf.eq(value)
        # the .bitwiseOr() method creates a composite mask by bitwise OR'ing the individual masks
        # using .bitwiseOr() combines multiple binary masks, each representing a specific class of interest
        #The result will be a mask where a pixel is selected (set to 1) if it belongs to any of the specified classes in the list
        END_mask = END_mask.bitwiseOr(END_class_mask)
    
    

    '''
    Calculating Mangrove pixel counts 
    '''
   
    # Apply the mask to the original images to extract pixels of the specific class
    START_result_image = START_classifiedRf.updateMask(START_mask)
    
    END_result_image = END_classifiedRf.updateMask(END_mask)
    
    '''
    Try-Except block where if the tilescale paramter is removed and a computation error occurs, the monitoring area will be split into a smaller cells to be processed individually
    This process takes a long time so for now, the tilescale paramter is kept 
    '''
    try:
 
        #Obtain pixel count for the START_result_image
        START_pixel_count = START_result_image.reduceRegion(
            reducer=ee.Reducer.count(),  # Compute the sum of mask values
            geometry= aoi, # Use the defined AOI
            bestEffort=True,  # Try to compute even if it's large
            maxPixels=1e14,  # Set a very large maximum number of pixels that the API will process
            
            #Remove this parameter if you want to use the batch processing function
            tileScale=5, # the tileScale parameter is used when performing computations that involve large amounts of data, such as reducing or mapping over an image. It controls the division of the computation into smaller "tiles" to manage memory and processing efficiency.
            
            scale=10  # Adjust the scale based on your image resolution
        )
        
        #Obtain pixel count for the END_result_image
        END_pixel_count = END_result_image.reduceRegion(
             reducer=ee.Reducer.count(),  # Compute the sum of mask values
             geometry= aoi, # Use the defined AOI
             bestEffort=True,  # Try to compute even if it's large
             maxPixels=1e14,  # Set a very large maximum number of pixels that the API will process
             
             #Remove this parameter if you want to use the batch processing function
             tileScale=5, # the tileScale parameter is used when performing computations that involve large amounts of data, such as reducing or mapping over an image. It controls the division of the computation into smaller "tiles" to manage memory and processing efficiency.
             
             scale=10  # Adjust the scale based on your image resolution
         )
    
    
        # Get the pixel count for the specific class for the start and end monitoring periods
        START_count = START_pixel_count.get("classification")
        
        END_count = END_pixel_count.get("classification")
        
        #Pull pixel counts from server side into the client side (i.e., this script)
        START_count_local = START_count.getInfo() 
        END_count_local = END_count.getInfo()
        
    #If an exception occurs, try again but use a batch processing method. I.e., the area is split into smaller cells that are processed individually.
    except ee.EEException:
        try:
            # Get the single multipolygon from the Feature Collection
            multipolygon = aoi.geometry()
        
            # Compute the bounding box (bounding rectangle) for the multipolygon
            bounding_box = multipolygon.bounds()

            # Extract the coordinates of the bounding box
            coordinates = ee.List(bounding_box.coordinates().get(0))

            # Find the minimum and maximum longitude and latitude values
            lon_list = coordinates.map(lambda point: ee.List(point).get(0))
            lat_list = coordinates.map(lambda point: ee.List(point).get(1))
            min_lon = lon_list.reduce(ee.Reducer.min())
            max_lon = lon_list.reduce(ee.Reducer.max())
            min_lat = lat_list.reduce(ee.Reducer.min())
            max_lat = lat_list.reduce(ee.Reducer.max())

            # Print the max and min longitude and latitude
            print(f"Max Longitude (lon): {max_lon.getInfo()}, Max Latitude (lat): {max_lat.getInfo()}")
            print(f"Min Longitude (lon): {min_lon.getInfo()}, Min Latitude (lat): {min_lat.getInfo()}")

            # Use the math libary to round the minium values down (.floor) and max values up (.ceil)
            # The bounding geometries need to be set to whole numbers for this function to work
            min_lon = math.floor(min_lon.getInfo() )
            max_lon = math.ceil(max_lon.getInfo() ) 
            min_lat = math.floor(min_lat.getInfo() )
            max_lat = math.ceil(max_lat.getInfo() ) 


            # Define the size of the grid cells (e.g., 0.1 degrees, adjust as needed)
            # the smaller you make it, the more cells you will make which will split up the processing further - adjust based on area sizes. 
            #E.g., 0.1 gives 100 polygons for the monitoring area used
            grid_size = 0.1

            # Create a grid of smaller polygons
            grid = ee.FeatureCollection([])

            # Loop through the longitude and latitude ranges to create grid cells bsed on the grid_size variable
            for lon in range(int(min_lon / grid_size), int(max_lon / grid_size)):
                for lat in range(int(min_lat / grid_size), int(max_lat / grid_size)):
                    cell = ee.Geometry.Rectangle([lon * grid_size, lat * grid_size, (lon + 1) * grid_size, (lat + 1) * grid_size])
                    cell = cell.intersection(aoi.geometry())  # Clip the cell to the specified range 
                    grid = grid.merge(cell)
                    
            #Localise cloud variable to alert the user to the number of cells their monitoring area has been split into
            grid_size = grid.size().getInfo()

            # Print the number of smaller polygons in the grid
            print("Number of smaller polygons to be analysed:", grid_size)


            '''
            #Loop through features in the "grid" feature collection. I.e., every smaller polygon created
            #Identify the number of pixels of the specified class - this is defined in the 'result_image' variable
            #Add this to the total_pixel_count 
            '''
            #Initialise the counts for the start and end monitoring periods.
            START_count_local = 0
            END_count_local = 0
            
            #Intilaise the polygon count variable - this will keep track fo the number of cells that have been processed
            polygon_count = 1
            
            grid_collection = grid.getInfo()
            # Loop through the features and call the process_feature function
            for feature in grid_collection['features']:
                
                
                # Access the geometry of the grid cell
                grid_geometry = ee.Geometry(feature['geometry'])
                
                # Perform operations using grid_geometry
                # For example, count pixels within this grid cell in result_image
                START_pixel_count = START_result_image.reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=grid_geometry,
                    bestEffort=True,
                    maxPixels=1e14,
                    scale=10
                )
                
                END_pixel_count = END_result_image.reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=grid_geometry,
                    bestEffort=True,
                    maxPixels=1e14,
                    scale=10
                )
                
                
                # Get the pixel count for the specific class and localise it with the .getInfo() function
                START_count = START_pixel_count.get("classification")
                START_count = START_count.getInfo()
                
                # Get the pixel count for the specific class and localise it with the .getInfo() function
                END_count = END_pixel_count.get("classification")
                END_count = END_count.getInfo()
                
                
                # Print the pixel count
                print("Classified",polygon_count,"out of",grid_size ,"polygons","for the start and end monitoring period")
                
                #Add pixel count to current pixel count
                START_count_local = START_count_local + START_count
                
                #Add pixel count to current pixel count
                END_count_local = END_count_local + END_count
                
                #Add 1 to the total polygon count
                polygon_count = polygon_count+1
        
        #If any other errors occur after this process, then print the error
        except Exception as e:
                print('Error: %s'%e)
    
    
    '''
    Print the pixel counts
    '''
    
    #Start of monitoring period
    print("Pixel count for the START of monitoring period in Mangrove class/classes", specific_class_values, ":", START_count_local)
    
    #End of monitoring period
    print("Pixel count for the END of monitoring period in Mangrove class/classes", specific_class_values, "  :", END_count_local)
    
    '''
    Calculating percentage change 
    '''
    
    # Function to calculate the percentage
    def percent(a, b) :
        
        #Calculate the percentage (this does not add a "%" symbol)
        result = int(((b - a) * 100) / a)
        
        
        return result
    
    
    #Calculate change percentage
    change_percentage = percent(START_count_local, END_count_local)
    print("Change percentage:", str(change_percentage) + "%")
   
    #Returned required variables and exit function
    return change_percentage, START_result_image, END_result_image, START_MONITOR_SARVV_mosaic, START_MONITOR_Optical_mosaic, END_MONITOR_SARVV_mosaic, END_MONITOR_Optical_mosaic


'''
Initialise the "offline_output_creation" function:
    - This function will be called if the user wishes to have offline copies of the validation map outputs
    - Due to current GEE processing limitations, this POC can only export the chosen client monitoring site outputs at a 20m resolution
    - However, the Gee Map package used currently has bugs associated with map element outputs
    - Further, these outputs will not affect the final pixel count while still offering an offline version with more spatial context for distribution 
    - This makes these offline outputs neccessary for those without internet access and for those that need context to validation outputs. E.g., a legend bar for the consistency map and classified monitoring areas
'''

def offline_output_creation(START_MONITOR_SARVV_mosaic, START_MONITOR_Optical_mosaic, END_MONITOR_SARVV_mosaic, END_MONITOR_Optical_mosaic, VV_Mosaic_RF_Train, Optical_Mosaic_RF_Train, consistency_map, START_result_image, END_result_image, aoi_geo):
    
    
    '''
    Create the "Offline Outputs" folder to put all map and data outputs
    '''

    # Specify the name of the new folder
    new_folder_name = "Offline_Map_Outputs"

    # Create the new folder
    new_folder_path = os.path.join(current_directory, new_folder_name)

    # Check if the folder already exists, and create it if not
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"\nFolder '{new_folder_name}' created at {new_folder_path}\n")
    else:
        print(f"\nFolder '{new_folder_name}' already exists at {new_folder_path}. No new folder will be created.\n")
        
  
    '''
    Create a new folder to store 4 mosaics. I.e., the start and end SAR and optical mosaics
    '''
    # Specify the name of the new folder
    new_folder_name_monitor = "Start_and_End_Monitoring_Mosaics"

    # Create the new folder
    new_folder_path_monitor = os.path.join(current_directory,"Offline_Map_Outputs", new_folder_name_monitor)

    # Check if the folder already exists, and create it if not
    if not os.path.exists(new_folder_path_monitor):
        os.makedirs(new_folder_path_monitor)
        print(f"\nFolder '{new_folder_name_monitor}' created at {new_folder_path}")
    else:
        print(f"\nFolder '{new_folder_name_monitor}' already exists at {new_folder_path}. No new folder will be created.\n")  

    
    
    print("\nSCRIPT UPDATE: Downloading all map outputs now...\n")


    '''
    Download and Add Map Elements to the SAR and Optical Mosaics Used to Train the Classifier
    '''    
    
    #Optical Mosaic Export
    
    #Select specific bands from the image collection for the RGB composition 
    Optical_Mosaic_RF_Train = Optical_Mosaic_RF_Train.select(['B2', 'B3', 'B4'])  # Replace bands with your desired ones
    
    
    geemap.ee_export_image(
        Optical_Mosaic_RF_Train,
        filename=os.path.join(current_directory, "Offline_Map_Outputs", 'Classifier_Training_Optical_Mosaic.tif'),
        scale= 20,  # Change the scale to 20 for 20 meters per pixel resolution
        region=aoi_geo,
    )
    
    
    print("\n")
 
    #SAR Mosaic Export
    
    geemap.ee_export_image(
        VV_Mosaic_RF_Train,
        filename=os.path.join(current_directory, "Offline_Map_Outputs", 'Classifier_Training_SAR_Mosaic.tif'),
        scale= 20,  # Change the scale to 20 for 20 meters per pixel resolution
        region=aoi_geo,
    )
    
    print("\n")
    '''
    #Download and add map elements to the Consistency Map
    '''    
    
    #Consistency Map Export
    
    geemap.ee_export_image(
        consistency_map,
        filename=os.path.join(current_directory, "Offline_Map_Outputs", 'Consistency_Map.tif'),
        scale= 20,  # Change the scale to 20 for 20 meters per pixel resolution
        region=aoi_geo,
    )
    print("\n")
    
    '''
    #Download and add map elements to the Start and End Monitoring Mosaics
    '''    
    
    #Start Optical Mosaic Export
    
    #Select specific bands from the image collection for the RGB composition 
    START_MONITOR_Optical_mosaic = START_MONITOR_Optical_mosaic.select(['B2', 'B3', 'B4'])  # Replace bands with your desired ones
    
    geemap.ee_export_image(
        START_MONITOR_Optical_mosaic,
        filename=os.path.join(current_directory, "Offline_Map_Outputs", "Start_and_End_Monitoring_Mosaics", 'START_of_Monitoring_Period_OPTICAL_Mosaic.tif'),
        scale= 20,  # Change the scale to 20 for 20 meters per pixel resolution
        region=aoi_geo,
    )
    
    print("\n")
 
    #End Optical Mosaic Export
    
    #Select specific bands from the image collection for the RGB composition 
    END_MONITOR_Optical_mosaic= END_MONITOR_Optical_mosaic.select(['B2', 'B3', 'B4'])  # Replace bands with your desired ones
    
    geemap.ee_export_image(
        END_MONITOR_Optical_mosaic,
        filename=os.path.join(current_directory, "Offline_Map_Outputs", "Start_and_End_Monitoring_Mosaics", 'END_of_Monitoring_Period_OPTICAL_Mosaic.tif'),
        scale= 20, # Change the scale to 20 for 20 meters per pixel resolution
        region=aoi_geo,
    )
    
    print("\n")
 
    #Start SAR Mosaic Export
    
    geemap.ee_export_image(
        START_MONITOR_SARVV_mosaic,
        filename=os.path.join(current_directory, "Offline_Map_Outputs","Start_and_End_Monitoring_Mosaics", 'START_of_Monitoring_Period_SAR_Mosaic.tif'),
        scale= 20,  # Change the scale to 20 for 20 meters per pixel resolution
        region=aoi_geo,
    )
    
    print("\n")
    
    #End SAR Mosaic Export 
    
    geemap.ee_export_image(
        END_MONITOR_SARVV_mosaic,
        filename=os.path.join(current_directory, "Offline_Map_Outputs", "Start_and_End_Monitoring_Mosaics", 'END_of_Monitoring_Period_SAR_Mosaic.tif'),
        scale= 20,  # Change the scale to 20 for 20 meters per pixel resolution
        region=aoi_geo,
    )
    
    print("\n")
    
    '''
   #Download and add map elements to the Classified Maps for the Start and End Monitoring Period
    '''    
    
     
    #Start of Monitoring Period Classified Mosaic Export
    
    geemap.ee_export_image(
        START_result_image,
        filename=os.path.join(current_directory, "Offline_Map_Outputs", 'START_Monitoring_Period_Classified.tif'),
        scale= 20, # Change the scale to 20 for 20 meters per pixel resolution
        region=aoi_geo,
    )
    print("\n")
 
    
    #End of Monitoring Period Classified Mosaic Export
    
    geemap.ee_export_image(
        END_result_image,
        filename=os.path.join(current_directory, "Offline_Map_Outputs", 'END_Monitoring_Period_Classified.tif'),
        scale= 20, # Change the scale to 20 for 20 meters per pixel resolution
        region=aoi_geo,
    )
    print("\n")
    
    
 
    print("\nSCRIPT UPDATE: All required outputs have been downloaded. Now adding map elements...")
    
    '''
    ####
    Create a new folder to store all outputs with added elements
    ####
    '''
    
    # Specify the name of the new folder
    new_folder_name_monitor = "Outputs_With_Added_Map_Elements"

    # Create the new folder
    new_folder_path_monitor = os.path.join(current_directory,"Offline_Map_Outputs", new_folder_name_monitor)

    # Check if the folder already exists, and create it if not
    if not os.path.exists(new_folder_path_monitor):
        os.makedirs(new_folder_path_monitor)
        print(f"\nFolder '{new_folder_name_monitor}' created at {new_folder_path}")
    else:
        print(f"\nFolder '{new_folder_name_monitor}' already exists at {new_folder_path}. No new folder will be created.\n")  
    
    
    '''
    ###
    All Optical Mosaic Outputs
    ###
    '''
    # Open the optical mosaic used to train the RF classifier 
    
    with rasterio.open(os.path.join(current_directory, "Offline_Map_Outputs", "Classifier_Training_Optical_Mosaic.tif")) as opt:
        # Read the individual bands for Red, Green, and Blue channels
        red = opt.read(3)   # Red band
        green = opt.read(2) # Green band
        blue = opt.read(1)  # Blue band

        # Normalize the bands based on visualization parameters
        # You can adjust additional visualization parameters here if needed
        min_value = 0
        max_value = 2500
        
        #Create normalised band values
        red = (red - min_value) / (max_value - min_value)
        green = (green - min_value) / (max_value - min_value)
        blue = (blue - min_value) / (max_value - min_value)

        # Ensure the pixel values are within the valid range (0 to 1)
        red = red.clip(0, 1)
        green = green.clip(0, 1)
        blue = blue.clip(0, 1)

        # Stack the bands together to create an RGB composite
        rgb = np.dstack([red, green, blue])

        #Initialise the plot for export
        plt.figure(figsize=(8, 8))        
        
        # Display the RGB composite image on the subplot
        plt.imshow(rgb)
        
        #Add a plot title
        plt.title('Optical Mosaic Used to Train the Classifier Model',fontweight='bold', style='italic')

        # Add a spatially accurate scale bar to the plot
        #Adjusting dx based on the actual spatial resolution of your data ensures that the scale bar 
        #accurately represents the distances within the plot according to the raster's spatial characteristics.
        scalebar = ScaleBar(dx= 20, units="m", location="lower right") # dx = 20 due to the current spatial resolution (I.e., 20m)
        plt.gca().add_artist(scalebar)


        # Add a north arrow
        x, y, arrow_length = 0.97, 0.99, 0.1
        plt.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                     arrowprops=dict(facecolor='black', width=5, headwidth=15),
                     ha='center', va='center', fontsize=20,  xycoords='axes fraction')
        
        
        # Turn off the axis (axes) for the subplot
        plt.axis('off')
        
        # Save the plot
        plt.savefig(os.path.join(current_directory, "Offline_Map_Outputs", "Outputs_With_Added_Map_Elements", "Classifier_Training_Optical_Mosaic.png"), bbox_inches='tight')  # Save the plot as PNG file
        
        #Close the plot
        plt.close()
        
        #Close the raster 
        opt.close()
    
    
    # Open the Start of monitoring period optical mosaic
    
    with rasterio.open(os.path.join(current_directory, "Offline_Map_Outputs", "Start_and_End_Monitoring_Mosaics", "START_of_Monitoring_Period_OPTICAL_Mosaic.tif")) as opt:
        # Read the individual bands for Red, Green, and Blue channels
        red = opt.read(3)   # Red band
        green = opt.read(2) # Green band
        blue = opt.read(1)  # Blue band

        # Normalize the bands based on visualization parameters
        # You can adjust additional visualization parameters here if needed
        min_value = 0
        max_value = 2500
        
        #Create the normalised bands
        red = (red - min_value) / (max_value - min_value)
        green = (green - min_value) / (max_value - min_value)
        blue = (blue - min_value) / (max_value - min_value)

        # Ensure the pixel values are within the valid range (0 to 1)
        red = red.clip(0, 1)
        green = green.clip(0, 1)
        blue = blue.clip(0, 1)

        # Stack the bands together to create an RGB composite
        rgb = np.dstack([red, green, blue])

        #Initialise the plot for export
        plt.figure(figsize=(8, 8))

        # Display the RGB composite image on the subplot
        plt.imshow(rgb)
        
        #Add a plot title
        plt.title('Optical Mosaic Captured at the Start of the Monitoring Period',fontweight='bold', style='italic')

        # Add a spatially accurate scale bar to the plot
        #Adjusting dx based on the actual spatial resolution of your data ensures that the scale bar 
        #accurately represents the distances within the plot according to the raster's spatial characteristics.
        scalebar = ScaleBar(dx= 20, units="m", location="lower right") # dx = 20 due to the current spatial resolution (I.e., 20m)
        plt.gca().add_artist(scalebar)


        # Add a north arrow
        x, y, arrow_length = 0.97, 0.99, 0.1
        plt.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=20,  xycoords='axes fraction')
        
        # Turn off the axis (axes) for the subplot
        plt.axis('off')
        
        # Save the plot
        plt.savefig(os.path.join(current_directory, "Offline_Map_Outputs", "Outputs_With_Added_Map_Elements", "START_of_Monitoring_Period_OPTICAL_Mosaic.png"), bbox_inches='tight')  # Save the plot as PNG file
        
        #Close the plot
        plt.close()
        
        #Close the raster 
        opt.close()
    
    
    # Open the End of monitoring period optical mosaic
    with rasterio.open(os.path.join(current_directory, "Offline_Map_Outputs", "Start_and_End_Monitoring_Mosaics", "END_of_Monitoring_Period_OPTICAL_Mosaic.tif")) as opt:
        # Read the individual bands for Red, Green, and Blue channels
        red = opt.read(3)   # Red band
        green = opt.read(2) # Green band
        blue = opt.read(1)  # Blue band

        # Normalize the bands based on visualization parameters
        # You can adjust additional visualization parameters here if needed
        min_value = 0
        max_value = 2500
        
        #Create the normalised bands
        red = (red - min_value) / (max_value - min_value)
        green = (green - min_value) / (max_value - min_value)
        blue = (blue - min_value) / (max_value - min_value)

        # Ensure the pixel values are within the valid range (0 to 1)
        red = red.clip(0, 1)
        green = green.clip(0, 1)
        blue = blue.clip(0, 1)

        # Stack the bands together to create an RGB composite
        rgb = np.dstack([red, green, blue])
        
        #Initialise the plot for export
        plt.figure(figsize=(8, 8))        

        # Display the RGB composite image on the subplot
        plt.imshow(rgb)
        
        #Add a plot title
        plt.title('Optical Mosaic Captured at the End of the Monitoring Period',fontweight='bold', style='italic')

        # Add a spatially accurate scale bar to the plot
        #Adjusting dx based on the actual spatial resolution of your data ensures that the scale bar 
        #accurately represents the distances within the plot according to the raster's spatial characteristics.
        scalebar = ScaleBar(dx= 20, units="m", location="lower right") # dx = 20 due to the current spatial resolution (I.e., 20m)
        plt.gca().add_artist(scalebar)

        # Add a north arrow
        x, y, arrow_length = 0.97, 0.99, 0.1
        plt.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                     arrowprops=dict(facecolor='black', width=5, headwidth=15),
                     ha='center', va='center', fontsize=20,  xycoords='axes fraction')
        
        # Turn off the axis (axes) for the subplot
        plt.axis('off')
        
        # Save the plot
        plt.savefig(os.path.join(current_directory, "Offline_Map_Outputs", "Outputs_With_Added_Map_Elements", "END_of_Monitoring_Period_OPTICAL_Mosaic.png"), bbox_inches='tight')  # Save the plot as PNG file
        
        #Close the plot
        plt.close()
        
        #Close the raster 
        opt.close()

    
    '''
    ###
    All SAR Mosaic Outputs
    ###
    '''
    
    # Open the VV mosiac used to train the RF classifier using rasterio
    with rasterio.open(os.path.join(current_directory, "Offline_Map_Outputs", "Classifier_Training_SAR_Mosaic.tif")) as sar:

        # Read the VV band data
        vv_band = sar.read(1)  # Change '1' to the appropriate band index if VV is in a different band
        
        # Apply visualization parameters
        # You can adjust additional visualization parameters here if needed
        vis_params = {
            'vmin': -15,
            'vmax': 0,
        }
        
        # Plotting the data
        plt.figure(figsize=(8, 8))
        
        plt.imshow(vv_band, cmap='gray', **vis_params)
        
        #Add a title
        plt.title('SAR Mosaic Used to Train the Classifier Model',fontweight='bold', style='italic')
    
        #Intilaise colorbar (cbar variable)
        cbar = plt.colorbar()
        
        # Set the tick labels for the color bar
        cbar.set_ticks([vis_params['vmin'], vis_params['vmax']])
        cbar.set_ticklabels(['High\nNegative\nBackscatter\nValue', 'High\nPositive\nBackscatter\nValue'],multialignment='center')
        
        # Add a spatially accurate scale bar to the plot
        #Adjusting dx based on the actual spatial resolution of your data ensures that the scale bar 
        #accurately represents the distances within the plot according to the raster's spatial characteristics.
        scalebar = ScaleBar(dx= 20, units="m", location="lower right") # dx = 20 due to the current spatial resolution (I.e., 20m)
        plt.gca().add_artist(scalebar)
        
        # Add a north arrow
        x, y, arrow_length = 0.97, 0.99, 0.1
        plt.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                     arrowprops=dict(facecolor='white', width=5, headwidth=15),
                     ha='center', va='center', fontsize=20, xycoords='axes fraction', color = 'white')
        
        plt.axis('off')
        
        # Adjust layout to ensure all elements are inside the figure boundaries
        plt.tight_layout()  
        
        #switch to 'agg' backend which is known to be more compatible for saving images.
        plt.switch_backend('agg')
        
        # Save the plot
        plt.savefig(os.path.join(current_directory, "Offline_Map_Outputs", "Outputs_With_Added_Map_Elements", "Classifier_Training_SAR_Mosaic.png"), bbox_inches='tight')  # Save the plot as PNG file
        
        #Close the plot
        plt.close()

        #Close the raster
        sar.close()
        
        
    #Open the VV SAR mosaic captured at the start of the monitoring period using rasterio
    with rasterio.open(os.path.join(current_directory, "Offline_Map_Outputs", "Start_and_End_Monitoring_Mosaics", "START_of_Monitoring_Period_SAR_Mosaic.tif")) as sar:

        # Read the VV band data
        vv_band = sar.read(1)  # Change '1' to the appropriate band index if VV is in a different band
        
        # Apply visualization parameters
        # You can adjust additional visualization parameters here if needed
        vis_params = {
            'vmin': -15,
            'vmax': 0,
        }
        
        # Plotting the data
        plt.figure(figsize=(8, 8))
        
        plt.imshow(vv_band, cmap='gray', **vis_params)
        
        plt.title('SAR Mosaic Captured at the Start of the Monitoring Period',fontweight='bold', style='italic')
        
        #Intialise the colorbar variable (cbar)
        cbar = plt.colorbar()
    
        # Set the tick labels for the color bar
        cbar.set_ticks([vis_params['vmin'], vis_params['vmax']])
        cbar.set_ticklabels(['High\nNegative\nBackscatter\nValue', 'High\nPositive\nBackscatter\nValue'],multialignment='center')
        
        # Add a spatially accurate scale bar to the plot
        #Adjusting dx based on the actual spatial resolution of your data ensures that the scale bar 
        #accurately represents the distances within the plot according to the raster's spatial characteristics.
        scalebar = ScaleBar(dx= 20, units="m", location="lower right") # dx = 20 due to the current spatial resolution (I.e., 20m)
        plt.gca().add_artist(scalebar)
        
        # Add a north arrow
        x, y, arrow_length = 0.97, 0.99, 0.1
        plt.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                     arrowprops=dict(facecolor='white', width=5, headwidth=15),
                     ha='center', va='center', fontsize=20, xycoords='axes fraction', color = 'white')
        
        #Turn the axis off
        plt.axis('off')
        
        # Adjust layout to ensure all elements are inside the figure boundaries
        plt.tight_layout()  
        
        #switch to 'agg' backend which is known to be more compatible for saving images.
        plt.switch_backend('agg')
        
        # Save the plot
        plt.savefig(os.path.join(current_directory, "Offline_Map_Outputs", "Outputs_With_Added_Map_Elements", "START_of_Monitoring_Period_SAR_Mosaic.png"), bbox_inches='tight')  # Save the plot as PNG file
        
        #Close the plot
        plt.close()
        
        #Close the raster
        sar.close()
        
        
    #Open the VV SAR mosaic captured at the end of the monitoring period using rasterio
    with rasterio.open(os.path.join(current_directory, "Offline_Map_Outputs", "Start_and_End_Monitoring_Mosaics", "END_of_Monitoring_Period_SAR_Mosaic.tif")) as sar:

        # Read the VV band data
        vv_band = sar.read(1)  # Change '1' to the appropriate band index if VV is in a different band
        
        # Apply visualization parameters
        # You can adjust additional visualization parameters here if needed
        vis_params = {
            'vmin': -15,
            'vmax': 0,
        }
        
        # Plotting the data
        plt.figure(figsize=(8, 8))
        
        plt.imshow(vv_band, cmap='gray', **vis_params)
        
        plt.title('SAR Mosaic Captured at the End of the Monitoring Period',fontweight='bold', style='italic')
        
        #Intialise the colorbar variable (cbar)
        cbar = plt.colorbar()
        
        # Set the tick labels for the color bar
        cbar.set_ticks([vis_params['vmin'], vis_params['vmax']])
        cbar.set_ticklabels(['High\nNegative\nBackscatter\nValue', 'High\nPositive\nBackscatter\nValue'],multialignment='center')
        
        # Add a spatially accurate scale bar to the plot
        #Adjusting dx based on the actual spatial resolution of your data ensures that the scale bar 
        #accurately represents the distances within the plot according to the raster's spatial characteristics.
        scalebar = ScaleBar(dx= 20, units="m", location="lower right") # dx = 20 due to the current spatial resolution (I.e., 20m)
        plt.gca().add_artist(scalebar)
        
        # Add a north arrow
        x, y, arrow_length = 0.97, 0.99, 0.1
        plt.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                     arrowprops=dict(facecolor='white', width=5, headwidth=15),
                     ha='center', va='center', fontsize=20, xycoords='axes fraction', color = 'white')
        
        #turn off the axis
        plt.axis('off')
    
        
        # Adjust layout to ensure all elements are inside the figure boundaries
        plt.tight_layout()  
        
        #switch to 'agg' backend which is known to be more compatible for saving images.
        plt.switch_backend('agg')
        
        # Save the plot
        plt.savefig(os.path.join(current_directory, "Offline_Map_Outputs", "Outputs_With_Added_Map_Elements", "END_of_Monitoring_Period_SAR_Mosaic.png"), bbox_inches='tight')  # Save the plot as PNG file
        
        #Close the plot
        plt.close()

        #Close the raster
        sar.close()
 
    '''
    ###
    All RF classified Outputs
    ###
    '''
    
    # Classified land cover image taken at the start of the monitoring period
    
    with rasterio.open(os.path.join(current_directory, "Offline_Map_Outputs", "START_Monitoring_Period_Classified.tif")) as classi:
        
        # Read the classified band data
        classified_band = classi.read(1)  
        
        #Set the colors for the map output
        #You can adjust additional visualization parameters here if needed
        colors = ['blue','purple','black','yellow','green']
        
        # Get the minimum and maximum values of the raster data
        data_min = 0
        data_max = 5
        
        # Create a masked array to exclude white values (i.e., the unclassified values that were not within the monitoring area)
        masked_array = np.ma.masked_where(classified_band == 0, classified_band)
        
        # Create a colormap with the new color scheme
        cmap_2 = ListedColormap(colors)
        
        # Plotting the raster data excluding white color
        plt.figure(figsize=(8, 8))
        plt.imshow(masked_array, cmap=cmap_2, vmin=1, vmax=data_max, interpolation='none')  # Set vmin and vmax based on the raster value range
        
        # Create a color bar excluding white (i.e., the unclassified values that were not within the monitoring area)
        cbar = plt.colorbar(ticks=range(data_min, data_max + 1),orientation='vertical')
        cbar.ax.set_yticklabels(['','Water', 'Bare Ground', 'Urban', 'Open Canopy\nMangroves', 'Closed Canopy\nMangroves'],multialignment='center')  # Set color bar labels as desired
        
        #Add a title to the plot
        plt.title('Start of Monitoring Period Classified Land Cover Map',fontweight='bold', style='italic')
        
        # Add a spatially accurate scale bar to the plot
        #Adjusting dx based on the actual spatial resolution of your data ensures that the scale bar 
        #accurately represents the distances within the plot according to the raster's spatial characteristics.
        scalebar = ScaleBar(dx= 20, units="m", location="lower right") # dx = 20 due to the current spatial resolution (I.e., 20m)
        plt.gca().add_artist(scalebar)
        
        # Add a north arrow
        x, y, arrow_length = 0.97, 0.99, 0.1
        plt.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                     arrowprops=dict(facecolor='black', width=5, headwidth=15),
                     ha='center', va='center', fontsize=20, xycoords='axes fraction', color = 'black')
        
        #Turn off the axis
        plt.axis('off')
        
        # Adjust layout to ensure all elements are inside the figure boundaries
        plt.tight_layout()  
        
        #switch to 'agg' backend which is known to be more compatible for saving images.
        plt.switch_backend('agg')
        
        # Save the plot
        plt.savefig(os.path.join(current_directory, "Offline_Map_Outputs", "Outputs_With_Added_Map_Elements", "START_Monitoring_Period_Classified.png"), bbox_inches='tight')  # Save the plot as PNG file
        
        
        #Close the plot
        plt.close()
        
        #Close the raster
        classi.close()
    
    
    # Classified land cover image taken at the end of the monitoring period
    
    with rasterio.open(os.path.join(current_directory, "Offline_Map_Outputs", "END_Monitoring_Period_Classified.tif")) as classi:
        
        # Read the classified band data
        classified_band = classi.read(1)  
        
        #Set the colors to use for the map output
        #You can adjust additional visualization parameters here if needed
        colors = ['blue','purple','black','yellow','green']
        
        # Get the minimum and maximum values of the raster data
        data_min = 0
        data_max = 5
        
        # Create a masked array to exclude white values (i.e., the unclassified values that were not within the monitoring area)
        masked_array = np.ma.masked_where(classified_band == 0, classified_band)
        
        # Create a colormap with the new color scheme
        cmap_2 = ListedColormap(colors)
        
        # Plotting the raster data excluding white color
        plt.figure(figsize=(8, 8))
        plt.imshow(masked_array, cmap=cmap_2, vmin=1, vmax=data_max, interpolation='none')  # Set vmin and vmax based on the raster value range
        
        # Create a color bar excluding white (i.e., the unclassified values that were not within the monitoring area)
        cbar = plt.colorbar(ticks=range(data_min, data_max + 1),orientation='vertical')
        cbar.ax.set_yticklabels(['','Water', 'Bare Ground', 'Urban', 'Open Canopy\nMangroves', 'Closed Canopy\nMangroves'],multialignment='center')  # Set color bar labels as desired
        
        #Add a title to the plot
        plt.title('End of Monitoring Period Classified Land Cover Map',fontweight='bold', style='italic')
        
        # Add a spatially accurate scale bar to the plot
        #Adjusting dx based on the actual spatial resolution of your data ensures that the scale bar 
        #accurately represents the distances within the plot according to the raster's spatial characteristics.
        scalebar = ScaleBar(dx= 20, units="m", location="lower right") # dx = 20 due to the current spatial resolution (I.e., 20m)
        plt.gca().add_artist(scalebar)
        
        # Add a north arrow
        x, y, arrow_length = 0.97, 0.99, 0.1
        plt.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                     arrowprops=dict(facecolor='black', width=5, headwidth=15),
                     ha='center', va='center', fontsize=20, xycoords='axes fraction', color = 'black')
        
        #Turn the axis off
        plt.axis('off')
        
        # Adjust layout to ensure all elements are inside the figure boundaries
        plt.tight_layout()  
        
        #switch to 'agg' backend which is known to be more compatible for saving images.
        plt.switch_backend('agg')
        
        # Save the plot
        plt.savefig(os.path.join(current_directory, "Offline_Map_Outputs", "Outputs_With_Added_Map_Elements", "END_Monitoring_Period_Classified.png"), bbox_inches='tight')  # Save the plot as PNG file
        
        #Close the plot
        plt.close()
        
        #Close the raster
        classi.close()
    
    
    '''
    ###
    Consistency Map Output
    ###
    '''
    with rasterio.open(os.path.join(current_directory, "Offline_Map_Outputs", "Consistency_Map.tif")) as con:
        

        # Read the consistency band data
        con_band = con.read(1)  
        
        '''
        #Convert hexcode colour to RGB tuple format suitable for the geemap library 
        '''
            
        #You can adjust additional visualization parameters here if needed
        #Set the rgb_colors to be converted to the hex format
        rgb_colors = [(247,252,245),(229,245,224),(199,233,192),(161,217,155),(116,196,118),(65,171,93),(35,139,69),(0,109,44),(0,68,27), (0,0,0)]
        
        
        #Convert RGB tuples to hexadecimal color codes as strings
        hex_colors = ['#%02x%02x%02x' % rgb for rgb in rgb_colors]
        
        # Get the minimum and maximum values of the raster data
        data_min = con_band.min()
        data_max = con_band.max()

      
        # Define the number of colors you want
        num_colors = 6

        # Sample the original colors evenly to create a new color scheme 
        new_hex_colors = [hex_colors[i * (len(hex_colors) - 1) // (num_colors - 1)] for i in range(num_colors)]

        # Create a colormap with the new color scheme
        cmap = ListedColormap(new_hex_colors)

        # Plotting the raster data using the custom colormap
        plt.figure(figsize=(8, 8))
        plt.imshow(con_band, cmap=cmap, vmin=data_min, vmax=data_max)  # Set vmin and vmax based on the raster value range
        plt.colorbar(label='Consistency Score')
        
        plt.title('Consistency Map of the Monitoring Area', fontweight='bold', style='italic')
        
        # Add a spatially accurate scale bar to the plot
        #Adjusting dx based on the actual spatial resolution of your data ensures that the scale bar 
        #accurately represents the distances within the plot according to the raster's spatial characteristics.
        scalebar = ScaleBar(dx= 20, units="m", location="lower right") # dx = 20 due to the current spatial resolution (I.e., 20m)
        plt.gca().add_artist(scalebar)
        
        # Add a north arrow
        x, y, arrow_length = 0.97, 0.99, 0.1
        plt.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                     arrowprops=dict(facecolor='black', width=5, headwidth=15),
                     ha='center', va='center', fontsize=20, xycoords='axes fraction', color = 'black')
        
        #Turn off the axis
        plt.axis('off')
                
        # Adjust layout to ensure all elements are inside the figure boundaries
        plt.tight_layout()  
        
        #Switch to 'agg' backend which is known to be more compatible for saving images.
        plt.switch_backend('agg')
        
        #Save the plot
        plt.savefig(os.path.join(current_directory, "Offline_Map_Outputs", "Outputs_With_Added_Map_Elements", "Consistency_Map.png"), bbox_inches='tight')  # Save the plot as PNG file
        
        #Close the plot
        plt.close()
        
        #Close the raster
        con.close()
    
    
    print("Script Update: Map elements have now been added and saved")
    
    #End the function
    return


'''
- Intialise the "SC_Token_Distribution" function
- This function states the change percentag of the monitoring area
- It then checks if all users are happy that this percentage has/hasn't met the required threshold
- Tokens are then distributed/not distributed based on this confirmation. The updated token counts are saved as a geodataframe and excel file
'''

def SC_Token_Distribution(GDF,Token_Holder_Index, Agreed_Change_Percentage, Token_Value, change_percentage, user_choice_TH, chosen_column_TA, mon_before_SD, mon_after_ED):
    #aoi, mon_before_SD, mon_before_ED, mon_after_SD,  mon_after_ED, GDF, Token_Holder_Index, Agreed_Change_Percentage, Token_Value
    
    '''
    Extracting the correct SC parameters to then print out for final confirmation
    '''
    FINAL_available_columns = GDF.columns
    
    chosen_column_index_TH = user_choice_TH - 1  # Adjusted for 0-based indexing
    chosen_column_TH = FINAL_available_columns[chosen_column_index_TH]
    
    #Intialise variable that represents the token holder who will receive tokens
    TH = GDF[chosen_column_TH].iloc[Token_Holder_Index] 
    
    #Conditional block to check that all users agree with the final results and that these results are either contract or non-contract compliant
    while True:
        user_input = input(f"\n* The final change percentage between the dates of {'mon_before_SD'} and {'mon_after_ED'} is '{change_percentage}'%.\
                           \n\n* The Smart Contract agreed by all parties requires that the monitored area of Token Holder '{TH}' had a total change percentage of '{Agreed_Change_Percentage}'%.\
                           \n\n* Do all parties agree that this requirement was met?\
                           \n\n* Type 'Y' for Yes or 'N' for No: ")
                         
        #If all users agree, distribute credits, update the intial geodataframe and export it as a geodataframe and excel file                   
        if user_input.upper() == "Y":
            print("\nYou entered 'Y' for Yes.")
            print("\nAll parties have agreed that the requirements have been met. Therefore,the agreed token amount will be distributed now...")
            
            #Convert the string previously taken from the user into an int type.
            Token_Value = int(Token_Value)

            #Add the agreed token amount to the moniotred users total     
            GDF[chosen_column_TA].iloc[Token_Holder_Index] = GDF[chosen_column_TA].iloc[Token_Holder_Index] + Token_Value 
            
            # Get the current directory of the Python script
            current_directory = os.path.dirname(os.path.abspath(__file__))

            # Create the new folder
            new_folder_path = os.path.join(current_directory, "Outputs")
            
            #Confirmation message
            print(f"\nThe agreed token amount of '{Token_Value}' has been successfully distributed")
            print(f"\n\nSaving the updated values in a .geojson (Updated_SC_GeoDataFrame.geojson) and .xlsx (Updated_SC_GeoDataFrame.xlsx) format to keep a record of the Transaction in the following directory: '{new_folder_path}'")
    
            '''
            Save the updated GeoDataFrame
            '''          
    
            GDF.to_file(os.path.join(new_folder_path, "Updated_SC_GeoDataFrame.geojson"), driver='GeoJSON')

            '''
            Save the updated GeoDataFrame as an xcel (.xlsx) file 
            '''
            
            GDF.to_excel(os.path.join(new_folder_path, "Updated_SC_GeoDataFrame.xlsx"), index=True)
            
            print("\n\nSave complete!")
        
            break
        
        #If the requirements have not been met then the GeoDataFrame remains unchanged so a new copy does not need to be created.
        elif user_input.upper() == "N":
            print("You entered 'N' for No.")
            print("\nNot all parties have agreed that the requirements have been met. Therefore, no tokens will be distributed")
            break
        
        else:
            print("Invalid input. Please type 'Y' for Yes or 'N' for No.")
        


'''
####!!!!!!!! FUNCTION CREATION ENDS HEREEEE !!!!!!!!!!####
'''


'''
Call GEE_Authenticate function
'''

GEE_Authenticate()


'''
Create the "Outputs" folder to put all map and data outputs
'''

# Get the current directory of the Python script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the name of the new folder
new_folder_name = "Outputs"

# Create the new folder
new_folder_path = os.path.join(current_directory, new_folder_name)

# Check if the folder already exists, and create it if not
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)
    print(f"\nFolder '{new_folder_name}' created at {new_folder_path}\n")
else:
    print(f"\nFolder '{new_folder_name}' already exists at {new_folder_path}. No new folder will be created.\n")


'''
Call the "create_smart_contract" function
'''
aoi, mon_before_SD, mon_before_ED, mon_after_SD,  mon_after_ED, GDF, Token_Holder_Index, Agreed_Change_Percentage, Token_Value, user_choice_TH, chosen_column_TA, aoi_geo = create_smart_contract()


print("\nSCRIPT UPDATE: Creating and validating RF MODEL")


'''
Call the "RF_Model_Creation_and_Validation" function:
'''

classifiedRf, trainedRf, trainingPts, consistency_map, VV_Mosaic_RF_Train, Optical_Mosaic_RF_Train = RF_Model_Creation_and_Validation(aoi)

print("\nSCRIPT UPDATE: RF MODEL has been created and validated")


'''
Call the "change_detection" function:
'''

print("\nSCRIPT UPDATE: Applying RF model to requested monitoring period")

change_percentage, START_result_image, END_result_image, START_MONITOR_SARVV_mosaic, START_MONITOR_Optical_mosaic, END_MONITOR_SARVV_mosaic, END_MONITOR_Optical_mosaic = change_detection(aoi, mon_before_SD, mon_before_ED, mon_after_SD, mon_after_ED, classifiedRf, trainedRf, trainingPts)


'''
Prompt the user to see if they want to download all map outputs and add additional map elements
'''

while True:
    
    user_input = input("\n\n* SCRIPT ALERT: The change percentage has now been calculated.\n* Would you like to download all map outputs and add additional map elements for spatial context?\
                       \n\n* Please note the following:\n1 - The process may take a long time\n2 - Due to current cloud processing limitations, the outputs can only be exported and displayed at a 20m resolution\
                       \n\n* Type 'Y' for Yes or 'N' for No: ")
                       
    if user_input.upper() == "Y":
        print("\nYou entered 'Y' for Yes.")
        
        
        #If the user says yes then call the "offline_output_creation" function
        
            
        offline_output_creation(START_MONITOR_SARVV_mosaic, START_MONITOR_Optical_mosaic, END_MONITOR_SARVV_mosaic, END_MONITOR_Optical_mosaic,VV_Mosaic_RF_Train, Optical_Mosaic_RF_Train,consistency_map, START_result_image, END_result_image, aoi_geo)
        
        break
        
    #If the user says no then the "offline_output_creation" script will be ignored.
    
    elif user_input.upper() == "N":
        print("You entered 'N' for No.")
        
        break
    
    #If a valid input is not provided, prompt the user to try again
    else:
        print("Invalid input. Please type 'Y' for Yes or 'N' for No.")


'''
# Mapping the RF CLASSIFIER results for mangrove land cover. These web maps could be inspected by all SC participants before the token distribution function is called.
'''

# Add the output of the training classification to the map
Map = geemap.Map()

#### Set the visualisation parameters for both START and END Classified mosaic fusions ####


# Define visualization parameters for classification display
#1 = water (blue), 2 = bare_ground (purple), 3 = Urban (black), 4 = Open Canopy (Yellow), 5 = Closed Canopy (green)
classVis = {
    'min': 1,
    'max': 5,
    'palette': ['blue', 'purple', 'black', 'yellow', 'green']
}

'''
# Add the START montoring period classified mosaic fusion  
'''

Map.addLayer(
    START_result_image.clipToCollection(aoi), classVis, 'FUSION Classes (RF)')

#Center web map to monitoring area
Map.centerObject(aoi, 12)


#Save the map output
Map.save(os.path.join(current_directory,"Outputs","START_Monitor_RF_Classifier_Output.html"))

#Attempt to open the classified fusion mosaics so the user can have an initial view of the results. Only the mangrove land cover will be shown
webbrowser.open_new_tab(os.path.join(current_directory, "Outputs", "START_Monitor_RF_Classifier_Output.html"))

#Confirmation message to let the user know that the script has been completed. 
print(f"\n\nSCRIPT UPDATE: The RF classified mosaic taken at the START of the monitoring period (START_Monitor_RF_Classifier_Output.html) has been saved in the following directory: '{current_directory}\Outputs'")

'''
# Add the END montoring period classified mosaic fusion
'''

# Add the output of the training classification to the map

Map.addLayer(
    END_result_image.clipToCollection(aoi), classVis, 'FUSION Classes (RF)')

#Center web map to monitoring area
Map.centerObject(aoi, 12)

#Save the map output
Map.save(os.path.join(current_directory,"Outputs","END_Monitor_RF_Classifier_Output.html"))

#Attempt to open the classified fusion mosaics so the user can have an initial view of the results. Only the mangrove land cover will be shown
webbrowser.open_new_tab(os.path.join(current_directory, "Outputs", "END_Monitor_RF_Classifier_Output.html"))

#Confirmation message to let the user know that the script has been completed. 
print(f"\nSCRIPT UPDATE: The RF classified mosaic taken at the END of the monitoring period (END_Monitor_RF_Classifier_Output.html) has been saved in the following directory: '{current_directory}\Outputs'")


'''
Enter the final phase - Call the "SC_Token_Distribution" function
'''

print("\n\nSCRIPT UPDATE: Moving onto the final phase. Smart Contract Validation...")
SC_Token_Distribution(GDF,Token_Holder_Index, Agreed_Change_Percentage, Token_Value, change_percentage, user_choice_TH, chosen_column_TA, mon_before_SD, mon_after_ED)


#Confirmation message to let the user know that the script has been completed. 
print("\nSmart Contract Script has now been Completed!")









