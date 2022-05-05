import os
from preprocessing.automatic_preprocessing_old import data_preprocessing
directory = os.getcwd()

departures = ['FRA','LHR','CDG','AMS','MAD','BCN','FCO','DUB','VIE','ZRH']

arrivals =   ['FRA','LHR','CDG','AMS','MAD','BCN','FCO','DUB','VIE','ZRH']



def iterate_files(departures,arrivals):

    # for root, dirs, files in os.walk("/Users/aarc8/Documents/github/DataProcess/Database/ADSB"):
    #     for file in files:
    #         if file.endswith("allFlights_"+departures+arrivals+".csv"):
    #             print(os.path.join(root, file))

    #             data_preprocessing(departures,arrivals)
    data_preprocessing(departures,arrivals)

    return


for i in range(len(departures)):
    for j in range(len(arrivals)):
        # if (i != j):
        if (i != j) and (i > 8) and (j > 1):
        # if (i != j) and (i > 6) and (i < 8) and (j>4) :
        # if (i != j) and (i > 6) and (i < 8):
            # iterate_files(departures[i],arrivals[j])

            data_preprocessing(departures[i],arrivals[j])