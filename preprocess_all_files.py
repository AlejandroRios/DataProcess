import os
from preprocessing.automatic_preprocessing import data_preprocessing
directory = os.getcwd()

departures = ['FRA','LHR','CDG','AMS','MAD','BCN','FCO','DUB','VIE','ZRH','ARN','DME','HEL','IST','KBP']
             #  0     1     2     3     4     5      6     7    8     9     10   11    12    13     14   
arrivals =   ['FRA','LHR','CDG','AMS','MAD','BCN','FCO','DUB','VIE','ZRH','ARN','DME','HEL','IST','KBP']



def iterate_files(departures,arrivals):

    # for root, dirs, files in os.walk("/Users/aarc8/Documents/github/DataProcess/Database/ADSB"):
    #     for file in files:
    #         if file.endswith("allFlights_"+departures+arrivals+".csv"):
    #             print(os.path.join(root, file))

    #             data_preprocessing(departures,arrivals)


    for root, dirs, files in os.walk("/Users/aarc8/Documents/DATABASE"):
        for file in files:
            if file.endswith("allFlights_"+departures+arrivals+".csv"):
                print(os.path.join(root, file))

                data_preprocessing(departures,arrivals)

    return


for i in range(len(departures)):
    for j in range(len(arrivals)):
        # if (i != j) and (i > 13):
        # if (i != j) and (i > 6) and (i < 8) and (j>11) and (j<13):
        # if (i != j) and (i > 8) and (i < 10) and (j>0) and (j<2):
        if (i != j) and (i > 8) and (i < 10) and (j>0) and (j<2):
            iterate_files(departures[i],arrivals[j])