# =============================================================================
# IMPORTS
# =============================================================================
import haversine
from haversine import haversine, Unit
from geopy.distance import great_circle
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
def haversine_distance(coordinates_departure,coordinates_arrival):
    """
    Description:
        - Perform haversine distance calculation in nautical miles
    Inputs:
        - coordinates_departure - latitude and longitude coordinates departure
        - coordinates_arrival - latitude and longitude coordinates arrival

    Outputs:
        - distance - distance between points [nm]
    """
    # Perform haversine distance calculation in nautical miles
    distance = float(haversine(coordinates_departure,coordinates_arrival,unit='nmi'))
    return distance

def actual_distance(lats,lons):
    
    dist_chunks_vec = []
    for j in range(len(lons)-1):
        coord1 = (lats[j],lons[j])
        coord2 = (lats[j+1],lons[j+1])

        distance = float(haversine(coord1,coord2,unit='nmi'))

        dist_chunks_vec.append(distance)
    
    return sum(dist_chunks_vec)


def horizontal_ineff(lats,lons):

    coord1 = (lats[0],lons[0])
    coord2 = (lats[-1],lons[-1])

    GCD = great_circle(coord1,coord2).nm

    actual = actual_distance(lats,lons)

    # print('GCD distance in nm:', GCD)
    # print('Actual distance in nm:', actual)

    return (((actual)-GCD)/GCD)*100
