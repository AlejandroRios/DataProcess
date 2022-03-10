# =============================================================================
# IMPORTS
# =============================================================================
import haversine
from haversine import haversine, Unit
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




distance = float(haversine(coord1,coord2,unit='nmi'))


