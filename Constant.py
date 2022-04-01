class Constant:
    NUM_ANTS = 100
    EVAP_FACTOR = 0.5 # This must be between 0 and 1
    INIT_PHEREMONE = 1.0 # This must be a float
    NUMERATOR_EDGE_CONST = 100.0 # Must be a float
    ALPHA = 1 # For each ant a random number between 0 and 1 is multiplied by this to get the true alpha value
    BETA = 5 # For each ant a random number between 0 and 1 is multiplied by this to get the true beta value
    ITER_WITHOUT_IMPROVEMENT = 1000 # If BSSF isn't updated in this number of iterations, algorithm exits
