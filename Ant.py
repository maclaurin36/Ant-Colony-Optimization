import math
from typing import List
from TSPClasses import *
from Constant import Constant
class Ant:

    # Initialize an ant
    # Time compleity: O(1)
    # Space complexity: O(n) - ant path will grow to n city size
    def __init__(self, startCityIndex) -> None:
        self.antPath = [startCityIndex]
        self.citiesVisited = set()
        self.citiesVisited.add(startCityIndex)
        self.currentCity = startCityIndex
        self.startCity = startCityIndex
        # Personality (Wasn't beneficial)
        # self.ALPHA_PHEREMONE = random.random() * Constant.ALPHA
        # self.BETA_LENGTH = random.random() * Constant.BETA
        self.ALPHA_PHEREMONE = Constant.ALPHA
        self.BETA_LENGTH = Constant.BETA
        self.distanceTraveled = 0

    # Visit a city by adding it to the ants path
    # Time Complexity: O(1)
    # Space Complexity: O(1)
    def visit_city(self, cityIndex: int, pathCost: float):
        self.antPath.append(cityIndex)
        self.citiesVisited.add(cityIndex)
        self.currentCity = cityIndex
        self.distanceTraveled += pathCost

    # Go through each of the possible cities and check for the max viable path balancing exploration and exploitation
    # Time Complexity: O(n) Go through each edge twice to check viability O(1) and probabilistically pick one
    # Space Complexity: O(n^2) Uses the edge and pheremone matrix to calculate probabilities
    def pick_next_city(self, cities: List[City], edgeMatrix: List[List[int]], pheremoneMatrix: List[List[int]]) -> int:

        visitingCity: City

        # Calculate the numerators - pheremone ^ Alpha * 1/L ^ Beta
        # Time Complexity: O(n) It is important to note that checking if the ant has visited a city is O(1) because we use a hashset
            # to store cities visited which has O(1) lookup time
        # Time Complexity: O(n^2) Uses edge and pheremone matrix (also uses O(n) space to store edge scores)
        cityNumeratorList = []
        cityDenominator = 0
        for visitingCity in cities:
            distToCity = edgeMatrix[self.currentCity][visitingCity._index]
            if not visitingCity._index in self.citiesVisited and distToCity != math.inf:
                pheremoneWeight = pow(pheremoneMatrix[self.currentCity][visitingCity._index], self.ALPHA_PHEREMONE)
                edgeWeight = pow(Constant.NUMERATOR_EDGE_CONST/(distToCity+0.0001), self.BETA_LENGTH)
                numerator = pheremoneWeight * edgeWeight
                cityNumeratorList.append((visitingCity._index, numerator))
                cityDenominator += numerator
        
        # No valid path out, so either ant can't finish or has visited all cities
        if cityDenominator == 0:
            return -1
        else:
            # Time Complexity: O(n)
            # Space Complexity: O(n)
            return self.run_roulette(cityNumeratorList, cityDenominator)
		
    # Probabilistically pick which city to visit next
    # Time complexity: O(n) Go throuch each edge score in the list while checking probability O(1)
    # Space complexity: O(n) Uses a list of edge scores in the numerator list
    def run_roulette(self, cityNumeratorList, cityDenominator):
        
        # Skip plain probability to cumulative probabilities and roulette
        prevProbability = 0
        randomFloat = random.random()

        # Time Complexity: O(n) also constant factors are minimized by not calculating all cumulative probabilities
        for i in range(0, len(cityNumeratorList)):
            cityIndex = cityNumeratorList[i][0]
            cityNumerator = cityNumeratorList[i][1]
            curProbability = (cityNumerator / cityDenominator) + prevProbability
            if randomFloat < curProbability:
                return cityIndex
            prevProbability = curProbability

	# def pick_next_city()
	# 	for all cities
	#		if cant reach or been to -> break
	# 		get score (list)
	#		sum += score

	# 	for all cities
	#		if cant reach or been to -> break
	#		p[j] = score[j] / sum

	# 	city = roulette(p)
	# 	return city

    # Distances to cities can be zero?