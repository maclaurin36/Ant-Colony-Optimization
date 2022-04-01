import math
from typing import List
from TSPClasses import *
from Constant import Constant
class Ant:

    def __init__(self, startCityIndex) -> None:
        self.antPath = [startCityIndex]
        self.citiesVisited = set()
        self.citiesVisited.add(startCityIndex)
        self.currentCity = startCityIndex
        self.startCity = startCityIndex
        # Personality surprisingly makes the ants better
        self.ALPHA_PHEREMONE = random.random() * Constant.ALPHA
        self.BETA_LENGTH = random.random() * Constant.BETA
        self.distanceTraveled = 0

    def visit_city(self, cityIndex: int, pathCost: float):
        self.antPath.append(cityIndex)
        self.citiesVisited.add(cityIndex)
        self.currentCity = cityIndex
        self.distanceTraveled += pathCost

    def pick_next_city(self, cities: List[City], edgeMatrix: List[List[int]], pheremoneMatrix: List[List[int]]) -> int:

        visitingCity: City

        # Calculate the numerators - pheremone ^ Alpha * 1/L ^ Beta
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
            return self.run_roulette(cityNumeratorList, cityDenominator)
		
    def run_roulette(self, cityNumeratorList, cityDenominator):
        
        # Skip plain probability to cumulative probabilities and roulette
        prevProbability = 0
        randomFloat = random.random()
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