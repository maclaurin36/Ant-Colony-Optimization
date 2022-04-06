#!/usr/bin/python3


# THIS IS A TEST -AB

# This is a test Adam
# This is from a branch

# AB TIME: 45



from typing import List, Set
from xmlrpc.client import Boolean
from Ant import Ant
from Constant import Constant
from SampleBranchAndBound import TSPSolverBandB

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	# Written by Jesse total time - 1.5 hours
	# Time complexity: O(n^2) - nearest neighbor * num cities
	# Space complexity: O(n) - route of cities
	def greedy( self,time_allowance=60.0 ):
		# Get the related information out of the class (make conceptually easier to understand)
		cities: List[City] = self._scenario.getCities()
		bssf = None

		# Spec assumes path exists for discussion of complexity, so this loop is not included in time complexity analysis
		startTime = time.time()
		solutionFound = False
		while not solutionFound and startTime - time.time() < time_allowance:

			# Pick a random city - constant time
			randCityIndex = random.randint(0, len(cities) - 1)

			# Initialize route and start city
			# Time complexity: O(1)
			# Space complexity: O(n) Eventually the route and set of cities visited will include all cities 2n
			startCity = cities[randCityIndex]
			cityVisitedSet = set()
			route: List[City] = []
			cityVisitedSet.add(startCity)
			route.append(startCity)

			# Get the first nearest neighbor
			# Time complexity: O(n) - must traverse all edges in matrix (exist or not)
			# Space complexity: O(1)
			nearestNeighbor = self.getGreedyNeighbor(startCity, cityVisitedSet)

			# While there is a neighboring city that hasn't been visited and all cities haven't been visited
			# Time complexity: O(n) - runs once for each city (except first city)
			while nearestNeighbor != None and len(cityVisitedSet) != len(cities):

				# Mark the current city as visited
				cityVisitedSet.add(nearestNeighbor)
				route.append(nearestNeighbor)

				# Go to that city
				nearestNeighbor = self.getGreedyNeighbor(nearestNeighbor, cityVisitedSet)

			# If all cities haven't been visited, restart (failed because greedy path didn't work)
			if len(cityVisitedSet) == len(cities) and route[len(route) - 1].costTo(startCity) != math.inf:

				# Get the solution in the correct return format
				# Time complexity: O(n) - traverses city list to calculate cost
				# Space complexity: O(n) - holds a list with all cities
				bssf = TSPSolution(route)
				solutionFound = True

		endTime = time.time()

		results = {}
		results['cost'] = bssf.cost if solutionFound else math.inf 	# Cost of best solution
		results['time'] = endTime - startTime 						# Time spent to find the best solution
		results['count'] = 1 if solutionFound else 0 				# Total number of solutions found
		results['soln'] = bssf 										# The best solution found
		results['max'] = None 										# Null
		results['total'] = None 									# Null
		results['pruned'] = None 									# Null
		return results

	# Return the city's neighbor that is the closest (and not visited)
	# Return None if all reachable cities have been visited
	# Note that edges are represented in a matrix, not a list
	# Time complexity: O(n) - loop through all of the possible edges of a city
	# Space complexity: O(1) - assuming that the cities are stored elsewhere, no space added just accessing initial city and edge lists
	def getGreedyNeighbor(self, city: City, cityVisitedSet: Set) -> City:
		cities: List[City] = self._scenario.getCities()
		edges: List[List[Boolean]] = self._scenario._edge_exists

		# Loop through all possible city connections
		lowestCost = math.inf
		lowestCity = None

		# Go through all city connections
		# Time complexity: O(n)
		for i in range(0, len(edges[city._index])):

			# If there exists an edge between two cities
			# Time complexity: O(1)
			if edges[city._index][i]:
				cityCost = city.costTo(cities[i])
				if cityCost < lowestCost and not cities[i] in cityVisitedSet :
					lowestCost = cityCost
					lowestCity = cities[i]

		return lowestCity

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	def branchAndBound( self, time_allowance=60.0 ):
		bbAlgo = TSPSolverBandB()
		bbAlgo.setupWithScenario(self._scenario)
		return bbAlgo.branchAndBound(time_allowance)



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	# 5.5 hours - Jesse
	def fancy( self,time_allowance=60.0 ):

		# PSEUDO-CODE FOR ANT SYSTEM ALGORITHM

		cities: List[City] = self._scenario.getCities()
		edges: List[List[Boolean]] = self._scenario._edge_exists

		# Create the cost matrix for easier reference
		# Initialize array P of pheromone strength for each edge ij - can't initialize to 0 because numerator is multiplied
		# Time Complexity: O(n^2) - City cost O(1) * Num connections O(n^2)
		# Space Complexity: O(n^2) - Store the pheremones and city costs
		edgeMatrix: List[List[int]] = []
		pheremoneMatrix: List[List[float]] = []
		for row in range(0, len(edges)):
			edgeMatrix.append([])
			pheremoneMatrix.append([])
			for col in range(0, len(edges)):
				pheremoneMatrix[row].append(Constant.INIT_PHEREMONE)
				if edges[row][col]:
					edgeMatrix[row].append(cities[row].costTo(cities[col]))
				else:
					edgeMatrix[row].append(math.inf)

		# Initialize list of k ants
		# Time Complexity: O(k)
		# Space Complexity: O(k)
		antColony = []
		for i in range(0, Constant.NUM_ANTS):
			startCity = random.randint(0, len(cities) - 1)
			antColony.append(Ant(startCity))

		startTime = time.time()
		terminatingCondition = False
		bestAnt: Ant = None
		iterationsWithoutImprovement = 0

		# Begin running the ants
		# Time Complexity: The complexity of the main algorithm is difficult to quantify, because the ant's convergence on a path
			# widely varies, and would be better analyzed by looking at the empirical data.  In each loop iteration we run all ants
			# this takes O(k*n) time because we run k ants and each ant travels through k cities.  Thus, the total complexity would
			# be O(k*n*number of iterations before convergence)
		# Space Complexity: O(n^2) to store pheremones and edges, O(k) to store ants, O(k*n) to store ant paths
		while not terminatingCondition and time.time() - startTime < time_allowance:

			# Time Complexity: O(k*n) construct k paths through n cities
			# Space Complexity: O(k*n) store k paths through n cities
			bestRoundAnt: Ant = self.run_ants(cities, edgeMatrix, pheremoneMatrix, antColony)

			# Update the best ant and check the terminating condition
			# Time Complexity: O(1)
			# Space Complexity: O(1)
			if bestAnt == None or bestRoundAnt.distanceTraveled < bestAnt.distanceTraveled:
				bestAnt = bestRoundAnt
				iterationsWithoutImprovement = 0
			else:
				iterationsWithoutImprovement += 1
				if iterationsWithoutImprovement > Constant.ITER_WITHOUT_IMPROVEMENT:
					terminatingCondition = True
			
			# Update the pheremone matrix with the new ant paths
			# Time Complexity: O(n^2)
			# Space Complexity: O(n^2)
			self.update_pheremones(pheremoneMatrix, antColony)

			# Clear the ants
			# Time Complexity: O(k)
			# Space Complexity: O(k)
			antColony = []
			for i in range(0, Constant.NUM_ANTS):
				startCity = random.randint(0, len(cities) - 1)
				antColony.append(Ant(startCity))

		endTime = time.time()

		# Create what the GUI needs to show the new path
		# Time complexity: O(n)
		# Space complexity: O(n)
		bssfRoute = []
		if bestAnt != None:
			for cityIndex in bestAnt.antPath:
				bssfRoute.append(cities[cityIndex])

		result = {}
		result['soln'] = TSPSolution(bssfRoute)
		result['cost'] = bestAnt.distanceTraveled
		result['time'] = endTime - startTime
		result['count'] = 0
		result['max'] = 0
		result['total'] = 0
		result['pruned'] = 0
		return result
		# while (termination condition not met)
		# 	run_ants()
		#	update_pheromones()
		# Return BSSF

	# Run an iteration of ants (each ant finds circuit through cities)
	# Time Complexity: O()
	def run_ants(self, cities: List[City], edgeMatrix: List[List[int]], pheremoneMatrix: List[List[int]], antColony: List[Ant]):
		newBest: Ant = None
		ant: Ant
		for ant in antColony:

			# Pick the next city that the ant will travel to probabilistically
			# -1 is returned if there is not a viable city that the ant can travel to
			# Time Complexity: O(n) - goes through each edge twice in the worst case doing O(1) work each time
			# Space Complexity: O(n^2) because it utilizes the edge and pheremone matrix (O(n) space used to store edge scores for the ant)
			nextCity = ant.pick_next_city(cities, edgeMatrix, pheremoneMatrix)
			while (nextCity != -1):
				ant.visit_city(nextCity, edgeMatrix[ant.currentCity][nextCity])
				nextCity = ant.pick_next_city(cities, edgeMatrix, pheremoneMatrix)

			# If the ant did not visit all of the cities, don't update the pheremone matrix with it by setting its distance travelled to infinite
			# Also update bssf
			# Time Complexity: O(1)
			# Space Complexity: O(n*k) to store path for bssf
			if len(ant.antPath) < len(cities):
				ant.distanceTraveled = math.inf
			else:
				ant.distanceTraveled += edgeMatrix[ant.currentCity][ant.startCity]
				if newBest == None or ant.distanceTraveled < newBest.distanceTraveled:
					newBest = ant
		return newBest

	# Update the pheremone matrix after finishing an iteration of running the ants
	# Time Complexity: O(n^2) + O(n*k) = O(n^2), O(n^2) for evaporating pheremone from the matrix and O(n*k) for adding new pheremone for each ant
	# Space Complexity: O(n^2) to store the pheremone matrix
	def update_pheremones(self, pheremoneMatrix: List[List[float]], antColony: List[Ant]):

		# Update the pheremone matrix by evaporating some of the previous matrix's pheremone
		# Time Complexity: O(n^2)
		# Space Complexity: O(n^2)
		for row in range(0, len(pheremoneMatrix)):
			for col in range(0, len(pheremoneMatrix[0])):
				pheremoneMatrix[row][col] = (1 - Constant.EVAP_FACTOR) * pheremoneMatrix[row][col]

		# Add ant pheremones
		# Time Complexity: O(n*k)
		# Space Complexity: O(n^2)
		for ant in antColony:
			# Ignore the ants that didn't make it
			if ant.distanceTraveled != math.inf:
				for i, cityIndex in enumerate(ant.antPath):
					pheremoneMatrix[cityIndex][ant.antPath[(i + 1) % len(ant.antPath)]] += 1 / ant.distanceTraveled

		# def run_ants()
		#	for each ant x:
		#		start at random city C
		#		append city to ant_path
		#		while (len(ant_path) < n and nextcity is not C)
		#			nextcity = pick_next_city()
		#			append city to path
		#		if ant_path < BSSF
		#			BSSF = ant_path


		# def update_pheromone()
		#	initialize new array S for each edge ij
		#	for each ant
		#		for each edge ij used by ant:
		#			S[i][j] = S[i][j] + 1/length of ant_path

		#	for each edge ij
		#			P[i][j] = P[i][j] * (1 - EVAPORATION_RATE) + S[i][j]