#!/usr/bin/python3

import copy
from typing import List, Set
from xmlrpc.client import Boolean
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



class TSPSolverBandB:
	def __init__( self ):
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
	# Space complexity: O(n^2) - the dominating space is the matrix of edge existence/costs
	def greedy( self,time_allowance=60.0 ):
		# Get the related information out of the class (make conceptually easier to understand)
		cities: List[City] = self._scenario.getCities()
		bssf = None

		# Spec assumes path exists for discussion of complexity, so this loop is not included in time complexity analysis
		startTime = time.time()
		solutionFound = False
		while not solutionFound and startTime - time.time() < time_allowance:

			# Pick a random city - constant time
			randCityIndex = random.randrange(0, len(cities) - 1)
			
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
	# Space complexity: O(n^2) - The dominating space is the matrix of edge existence/costs
	def getGreedyNeighbor(self, city: City, cityVisitedSet: Set) -> City:
		cities: List[City] = self._scenario.getCities()
		edges: List[List[Boolean]] = self._scenario._edge_exists

		# Initialize the lowest cost city to travel to
		lowestCost = math.inf
		lowestCity = None

		# Go through all city connections
		# Time complexity: O(n)
		for i in range(0, len(edges[city._index])):

			# Update the lowest cost city if the current city costs less
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
		# Get references to the scenario data
		# Time complexity: O(1)
		# Space complexity: O(n) for cities and O(n^2) for edges because we are using a matrix representation
		cities: List[City] = self._scenario.getCities()
		edges: List[List[Boolean]] = self._scenario._edge_exists

		# Pick arbitrary node to start from
		# Time complexity: O(1)
		randCityIndex = random.randrange(0, len(cities) - 1)

		# Create the reduced cost matrix
		# Time Complexity: O(n^2) - Constant to calculate the cost, n^2 items in the matrix
		# Space Complexity: O(n^2) - Matrix to represent edge lengths
		matrixList: List[List[int]] = []
		for row in range(0, len(edges)):
			matrixList.append([])
			for col in range(0, len(edges)):
				if edges[row][col]:
					matrixList[row].append(cities[row].costTo(cities[col]))
				else:
					matrixList[row].append(math.inf)

		# Initialize my state data structure
		# Time complexity: O(n) - creates two arrays that show that none of the columns or rows have been marked
		# Space complexity: O(n^2) - stores the previously created matrix
		matrix = State(matrixList, randCityIndex)

		# Branch and Bound Algorithm start
		startTime = time.time()

		# Reduce the initial matrix
		# Time complexity: O(n^2) - go through each row and column twice to reduce
		matrix.reduceMatrix()

		# Get the initial best solution using greedy, run multiple times for solution consistency and because it costs so little
		# Time complexity: O(n^2)
		# Space complexity: O(n^2) - you need the graph edges
		greedyBssf: TSPSolution = None
		for i in range(0, 5):
			greedyResults = self.greedy()
			if greedyBssf == None or greedyResults['cost'] < greedyBssf.cost:
				greedyBssf = greedyResults['soln']

		# Initialize the priority queue and bssf
		# Time complexity: O(1)
		# Space complexity: ???
		pQueue: List[State] = []
		bssfCost = greedyBssf.cost
		bssfList = []

		# Initialize the result information
		# Time complexity: O(1)
		solutionCount = 0
		pruned = 0
		childrenCreated = 1
		maxQueue = 0

		# Expand the initial state
		# Time complexity: O(n^3) - see expandState for more detail
		# Space complexity: O(n^3)
		bssfCost, bssfList, solutionCount, pruned, childrenCreated = matrix.expandState(pQueue, bssfCost, bssfList, solutionCount, pruned, childrenCreated)

		# Process each of the states in the priority queue, expanding them if they are viable
		# Time complexity: O(n^2 * number of states) which worse case could be O(n^2*(n+1)!) however the actual number of states created
			# can vary depending on the distribution of the cities. My heap prioritization algorithm works on an "average distance per city" heuristic
			# this means that if you have explored more cities and have a lower bound you are more likely to select that path.  This makes sense because
			# the actual solution will have the optimal minimum average distance between all cities.  However, it does make it a more "greedy" heuristic 
			# so my algorithm will not go as fast when there are clusters of cities, but will perform better when the cities are more randomly distributed
			# or all found within the same cluster.
		# Space complexity: O(n^2 * max number of states on the priority queue) if we were to go about state prioritization from a breadth-first approeach,
			# this could be the number of states in the final level of the tree or about (n-1)!/2 states.  However, in my algorithm, using the average distance per city
			# heuristic, I dig deeper than this breadth-first search approach to prune earlier and avoid a large queue size.
		while len(pQueue) > 0 and time.time() - startTime <= time_allowance:

			# Update the maximum number of elements on the queue after expansion
			# Time complexity: O(1)
			if len(pQueue) > maxQueue:
				maxQueue = len(pQueue)

			# Get the state with the lowest key value
			# Time complexity: O(log(# states on queue))
			curMatrix = heapq.heappop(pQueue)

			# Either prune the state popped off the queue or expand its children
			if curMatrix.lowerBound < bssfCost:
				# Time complexity: O(n^3)
				bssfCost, bssfList, solutionCount, pruned, childrenCreated = curMatrix.expandState(pQueue, bssfCost, bssfList, solutionCount, pruned, childrenCreated)
			else:
				pruned += 1

		endTime = time.time()

		# Prune the remaining items on the queue that are worse than bssf for reporting
		# Time complexity: O(max number of states on the priority queue) this will pretty much always be better than the worse case number of states on the queue
			# because it is unlikely that the heuristic puts everything on the queue and that we stop at that moment.
		if endTime - startTime >= 60:
			for matrix in pQueue:
				if matrix.lowerBound >= bssfCost:
					pruned += 1

		# Create what the GUI needs to show the new path
		# Time complexity: O(n)
		# Space complexity: O(n)
		bssfRoute = []
		if bssfList != []:
			for cityIndex in bssfList:
				bssfRoute.append(cities[cityIndex])

		finalBssf = TSPSolution(greedyBssf.route) if len(bssfRoute) == 0 else TSPSolution(bssfRoute)
		results = {}
		results['cost'] = finalBssf.cost 							# Cost of best solution
		results['time'] = endTime - startTime						# Time spent to find the best solution
		results['count'] = solutionCount 				# Total number of solutions found
		results['soln'] = finalBssf 										# The best solution found
		results['max'] = maxQueue 										# Null
		results['total'] = childrenCreated 									# Null
		results['pruned'] = pruned 									# Null
		return results



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		pass

class State:
	# Create a state given its reduced cost matrix and the city most recently visited
	# Time complexity: O(n) - create the lists that show rows and columns have been marked/visited
	# Space complexity: O(n^2) - stores the reduced cost matrix of the state
	def __init__(self, matrix: List[List[int]], cityIndex):
		self.matrix = matrix
		self.lowerBound = 0
		self.markedRows = []
		self.markedColumns = []
		for i in range(0, len(matrix)):
			self.markedRows.append(False)
			self.markedColumns.append(False)
		self.cityIndex = cityIndex
		self.citiesVisited = [cityIndex]
		self.markedColumns[cityIndex] = True

	# Prioritize states by their average distance traveled per node. This way if a state is lower down and has a low bound,
	# its average will be low as well, and shallower, poorer states will likely get higher averages quickly. This is used
	# when prioritizing states in the priority queue
	# Time complexity: O(1)
	def __lt__(self, other) -> bool:
		if self.lowerBound / len(self.citiesVisited) < other.lowerBound / len(other.citiesVisited):
		#if self.lowerBound - len(self.citiesVisited) * 500 < other.lowerBound - len(other.citiesVisited) * 500:
			return True
		return False

	# Time complexity: O(n^2) - Must go through each row and column two times to find the min and reduce the row/col
	# Space complexity: O(n^2) - This algorithm is working on the edge matrix which is n^2
	def reduceMatrix(self):
		for row in range(0, len(self.matrix)):
			if not self.markedRows[row]:
				minValue = math.inf
				# Get the minimum for the row
				for col in range(0, len(self.matrix)):
					if self.matrix[row][col] < minValue:
						minValue = self.matrix[row][col]
				# Reduce the row
				for col in range(0, len(self.matrix)):
					self.matrix[row][col] -= minValue
				# Increment lower-bound
				self.lowerBound += minValue
			
		for col in range(0, len(self.matrix)):
			if not self.markedColumns[col]:
				minValue = math.inf
				# Get the minimum for the column
				for row in range(0, len(self.matrix)):
					if self.matrix[row][col] < minValue:
						minValue = self.matrix[row][col]
				# Reduce the column
				for row in range(0, len(self.matrix)):
					self.matrix[row][col] -= minValue
				# Increment lower-bound
				self.lowerBound += minValue
	
	# Takes a state, creates its children, if applicable puts them on the queue
	# Time complexity: O(n^3) - The first state, when expanded will have n-1 expansions for a fully connected graph. In each expansion, we are copying the matrix
		# O(n^2)
	# Space complexity: n^2 * max size of heap priority queue
	def expandState(self, pQueue: List, bssfCost, bssfList, solutionCount, pruned, childrenCreated):
		# For each city that hasn't already been visited
		# Create its reduced cost matrix by marking its row and column, setting each entry in the row and column to infinity, then reducing
		# If the lowerbound is less than the BSSF add it to the queue
		for i in range(0, len(self.markedColumns)):
			if not self.markedColumns[i]:
				# Create a copy of the parent state
				# Time complexity: O(n^2) - copying the matrix
				# Space complexity: O(n^2) - storing the matrix
				childState = copy.deepcopy(self)
				childrenCreated += 1

				# Mark the column and row because we can't enter or leave from this city again
				# Time complexity: O(n) - Go through one row and one column each of length n
				# Space complexity: O(n^2) - Operating on the cost matrix of size n^2
				childState.markEdge(self.cityIndex, i)

				# Reduce the child matrix
				# Time complexity: O(n^2)
				# Space complexity: O(n^2)
				childState.reduceMatrix()

				# Check if its a solution
				if len(childState.citiesVisited) == len(childState.markedColumns):
					# Update the solution if it is good and better than bssf
					# Time complexity: O(1)
					# Space complexity: O(n) - the path of bssf is also saved
					solutionCount += 1
					if childState.lowerBound + childState.matrix[i][self.citiesVisited[0]] < bssfCost:
						bssfCost = childState.lowerBound + childState.matrix[i][self.citiesVisited[0]]
						bssfList = childState.citiesVisited
				else:
					# If the state is not pruned, add it to the heap
					# Time complexity: log(v) - here v represents the number of states on the queue, this could overtake the overall time complexity of 
						# state expansion if the queue got very large, however based on my empirical analysis, n^2 is generally much larger than log(v).
						# The time complexity is log(v) because the heap I use is implemented by a binary heap.
					# Space complexity: Based off of the max number of states on the queue, so n^2 * max num states on queue.
					if childState.lowerBound < bssfCost:
						heapq.heappush(pQueue, childState)
					else:
						pruned += 1
		return bssfCost, bssfList, solutionCount, pruned, childrenCreated


	# Sets the edges to infinite into the column city and out of row city, increments lowerbound, and marks the col and row to not be processed
	# Time complexity: O(n) - must traverse the entire row and column in the matrix
	# Space complexity: O(n^2) - the state matrix on which it is operating is O(n^2)
	def markEdge(self, row, col):
		self.lowerBound += self.matrix[row][col]
		for i in range(0, len(self.matrix)):
			self.matrix[i][col] = math.inf
			self.matrix[row][i] = math.inf
		self.markedRows[row] = True
		self.markedColumns[col] = True
		self.citiesVisited.append(col)
		self.cityIndex = col