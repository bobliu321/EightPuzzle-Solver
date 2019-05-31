# a1.py

from search import *
import random
import time
#Global Variable
nodesRemoved = 0

def best_first_graph_search1(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    global nodesRemoved#added
    nodesRemoved=0#added
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        nodesRemoved+=1#added
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None






def astar_search1(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    
    return best_first_graph_search1(problem, lambda n: n.path_cost + h(n))




def manhattan(node):
        state = node.state
        index_goal = {0:[2,2], 1:[0,0], 2:[0,1], 3:[0,2], 4:[1,0], 5:[1,1], 6:[1,2], 7:[2,0], 8:[2,1]}
        index_state = {}
        index = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]
        x, y = 0, 0
        
        for i in range(len(state)):
            index_state[state[i]] = index[i]
        
        mhd = 0
        for i in range(1,8):#changed
            for j in range(2):
                mhd = abs(index_goal[i][j] - index_state[i][j]) + mhd
        
        return mhd

def chooseMax(node):
        goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)
        state = node.state
        index_goal = {0:[2,2], 1:[0,0], 2:[0,1], 3:[0,2], 4:[1,0], 5:[1,1], 6:[1,2], 7:[2,0], 8:[2,1]}
        index_state = {}
        index = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]
        x, y = 0, 0
        
        for i in range(len(state)):
            index_state[state[i]] = index[i]
        
        mhd = 0
        for i in range(1,8):#changed
            for j in range(2):
                mhd = abs(index_goal[i][j] - index_state[i][j]) + mhd
        
        chosenMax=max(sum(s != g for (s, g) in zip(node.state, goal)), mhd)
	
        return chosenMax
	

#Display Function for EightPuzzle
def display(state):
	state=list(state)
	for n, i in enumerate(state):
		if i == 0:
			state[n] = '*'
	state=tuple(state)
	print(state[0], state[1], state[2])
	print(state[3], state[4], state[5])
	print(state[6], state[7], state[8])

#Make random EightPuzzle Initial State
def make_rand8puzzle():
	y = [x for x in range(9)]
	random.shuffle(y)
	y = tuple(y)
	
	while EightPuzzle(y).check_solvability(y) == 0:
		y=list(y)
		random.shuffle(y)
		y=tuple(y)
	return y

###EightPuzzle main code begins

#make 10 instances of EightPuzzle
for i in range (10):	
     print("EightPuzzle Instance: ", i+1)
     test = make_rand8puzzle()
     display(test)
     A=EightPuzzle(test)
     start = time.time()
     lenSol1 = len(astar_search1(A).solution())
     end = time.time()
     print('Misplaced Tile Hueristic')
     print('Running Time: ',end-start)
     print('Length of Solution: ', lenSol1)
     print('Nodes Removed :', nodesRemoved)

     print('--------------------------------------')

     start = time.time()
     lenSol2 = len(astar_search1(A,manhattan).solution())
     end = time.time()
     print('Manhattan distance heuristic')
     print('Running Time: ',end-start)
     print('Length of Solution: ', lenSol2)
     print('Nodes Removed :', nodesRemoved)

     print('--------------------------------------')

     start = time.time()
     lenSol3 = len(astar_search1(A,chooseMax).solution())
     end = time.time()
     print('Max of Misplaced Tile Hueristic and Manhattan distance heuristic')
     print('Running Time: ',end-start)
     print('Length of Solution: ', lenSol3)
     print('Nodes Removed :', nodesRemoved)

     print('----------------------------------------------------------')
     print('----------------------------------------------------------')

###EightPuzzle main code Ends
#--------------------------------------------


class YPuzzle(Problem):

    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board,
    where one of the squares is a blank. A state is represented as a tuple of length 9,
    where element at index i represents the tile number  at index i (0 if it's an empty square) """
 
    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """

        self.goal = goal
        Problem.__init__(self, initial, goal)
    
    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)
    
    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """
        
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']       
        index_blank_square = self.find_blank_square(state)
	
	#Modified for Ypuzzle
        if (index_blank_square == 0 or index_blank_square == 1 or index_blank_square == 2 or index_blank_square == 5 or index_blank_square == 8):
            possible_actions.remove('LEFT')
        if (index_blank_square == 0 or index_blank_square == 3 or index_blank_square == 1):
            possible_actions.remove('UP')
        if (index_blank_square == 0 or index_blank_square == 1 or index_blank_square == 4 or index_blank_square == 7 or index_blank_square == 8):
            possible_actions.remove('RIGHT')
        if (index_blank_square == 5 or index_blank_square == 7 or index_blank_square == 8):
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)
	#Modified for Ypuzzle
        if(blank==0 or blank==6): 
            delta = {'UP':-3, 'DOWN':2, 'LEFT':-1, 'RIGHT':1}
        elif(blank==2 or blank==8):
            delta = {'UP':-2, 'DOWN':3, 'LEFT':-1, 'RIGHT':1}
        else:
            delta = {'UP':-3, 'DOWN':3, 'LEFT':-1, 'RIGHT':1}

        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    
    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is 
        h(n) = number of misplaced tiles """

        return sum(s != g for (s, g) in zip(node.state, self.goal))




#Display Function for Ypuzzle
def displayY(state):
	state=list(state)
	for n, i in enumerate(state):
		if i == 0:
			state[n] = '*'
	state=tuple(state)
	print(state[0], " ", state[1])
	print(state[2], state[3], state[4])
	print(state[5], state[6], state[7])
	print(" ", state[8]," ")

#Create random Ypuzzle initial State
def make_rand8puzzle_forY():
        y = (1,2,3,4,5,6,7,8,0)
        Puz = YPuzzle(y)
	
        for x in range(1000):
                posAction = Puz.actions(y)
                z=random.randint(0,len(posAction)-1)
                posAction=list(posAction)
                y=Puz.result(y,posAction[z])

        return y


###YPuzzle main code begins
print("YPuzzle :")
print('--------------------------------------')




#make 10 instances of YPuzzle
for i in range (10):	
     print("YPuzzle Instance: ", i+1)
     z = make_rand8puzzle_forY()
     randY = YPuzzle(z)
     displayY(z)
     start = time.time()
     lenSol1 = len(astar_search1(randY).solution())
     end = time.time()

     print('Misplaced Tile Hueristic')
     print('Running Time: ',end-start)
     print('Length of Solution: ', lenSol1)
     print('Nodes Removed :', nodesRemoved)
     #print(astar_search1(randY).solution())
     print('--------------------------------------')

     start = time.time()
     lenSol2 = len(astar_search1(randY,manhattan).solution())
     end = time.time()
     print('Manhattan distance heuristic')
     print('Running Time: ',end-start)
     print('Length of Solution: ', lenSol2)
     print('Nodes Removed :', nodesRemoved)
     #print(astar_search1(randY,manhattan).solution())
     print('--------------------------------------')

     start = time.time()
     lenSol3 = len(astar_search1(randY,chooseMax).solution())
     end = time.time()
     print('Max of Misplaced Tile Hueristic and Manhattan distance heuristic')
     print('Running Time: ',end-start)
     print('Length of Solution: ', lenSol3)
     print('Nodes Removed :', nodesRemoved)
     #print(astar_search1(randY,chooseMax).solution())

     print('----------------------------------------------------------')
     print('----------------------------------------------------------')

###YPuzzle main code ends
# ...
