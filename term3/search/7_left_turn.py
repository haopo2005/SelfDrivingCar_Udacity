import numpy as np

grid = [[1,1,1,0,0,0],
	    [1,1,1,0,1,0],
		[0,0,0,0,0,0],
		[1,1,1,0,1,1],
		[1,1,1,0,1,1]]

goal = [2,0]
init = [4,3,0]

forward = [[-1, 0], #up
		   [ 0, -1],#left
		   [ 1, 0], #down
		   [ 0, 1]] #right

#cost, right turn, no turn, left turn
cost = [2, 1, 20] #[2, 1, 10] ,quite different result
#假设汽车没有倒车档，只有右转、直行、左转三个
action = [-1, 0, 1] #这些值是应用在forward之上的，比如+1, go down to go right = go left
action_name = ['R', '#', 'L']

value = [[[999 for row in range(len(grid[0]))] for col in range(len(grid))],
		 [[999 for row in range(len(grid[0]))] for col in range(len(grid))],
		 [[999 for row in range(len(grid[0]))] for col in range(len(grid))],
		 [[999 for row in range(len(grid[0]))] for col in range(len(grid))]]

policy = [[[" " for row in range(len(grid[0]))] for col in range(len(grid))],
		  [[" " for row in range(len(grid[0]))] for col in range(len(grid))],
		  [[" " for row in range(len(grid[0]))] for col in range(len(grid))],
		  [[" " for row in range(len(grid[0]))] for col in range(len(grid))]]

policy2D = [[" " for row in range(len(grid[0]))] for col in range(len(grid))]

flag = True

while flag:
	flag = False
	for x in range(len(grid)):
		for y in range(len(grid[0])):
			for ori in range(4):
				if goal[0] == x and goal[1] == y:
					if value[ori][x][y] > 0:
						flag = True
						value[ori][x][y] = 0
						policy[ori][x][y] = "*"
				
				elif grid[x][y] == 0:
					for i in range(len(action)):
						orientation = (ori + action[i]) % 4
						new_x = x + forward[orientation][0]
						new_y = y + forward[orientation][1]
						
						if new_x >=0 and new_x < len(grid) and new_y >=0 and new_y < len(grid[0]) \
						   and grid[new_x][new_y] == 0:
							v2 = value[orientation][new_x][new_y] + cost[i]
							if v2 < value[ori][x][y]:
								value[ori][x][y] = v2
								flag = True
								policy[ori][x][y] = action_name[i]

x = init[0]
y = init[1]
ori = init[2]
policy2D[x][y] = policy[ori][x][y]
o2 = 0
while policy[ori][x][y] != '*':
	if policy[ori][x][y] == '#':
		o2 = ori
	elif policy[ori][x][y] == 'R':
		o2 = (ori - 1) % 4
	elif policy[ori][x][y] == 'L':
		o2 = (ori + 1) % 4
	
	x = x + forward[o2][0]
	y = y + forward[o2][1]
	ori = o2
	policy2D[x][y] = policy[ori][x][y]
	
import pprint
pprint.pprint(policy2D)
						
