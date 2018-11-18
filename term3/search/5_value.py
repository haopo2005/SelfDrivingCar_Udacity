import numpy as np
#通过死循环
#value矩阵，将从右下角开始逐步向起始点填充
grid = [[0,0,1,0,0,0],
        [0,0,1,0,0,0],
        [0,0,1,0,1,0],
        [0,0,0,0,1,0],
        [0,0,1,0,1,0]]

init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
delta = [[-1,0],
		 [0,-1],
		 [1,0],
		 [0,1]]
value_matrix = [[99 for i in range(len(grid[0]))] for j in range(len(grid))]
flag = True

while flag == True:
	flag = False
	for x in range(len(grid)):
		for y in range(len(grid[0])):
			if goal[0] == x and goal[1] == y:
				if value_matrix[x][y] >0:
					value_matrix[x][y] = 0
					flag = True
			
			elif grid[x][y] == 0:
				for i in range(len(delta)):
					new_x = x + delta[i][0]
					new_y = y + delta[i][1]
					
					if new_x >=0 and new_x < len(grid) and new_y >=0 and new_y < len(grid[0]) \
						and grid[new_x][new_y] == 0:
							v2 = value_matrix[new_x][new_y] + 1
							if v2 < value_matrix[x][y]:
								flag = True
								value_matrix[x][y] = v2
import pprint
pprint.pprint(value_matrix)
	