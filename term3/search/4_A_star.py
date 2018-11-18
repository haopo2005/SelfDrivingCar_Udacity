import numpy as np
'''
grid = [[0,0,1,0,0,0],
        [0,0,1,0,0,0],
        [0,0,1,0,1,0],
        [0,0,0,0,1,0],
        [0,0,1,0,1,0]]
'''
grid = [[0,0,0,0,0,0],
        [0,1,1,1,1,0],
        [0,1,1,0,0,0],
        [0,1,0,0,0,0],
        [0,1,0,0,1,0]]

heuristic = [[9,8,7,6,5,4],
             [8,7,6,5,4,3],
             [7,6,5,4,3,2],
             [6,5,4,3,2,1],
             [5,4,3,2,1,0]]

init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
delta = [[-1,0],
         [0,-1],
         [1,0],
	 [0,1]]

row = len(grid)
col = len(grid[0])
marked = np.zeros([row,col])
expand = np.zeros([row,col])
for i in range(row):
    for j in range(col):
            expand[i][j] = -1

f_value = heuristic[init[0]][init[1]]
open_list = [[0+f_value,0,f_value,init[0],init[1]]]
marked[init[0]][init[1]] = 1
found = False
cant_find = False
count = 0

while found is False and cant_find is False:
	if len(open_list) == 0:
		cant_find = True
		print("search failed!")
		continue

	open_list.sort() #权值从大到小排序,根据open_list第一列元素
	open_list.reverse() #权值从小到达排序
	node = open_list.pop() #取权重最小的
	g = node[1]
	x = node[3]
	y = node[4]
	expand[y][x] = count
	count = count + 1

	if x == goal[1] and y == goal[0]:
		found = True
		print("found!!!",g,y,x)
		continue

	for i in range(len(delta)):
		new_x = x + delta[i][0]
		new_y = y + delta[i][1]
		#print(new_x,new_y)
		if new_x>=0 and new_x<col and new_y>=0 and new_y<row:
			if marked[new_y][new_x] != 1 and grid[new_y][new_x] != 1:
                                new_x = x + delta[i][0]
                                new_y = y + delta[i][1]
                                f_value = heuristic[new_y][new_x]
                                open_list.append((g+1+f_value,g+1,f_value,new_x,new_y))	
                                marked[new_y][new_x] = 1
print(expand)
