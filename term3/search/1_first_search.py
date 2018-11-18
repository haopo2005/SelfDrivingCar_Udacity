import numpy as np
#本次算法计算了出从出发点到终点的cost,暂时没有给出的连续路径
#并不像https://www.cnblogs.com/mr-stn/p/9022142.html，会把多余的访问节点剔除[四个方向都探索过的位置，继续往回走]
#grid format:
#0 = navigable space
#1 = occupied space
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

row = len(grid)
col = len(grid[0])
marked = np.zeros([row,col])

open_list = [[0,init[0],init[1]]]
marked[init[0]][init[1]] = 1
found = False
cant_find = False

while found is False and cant_find is False:
	#print(open_list)
	#print("===============")
	if len(open_list) == 0:
		cant_find = True
		print("search failed!")
		continue

	open_list.sort() #权值从大到小排序,根据open_list第一列元素
	open_list.reverse() #权值从小到达排序
	node = open_list.pop() #取权重最小的
	g = node[0]
	x = node[1]
	y = node[2]

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
				open_list.append((g+1,new_x,new_y))
				marked[new_y][new_x] = 1
		
