import numpy as np
#grid format:
#0 = navigable space
#1 = occupied space
grid = [[0,0,1,0,0,0],
        [0,0,1,0,0,0],
        [0,0,1,0,1,0],
        [0,0,1,0,1,0],
        [0,0,0,0,1,0]]

init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
delta = [[-1,0],
         [0,-1],
         [1,0],
         [0,1]]
delta_name = ['^','<','v','>']
row = len(grid)
col = len(grid[0])
marked = np.zeros([row,col])

#用一个policy数组作为栈，记录每次前行方向
#打印路径时，再pop所有policy，逆序
policy = [[0 for i in range(col)] for j in range(row)]
path = [[' ' for i in range(col)] for j in range(row)]



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
    
    if x == goal[0] and y == goal[1]:
        found = True
        print("found!!!",g,y,x)
        continue
    
    for i in range(len(delta)):
        new_x = x + delta[i][0]
        new_y = y + delta[i][1]
        #print(new_x,new_y)
        if new_x>=0 and new_x<row and new_y>=0 and new_y<col:
            if marked[new_x][new_y] != 1 and grid[new_x][new_y] != 1:
                open_list.append((g+1,new_x,new_y))
                marked[new_x][new_y] = 1
                policy[new_x][new_y] = i

path[goal[0]][goal[1]] = '*'
x = goal[0]
y = goal[1]

#or,仅当xy回到初始，停止循环，两个非，导致and 变成or
while x != init[0] or y != init[1]:
    x_next = x - delta[policy[x][y]][0]
    y_next = y - delta[policy[x][y]][1]
    path[x_next][y_next] = delta_name[policy[x][y]] 
    x = x_next
    y = y_next

import pprint
pprint.pprint(path)                

