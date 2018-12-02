#!/usr/bin/env python

from ptg import PTG
from helpers import Vehicle, show_trajectory

# the self driving car[blue] was trying to get behind the target vehicle[red], 
# but the cost functions it was using weren't weighted appropriately and so it didn't behave as expected.

def main():
	vehicle = Vehicle([0,10,0, 0,0,0]) #定义参考汽车
	predictions = {0: vehicle} #参考物的运动预测
	target = 0				   #参考物的索引号,This is the vehicle that we are setting our trajectory relative to.
	delta = [0, 0, 0, 0, 0 ,0] #衡量参考物位置与目标位置之间的偏移量
	start_s = [10, 10, 0] #经度坐标，速度，加速度
	start_d = [4, 0, 0]   #维度坐标，速度，加速度
	T = 5.0               #期望到达目的位置时间
	best = PTG(start_s, start_d, target, delta, T, predictions)
	show_trajectory(best[0], best[1], best[2], vehicle)

if __name__ == "__main__":
	main()