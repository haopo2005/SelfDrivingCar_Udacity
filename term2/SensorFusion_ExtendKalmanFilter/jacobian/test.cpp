#include <iostream>
#include "Dense"

using namespace std;
using namespace Eigen;


MatrixXd CalculateJacobian(const VectorXd &x_state);

int main()
{
	//模拟lidar的预测结果
	VectorXd x_predicted(4);
	x_predicted << 1, 2, 0.2, 0.4;
	
	MatrixXd Hj = CalculateJacobian(x_predicted);
	
	cout << "Hj:" << endl << Hj << endl;
	return 0;
}

MatrixXd CalculateJacobian(const VectorXd &x_state)
{
	MatrixXd Hj(3,4);
	
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);
	
	//check division by zero
	if((px*px + py*py) < 1e-6)
		cout << "warning, divided by zero" << endl;
	//compute the Jacobian matrix
	Hj(0,0) = px / sqrt(px*px+py*py);
	Hj(0,1) = py / sqrt(px*px+py*py);
	Hj(0,2) = 0;
	Hj(0,3) = 0;
	
	Hj(1,0) = -py / (px*px+py*py);
	Hj(1,1) = px / (px*px+py*py);
	Hj(1,2) = 0;
	Hj(1,3) = 0;
	
	Hj(2,0) = py*(vx*py-vy*px)/pow((px*px+py*py),1.5);
	Hj(2,1) = px*(vy*px-vx*py)/pow((px*px+py*py),1.5);
	Hj(2,2) = px / sqrt(px*px+py*py);
	Hj(2,3) = py / sqrt(px*px+py*py);
	
	return Hj;
	
}