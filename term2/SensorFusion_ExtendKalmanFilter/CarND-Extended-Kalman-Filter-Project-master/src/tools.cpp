#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
    VectorXd rmse(4);
    rmse << 0,0,0,0;
	
	if(estimations.size() != ground_truth.size() || estimations.size() == 0){
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}
	
	for(unsigned int i=0; i < estimations.size(); ++i){

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
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
