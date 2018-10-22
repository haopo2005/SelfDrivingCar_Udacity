#include <iostream>
#include "ukf.h"

UKF::UKF() {
  //TODO Auto-generated constructor stub
  Init();
}

UKF::~UKF() {
  //TODO Auto-generated destructor stub
}

void UKF::Init() {

}


/*******************************************************************************
* Programming assignment functions: 
*******************************************************************************/

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out) {

  //set state dimension
  int n_x = 5;

  //set augmented dimension
  int n_aug = 7;

  //create example sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
     Xsig_aug <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  double delta_t = 0.1; //time diff in sec
/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //predict sigma points
  //avoid division by zero
  //write predicted sigma points into right column
  for(int i=0;i<2*n_aug+1;i++)
  {
    VectorXd temp = Xsig_aug.col(i);
    float px = temp(0);
    float py = temp(1);
    float v = temp(2);
    float phi = temp(3);
    float phi_dot = temp(4);
    float v_a = temp(5);
    float v_phi_dot2 = temp(6);
    
    VectorXd second_term(5);
    if(fabs(phi_dot) > 1e-6)
    {
      second_term << px+(v/phi_dot)*(sin(phi+phi_dot*delta_t)-sin(phi)),
                   py+(v/phi_dot)*(-cos(phi+phi_dot*delta_t)+cos(phi)),
                   0,
                   phi+phi_dot*delta_t,
                   phi_dot;
    }else
    {
      std::cout<<"warning, divied by zero"<<std::endl;
      //no more angle velocity noise
      second_term << px+v*delta_t*cos(phi),
                   py+v*delta_t*sin(phi),
                   0,
                   phi+phi_dot*delta_t,
                   phi_dot;
    }
   
    
    VectorXd third_term(5);
    third_term << 0.5*delta_t*delta_t*cos(phi)*v_a,
                  0.5*delta_t*delta_t*sin(phi)*v_a,
                  v+v_a*delta_t,
                  0.5*delta_t*delta_t*v_phi_dot2,
                  delta_t*v_phi_dot2;
    
    Xsig_pred.col(i) = second_term + third_term;
  }

/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  //write result
  *Xsig_out = Xsig_pred;

}
