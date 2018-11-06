#include "PID.h"
using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double p, double i, double d) {
    Kp = p;
    Ki = i;
    Kd = d;

    p_error = 0.0;
    i_error = 0.0;
    d_error = 0.0;
    cte_pre = 0.0;
    first_cte = false;
}

void PID::UpdateError(double cte) {
    if(!first_cte)
    {
        cte_pre = cte;
        first_cte = true;
    }
    i_error += cte;
    d_error = cte - cte_pre;
    cte_pre = cte;
    p_error = cte;
}

double PID::TotalError() {
    return -Kp*p_error-Kd*d_error-Ki*i_error;
}

