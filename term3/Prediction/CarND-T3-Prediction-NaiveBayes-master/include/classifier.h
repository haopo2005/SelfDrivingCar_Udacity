#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {

public:
        double lane_width = 4.0; // Lane width is 4 meters.

       
        // Feature average and deviation
        vector<double> s_avg;
        vector<double> d_avg;
        vector<double> s_dot_avg;
        vector<double> d_dot_avg;
        
        vector<double> s_std;
        vector<double> d_std;
        vector<double> s_dot_std;
        vector<double> d_dot_std;

	/**
  	* Constructor
  	*/

 	GNB();

	/**
 	* Destructor
 	*/

 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);

  	string predict(vector<double>);

        double gaussian_prob(double input, double avg, double std);
};



#endif




