#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */

GNB::GNB() {
    for(int i=0;i<3;i++)
    {
        s_avg.push_back(0.0);
        d_avg.push_back(0.0);
        s_dot_avg.push_back(0.0);
        d_dot_avg.push_back(0.0);
        
        s_std.push_back(0.0);
        d_std.push_back(0.0);
        s_dot_std.push_back(0.0);
        d_dot_std.push_back(0.0);
    }
}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{
	/*
		Trains the classifier with N data points and labels.

		INPUTS

		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/

    //
    // Statistics for p_left, p_keep and p_right
    // p_total = p_left + p_keep + p_right = 1
    // p_left = left_cnt/total_cnt and so on.
    //
    // Or shall we assume 33.3% of all three cases?
    //

    // Features:
    // Use  d_in_lane = d % lane_width,
    //      d_dot = delta_d/delta_t
    //      s_dot = delta_s/delta_t
    // Use Gaussian distribution to emulate the probability
    //      need to get u and sigma of each feature
    //      u = average over all same labelled data
    //      sigma^2 = sum((x_i - u)^2)/N
    int left_num = 0; 
    int keep_num = 0;  
    int right_num = 0;
   
    //求mu，均值
    for(int i=0;i<labels.size();i++)
    {
        if(labels[i] == "left")
        {
            s_avg[0] += data[i][0];
            d_avg[0] += data[i][1];
            s_dot_avg[0] += data[i][2];
            d_dot_avg[0] += data[i][3];
            left_num++;
        }else if(labels[i] == "keep")
        {
            s_avg[1] += data[i][0];
            d_avg[1] += data[i][1];
            s_dot_avg[1] += data[i][2];
            d_dot_avg[1] += data[i][3];
            keep_num++;
        }else if(labels[i] == "right")
        {
            s_avg[2] += data[i][0];
            d_avg[2] += data[i][1];
            s_dot_avg[2] += data[i][2];
            d_dot_avg[2] += data[i][3];
            right_num++;
        }
    }
    s_avg[0] /= left_num;
    s_avg[1] /= keep_num;
    s_avg[2] /= right_num;
    
    d_avg[0] /= left_num;
    d_avg[1] /= keep_num;
    d_avg[2] /= right_num;
    
    s_dot_avg[0] /= left_num;
    s_dot_avg[1] /= keep_num;
    s_dot_avg[2] /= right_num;
    
    d_dot_avg[0] /= left_num;
    d_dot_avg[1] /= keep_num;
    d_dot_avg[2] /= right_num;
    
    
    //求方差  
    for(int i=0;i<labels.size();i++)
    {
        if(labels[i] == "left")
        {
            s_std[0] += (data[i][0] - s_avg[0])*(data[i][0] - s_avg[0]);
            d_std[0] += (data[i][1] - d_avg[0])*(data[i][1] - d_avg[0]);
            s_dot_std[0] += (data[i][2] - s_dot_avg[0])*(data[i][2] - s_dot_avg[0]);
            d_dot_std[0] += (data[i][3] - d_dot_avg[0])*(data[i][3] - d_dot_avg[0]);
        }else if(labels[i] == "keep")
        {
            s_std[1] += (data[i][0] - s_avg[1])*(data[i][0] - s_avg[1]);
            d_std[1] += (data[i][1] - d_avg[1])*(data[i][1] - d_avg[1]);
            s_dot_std[1] += (data[i][2] - s_dot_avg[1])*(data[i][2] - s_dot_avg[1]);
            d_dot_std[1] += (data[i][3] - d_dot_avg[1])*(data[i][3] - d_dot_avg[1]);
        }else if(labels[i] == "right")
        {
            s_std[2] += (data[i][0] - s_avg[2])*(data[i][0] - s_avg[2]);
            d_std[2] += (data[i][1] - d_avg[2])*(data[i][1] - d_avg[2]);
            s_dot_std[2] += (data[i][2] - s_dot_avg[2])*(data[i][2] - s_dot_avg[2]);
            d_dot_std[2] += (data[i][3] - d_dot_avg[2])*(data[i][3] - d_dot_avg[2]);
        }
    }
    s_std[0] = s_std[0]/left_num;
    s_std[1] = s_std[1]/keep_num;
    s_std[2] = s_std[2]/right_num;
    
    d_std[0] = d_std[0]/left_num;
    d_std[1] = d_std[1]/keep_num;
    d_std[2] = d_std[2]/right_num;
    
    s_dot_std[0] = s_dot_std[0]/left_num;
    s_dot_std[1] = s_dot_std[1]/keep_num;
    s_dot_std[2] = s_dot_std[2]/right_num;
    
    d_dot_std[0] = d_dot_std[0]/left_num;
    d_dot_std[1] = d_dot_std[1]/keep_num;
    d_dot_std[2] = d_dot_std[2]/right_num;
}



string GNB::predict(vector<double> sample)

{
	/*
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""

		# TODO - complete this
	*/

    //
    // Prediction can be done using Gaussian Naive Bayes method.
    // Wikipedia has a very good example of sex classification here:
    //  https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes
    // The theoretcal explanation is also good:
    //  https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Parameter_estimation_and_event_models
    //
    // Behavior occuring probability
    double p_left = 1.0;
    double p_keep = 1.0;
    double p_right = 1.0;
       
    p_left *= gaussian_prob(sample[0], s_avg[0], s_std[0]);
    p_left *= gaussian_prob(sample[1], d_avg[0], d_std[0]);
    p_left *= gaussian_prob(sample[2], s_dot_avg[0], s_dot_std[0]);
    p_left *= gaussian_prob(sample[3], d_dot_avg[0], d_dot_std[0]);

    p_keep *= gaussian_prob(sample[0], s_avg[1], s_std[1]);
    p_keep *= gaussian_prob(sample[1], d_avg[1], d_std[1]);
    p_keep *= gaussian_prob(sample[2], s_dot_avg[1], s_dot_std[1]);
    p_keep *= gaussian_prob(sample[3], d_dot_avg[1], d_dot_std[1]);

    p_right *= gaussian_prob(sample[0], s_avg[2], s_std[2]);
    p_right *= gaussian_prob(sample[1], d_avg[2], d_std[2]);
    p_right *= gaussian_prob(sample[2], s_dot_avg[2], s_dot_std[2]);
    p_right *= gaussian_prob(sample[3], d_dot_avg[2], d_dot_std[2]);
    
    
    if(p_left > p_keep && p_left > p_right)
    {
        return "left";
    }else if(p_keep > p_left && p_keep > p_right)
    {
        return "keep";
    }else if(p_right > p_left && p_right > p_keep)
    {
        return "right";
    }
}

double GNB::gaussian_prob(double input, double avg, double std)
{
    double num = (input - avg) * (input - avg);
    double denum = 2*std;
    double norm = 1 / sqrt(2*3.1415926*std);
    return norm * exp(-num/denum);
}
