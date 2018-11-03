/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	is_initialized = true;
	num_particles = 100;
	double std_x, std_y, std_psi; // Standard deviations for x, y, and psi

	default_random_engine gen;

	std_x = std[0];
	std_y = std[1];
	std_psi = std[2];
    // This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_psi(theta, std_psi);
	
	for (int i = 0; i < num_particles; ++i) {
		Particle particle;
		
        particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_psi(gen);
		particle.id = i;
		particle.weight = 1.0;

		weights.push_back(1.0);
		particles.push_back(particle);
		//std::cout<<"particles[i].x:"<<particles[i].x<<",particles[i].y:"<<particles[i].y<<",particles[i].theta:"<<particles[i].theta<<std::endl;
	}
}

//update particle location
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	double std_x, std_y, std_psi; // Standard deviations for x, y, and psi
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_psi = std_pos[2];

	for(int i=0;i<particles.size();i++)
	{
		normal_distribution<double> dist_x(particles[i].x, std_x);
		normal_distribution<double> dist_y(particles[i].y, std_y);
		normal_distribution<double> dist_psi(particles[i].theta, std_psi);

		if (fabs(yaw_rate) > 0.001) {
			particles[i].x = velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta)) + dist_x(gen);
			particles[i].y = velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t)) + dist_y(gen);
			
		}else{
			particles[i].x = velocity*delta_t*cos(particles[i].theta) + dist_x(gen);
			particles[i].y = velocity*delta_t*sin(particles[i].theta) + dist_y(gen);
		}

		particles[i].theta = yaw_rate*delta_t + dist_psi(gen);
        //不需要考虑angle normalization，否则yaw rate报错超过答案标准	
		//std::cout<<"particles[i].x:"<<particles[i].x<<",particles[i].y:"<<particles[i].y<<",particles[i].theta:"<<particles[i].theta<<std::endl;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}

void JstFindNN(Particle &particle, const Map &map_landmarks){
	for(int i=0;i<particle.sense_x.size();i++)
	{
		float min_dist = sqrt(pow((particle.sense_x[i] - map_landmarks.landmark_list[0].x_f),2) + pow(fabs(particle.sense_y[i] - map_landmarks.landmark_list[0].y_f),2));
		int index = 0;
		for(int j=0;j<map_landmarks.landmark_list.size();j++)
		{
			float dist = sqrt(pow((particle.sense_x[i] - map_landmarks.landmark_list[j].x_f),2) + pow(fabs(particle.sense_y[i] - map_landmarks.landmark_list[j].y_f),2));
			if(dist < min_dist)
			{
				min_dist = dist;
				index = j;
			}
		}
		particle.associations.push_back(index);
	}
	
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	float sum_weight = 0.0f;
	for(int i=0;i<particles.size();i++)
	{
		Particle particle = particles[i];
		for(int j=0;j<observations.size();j++)
		{
			float x = particle.x + cos(particle.theta) * observations[j].x - sin(particle.theta) * observations[j].y;
			float y = particle.y + sin(particle.theta) * observations[j].x + cos(particle.theta) * observations[j].y;

			if(fabs(x-particle.x)>sensor_range || fabs(y-particle.y)>sensor_range)
			{
				std::cout<<"exceed the sensor ranges, abandoned!"<<std::endl;
				continue;
			}
			particle.sense_x.push_back(x);
			particle.sense_y.push_back(y);
		}
		//看了5个人的答案，最好不要学我，应该先通过sensor range挑选候选map landmark，作为predict set
		//然后，再对上面的measurment(全局坐标)赋予最近邻的map landmark(predict set)
		//我这边的最近邻效率没有上述高，统统丢进去找最近邻了
		JstFindNN(particle, map_landmarks);
		float gauss_norm = (1/(2 * M_PI * std_landmark[0] * std_landmark[1]));
		float prob = 1.0;
		for(int j=0;j<particle.sense_x.size();j++)
		{
			float x = particle.sense_x[j];
			float x_mu = map_landmarks.landmark_list[particle.associations[j]].x_f;
			float y = particle.sense_y[j];
			float y_mu = map_landmarks.landmark_list[particle.associations[j]].y_f;
			//std::cout<<"x:"<<x<<"y:"<<y<<",x_mu:"<<x_mu<<",y_mu:"<<y_mu<<std::endl;
			float exponent = (x-x_mu)*(x-x_mu)/(2*std_landmark[0]*std_landmark[0]) + (y-y_mu)*(y-y_mu)/(2*std_landmark[1]*std_landmark[1]);
			//std::cout<<"exponent:"<<exponent<<std::endl;
			prob = prob*gauss_norm * exp(-exponent);
		}
		//particle.weight = prob;
		weights[i] = prob;
		particles[i].weight = prob;
		sum_weight += prob;
	}
	//权值归一化非必须，normalization
	for(int i=0;i<num_particles;i++)
	{
		weights[i] /= sum_weight;
		particles[i].weight /= sum_weight;
		//std::cout<<"weight:"<<weights[i]<<std::endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	vector<Particle> p3;

    discrete_distribution<size_t> d(weights.begin(),weights.end());
	/*
	int index = d(gen);
    float beta = 0.0;
    
	auto maxPosition = max_element(weights.begin(), weights.end());
	float mw = *maxPosition;
    for(int i=0;i<num_particles;i++){
        beta += random() * 2.0 * mw;
        while(beta > weights[index]){
            beta -= weights[index];
            index = (index + 1) % num_particles;
		}
        p3.push_back(particles[index]);
	}*/
	for(int i=0;i<num_particles;i++){
		int index = d(gen);
		p3.push_back(particles[index]);
	}
    particles = p3;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
