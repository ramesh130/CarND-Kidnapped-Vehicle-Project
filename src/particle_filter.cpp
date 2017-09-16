/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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
    num_particles = 4; //Don't need more than that.
    
    default_random_engine gen;
    
    normalDistribution N_x(x, std[0]);
    normalDistribution N_y(y, std[1]);
    normalDistribution N_theta(theta, std[2]);
    
    for (int i = 0; i < num_particles; ++i) {
        Particle particle;
        particle.id = i;
        particle.x = N_x(gen);
        particle.y = N_y(gen);
        particle.theta = N_theta(gen);
        particle.weight = 1;
        
        particles.push_back(particle);
    }
    
    is_initialized = true;
}



void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    default_random_engine gen;
    
    for (Particle& particle: particles) {
        double predictionX;
        double predictionY;
        double predictionTh;
        
        if (yaw_rate == 0) {
            predictionTh = particle.theta;
            predictionX = particle.x + velocity * delta_t * cos(particle.theta);
            predictionY = particle.y + velocity * delta_t * sin(particle.theta);
        } else {
            predictionTh = particle.theta + yaw_rate * delta_t;
            predictionX = particle.x + velocity / yaw_rate * (sin(predictionTh) - sin(particle.theta));
            predictionY = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(predictionTh));
        }
        
        normalDistribution ndX(predictionX, std_pos[0]);
        normalDistribution ndY(predictionY, std_pos[1]);
        normalDistribution ndTh(predictionTh, std_pos[2]);
        
        particle.x = ndX(gen);
        particle.y = ndY(gen);
        particle.theta = ndTh(gen);
    }
}

// See comment in ParticleFilter::closestLandmarkLocation() implementation below
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

LandmarkObs ParticleFilter::transformObservation(Particle& particle, LandmarkObs& observation) {
    LandmarkObs transformedObs;
    
    transformedObs.x = particle.x + (observation.x*cos(particle.theta) - observation.y*sin(particle.theta));
    transformedObs.y = particle.y + (observation.x*sin(particle.theta) + observation.y*cos(particle.theta));
    
    return transformedObs;
}

ObsevationPackage ParticleFilter::closestLandmarkLocation(LandmarkObs &obs, Map map_landmarks, double threshold) {
    
    // introduce a minimal found distance (minDist) from each observation to some (yet) unidentified landmark.
    // Initially set it to the radar range, (as maximum allowed value of interest)
    double minDist = threshold;
    double x = 0, y = 0;
    int landmarkID = 0;
    
    for (auto& landmark: map_landmarks.landmark_list) {
        double currrentDist = dist(landmark.x_f, landmark.y_f, obs.x, obs.y);
        if (currrentDist < minDist) {
            minDist = currrentDist;
            x = landmark.x_f;
            y = landmark.y_f;
            landmarkID = landmark.id_i; // setting potential association inline, so no need to additionally call ParticleFilter::dataAssociation(..) method.
        }
    }
    
    ObsevationPackage package;
    package.x = x;
    package.y = y;
    package.landmarkID = landmarkID;
    package.valid = (minDist!=threshold);
    
    return package;
}

void ParticleFilter::updateWeight(Particle& particle,
                                  double sensor_range,
                                  double std_landmark[],
                                  std::vector<LandmarkObs> observations,
                                  Map map_landmarks) {
    // 1. Set particle weight to 1.
    particle.weight = 1;
    
    vector<int> associations;
    vector<double> senseX;
    vector<double> senseY;
    
    // 2. Analyze observations and update weight accordingly
    for (LandmarkObs& obs: observations) {
        
        // 2.a. First, transform observation into map's coordinate system
        LandmarkObs transformedObs = transformObservation(particle, obs);
        
        // 2.b. Find location (x, y) of the closest landmark.
        ObsevationPackage minLoc = closestLandmarkLocation(transformedObs, map_landmarks, sensor_range);
        
        if (minLoc.valid) {
            double mux = minLoc.x;
            double muy = minLoc.y;
            
            double obsMultiVariateGaussProb = multivariateGaussProb(std_landmark[0], std_landmark[1], transformedObs.x, transformedObs.y, mux, muy);
            if (obsMultiVariateGaussProb > 0) {
                // 2.c. Update particle's weight if all those nested 'if' conditions have been met.
                particle.weight *= obsMultiVariateGaussProb;
                
                // 2.d. Accumulate associations and transformed observations (for visualization purposes).
                associations.push_back(minLoc.landmarkID);
                senseX.push_back(transformedObs.x);
                senseY.push_back(transformedObs.y);
            }
        }
    }
    
    // 3. Set associations (for visualization purposes).
    SetAssociations(particle, associations, senseX, senseY);
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
    
    for (Particle& particle: particles) {
        updateWeight(particle, sensor_range, std_landmark, observations, map_landmarks);
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    default_random_engine gen;
    
    std::vector<double> weights;
    for (Particle& p: particles) {
        weights.push_back(p.weight);
    }
    
    discrete_distribution<int> distribution(weights.begin(), weights.end());
    
    vector<Particle> resample_particles;
    
    for (int i = 0; i < num_particles; ++i) {
        resample_particles.push_back(particles[distribution(gen)]);
    }
    
    particles = resample_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    ///particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    /// associations: The landmark id that goes along with each listed association
    /// sense_x: the associations x mapping already converted to world coordinates
    /// sense_y: the associations y mapping already converted to world coordinates
    
    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    
    particle.associations = associations;
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
