/*
 * node.h
 *
 *  Created on: Feb 27, 2014
 *      Author: rodrigo
 */

#ifndef NODE_HEX_H_
#define NODE_HEX_H_

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "utils.h"

using namespace std;

class NodeHex {

private:
	
	vector<float> weights;
	
	Eigen::MatrixXf w;

	int top, bottom, left_top, right_top, left_bottom, right_bottom;

public:

	Eigen::VectorXf wv;
	float x,y;
	int count_wins;
	int id;

	NodeHex(int num_weights)
	{
		
		Eigen::MatrixXf weights(num_weights,1);
		Eigen::VectorXf we(num_weights);

		for(int i = 0; i < num_weights; i++)
		{
			//weights(i,1) = RandFloat();
			we(i) = RandFloat();
		}

		wv = we;

//		cout << we;

		//cout << weights << endl;

		//w = weights;

		//cout << w << endl;

	}

	float get_distance(const vector<float> &input_vector)
	{
		float distance = 0;

		for(int i = 0; i < weights.size(); i++)
		{
			distance += (input_vector[i] - weights[i]) * (input_vector[i] - weights[i]);
		}

		return sqrt(distance);
	}

	void adjust_weights(const vector<float> &target, const float learning_rate, const float influence)
	{
		for(int w = 0; w < weights.size(); w++)
		{
			weights[w] += learning_rate * influence * (target[w] - weights[w]);
		}
	}

	void count_win()
	{
		count_wins = count_wins + 1;
	}
	

};

#endif /* NODE_HEX_H_ */
