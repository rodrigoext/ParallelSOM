/*
 * node.h
 *
 *  Created on: Feb 27, 2014
 *      Author: rodrigo
 */

#ifndef NODE_H_
#define NODE_H_

#include <iostream>
#include <vector>
#include "utils.h"

using namespace std;

class Node {

private:
	
	vector<float> weights;

	int left, right, top, bottom;

public:

	float x,y;
	int count_wins;
	int id;


	Node(int Left, int Right, int Top, int Bottom, int num_weights, int Id) :left(Left), right(Right), top(Top), bottom(Bottom), id(Id)
	{
		for(int w = 0; w < num_weights; w++)
		{
			weights.push_back(RandFloat());
		}

		//calculate the node's center
		x = left + (float)(right - left)/2;
		y = top + (float)(bottom - top)/2;
		count_wins = 0;
		//id = id_node;
	}

	float get_distance(const vector<float> &input_vector)
	{
		float distance = 0;

		for(int i = 0; i < weights.size(); i++)
		{
			distance += (input_vector[i] - weights[i]) * (input_vector[i] - weights[i]);
		}

		//return sqrt(distance);
		return distance;
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
	
	float X()
	{
		//return weights[0];
		return x;
	}

	float Y()
	{
		//return weights[1];
		return y;
	}

	float pesoX()
	{
		return weights[0];
	}

	float pesoY()
	{
		return weights[1];
	}

	vector<float> get_weigths()
	{
		return weights;
	}

};

#endif /* NODE_H_ */
