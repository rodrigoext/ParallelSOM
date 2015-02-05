#ifndef SOM_H
#define SOM_H

#include "node.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <Eigen/Dense>


#include "constants.h"

using namespace std;
using Eigen::MatrixXd;

class Som {

private:
	
	

	Node* winning_node;

	float map_radius;

	float time_constant;

	int num_iteractions;

	int iteraction_count;

	float neighbourhood_radius;

	float influence;

	float learning_rate;

	bool done;

	float cell_width, cell_height;

	Node* find_best_matching_node(const vector<float> &vec)
	{
		Node* winner = NULL;

		float lowest_distance = 999999;

		for (int n = 0; n < SOM.size(); n++)
		{
			float dist = SOM[n].get_distance(vec);

			if(dist < lowest_distance)
			{
				lowest_distance = dist;

				winner = &SOM[n];
			}
		}

		//winner->count_win();

		return winner;
	}

	inline float gaussian(const float dist, const float sigma);

public:

	vector<Node> SOM;
	vector<int> result;

	int map_x, map_y;
	int qtd_neuronios;

	Som():cell_width(0), cell_height(0), winning_node(NULL), iteraction_count(1), num_iteractions(0), time_constant(0), map_radius(0), influence(0), learning_rate(constStartLearningRate), done(false)
	{
	}

	void create(int Width, int Height, int cells_up, int cells_across, int Num_iteractions)
	{
		cell_width = (float)Width / (float)cells_across;
		cell_height = (float)Height / (float)cells_up;

		num_iteractions = Num_iteractions;

		//int id_node = 1;
		//create all nodes
		for (int i = 0; i < cells_up; i++)
		{
			
			for (int j = 0; j < cells_across; j++)
			{
				

				SOM.push_back(
					Node(j * cell_width, //left
					(j+1) * cell_width,	 //right
					i * cell_height,	 //top
					(i+1) * cell_height, //bottom
					3) //num weights
					);

				//id_node++;
			}
		}

		map_radius = max(Width, Height)/2;

		time_constant = num_iteractions/log(map_radius);
	}

	

	void ninv(int n, int &neuronio_altura, int &neuronio_largura)
	{
		if (n < qtd_neuronios)
			return;
		neuronio_largura = (n - 1) / map_x + 1;
		neuronio_altura = n - (neuronio_largura - 1) * map_x;
	}

	bool epoch(const vector<vector<float>> &data)
	{
		//if(data[0].size() != constSizeOfInputVector)
		//	return false;

		if(done)
			return true;

		if(--num_iteractions)
		{
			int this_vector = RandInt(0, data.size()-1);

			winning_node = find_best_matching_node(data[this_vector]);

			winning_node->count_win();

//			result.push_back(winning_node->id);

			neighbourhood_radius = map_radius * exp(-(float)iteraction_count/time_constant);

			for(int n = 0; n < SOM.size(); n++)
			{
				//calculate the Euclidean distance (squared) to this node from the
				//BMU
				float dist_to_node_sq = (winning_node->x - SOM[n].x) *
										(winning_node->x - SOM[n].x) +
										(winning_node->y - SOM[n].y) *
										(winning_node->y - SOM[n].y);

				float width_sq = neighbourhood_radius * neighbourhood_radius;

				if(dist_to_node_sq < width_sq)
				{
					influence = exp(-(dist_to_node_sq) / (2*width_sq));

					SOM[n].adjust_weights(data[this_vector], learning_rate, influence);
				}
			}//next node

			//reduce learning rate
			learning_rate = constStartLearningRate * exp(-(float)iteraction_count/num_iteractions);

			++iteraction_count;

		}
		
		else
		{
			done = true;
		}

		return true;
	}

	bool finished_training() const
	{
		return done;
	}

	void calculate_result_bmu(const vector<vector<float>> &data)
	{
		for(int i = 0; i < data.size(); ++i)
		{
			int this_vector = RandInt(0, data.size()-1);

			winning_node = find_best_matching_node(data[this_vector]);

			result.push_back(winning_node->id);
		}
	}


};

#endif
