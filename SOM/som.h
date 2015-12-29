#ifndef SOM_H
#define SOM_H

#include "node.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <Eigen/Dense>
#include <omp.h>


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
		
		int size = SOM.size();

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

	Node* find_best_matching_node_parallel(const vector<float> &vec)
	{
		Node* winner = NULL;

		float lowest_distance = 999999;

		int size = SOM.size();
		int job_thread = (int) size / omp_get_max_threads();

		#pragma omp parallel
		{
			for (int n = omp_get_thread_num(); n < job_thread * omp_get_thread_num(); n++)
			{
				float dist = SOM[n].get_distance(vec);

				if (dist < lowest_distance)
				{
					lowest_distance = dist;
					
					//Critial
					#pragma critical
					{
						winner = &SOM[n];
					}
				}
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
	int dimension_neurons;

	Som():cell_width(0), 
		cell_height(0), 
		winning_node(NULL), 
		iteraction_count(1), 
		num_iteractions(0), 
		time_constant(0), 
		map_radius(0), 
		influence(0), 
		learning_rate(constStartLearningRate), 
		done(false)
	{
	}

	void create(int Width, int Height, int cells_up, int cells_across, int Num_iteractions, int dimension)
	{
		cell_width = (float)Width / (float)cells_across;
		cell_height = (float)Height / (float)cells_up;

		map_x = Width;
		map_y = Height;

		num_iteractions = Num_iteractions;
		dimension_neurons = dimension;

		int id_node = 1;
		//create all nodes

		int job_cells_up = (int) cells_up / omp_get_max_threads();
		int job_cells_across = (int) cells_across / omp_get_max_threads();

		for (int i = 0; i < cells_up; i++)
		{
			for (int j = 0; j < cells_across; j++)
			{
				SOM.push_back(
					Node(j * cell_width, //left
					(j+1) * cell_width,	 //right
					i * cell_height,	 //top
					(i+1) * cell_height, //bottom
					dimension, //dimension of weights
					id_node) //id of nodes
					);
				id_node++;
			}
		}

		map_radius = max(Width, Height)/2;

		time_constant = num_iteractions/log(map_radius);
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

			winning_node = find_best_matching_node_parallel(data[this_vector]);

			winning_node->count_win();

//			result.push_back(winning_node->id);

			neighbourhood_radius = map_radius * exp(-(float)iteraction_count/time_constant);

			for(int n = 0; n < SOM.size(); n++)
			{
				//calculate the Euclidean distance (squared) to this node from the
				//BMU
				float dist_to_node_sq = (winning_node->x - SOM[n].X()) *
										(winning_node->x - SOM[n].X()) +
										(winning_node->y - SOM[n].Y()) *
										(winning_node->y - SOM[n].Y());

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

	bool epoch_parallel(const vector<vector<float>> &data)
	{
		//if(data[0].size() != constSizeOfInputVector)
		//	return false;

		if (done)
			return true;

		if (--num_iteractions)
		{
			int this_vector = RandInt(0, data.size() - 1);

			winning_node = find_best_matching_node(data[this_vector]);

			winning_node->count_win();

			//			result.push_back(winning_node->id);

			neighbourhood_radius = map_radius * exp(-(float)iteraction_count / time_constant);

			for (int n = 0; n < SOM.size(); n++)
			{
				//calculate the Euclidean distance (squared) to this node from the
				//BMU
				float dist_to_node_sq = (winning_node->x - SOM[n].X()) *
					(winning_node->x - SOM[n].X()) +
					(winning_node->y - SOM[n].Y()) *
					(winning_node->y - SOM[n].Y());

				float width_sq = neighbourhood_radius * neighbourhood_radius;

				if (dist_to_node_sq < width_sq)
				{
					influence = exp(-(dist_to_node_sq) / (2 * width_sq));

					SOM[n].adjust_weights(data[this_vector], learning_rate, influence);
				}
			}//next node

			//reduce learning rate
			learning_rate = constStartLearningRate * exp(-(float)iteraction_count / num_iteractions);

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

//			result.push_back(winning_node->id);
		}
	}

	float get_learning_rate()
	{
		return this->learning_rate;
	}

	vector<int> get_umat_simples(vector<vector<float>> &data)
	{

		vector<int> umat;

		for (int i = 0; i < data.size(); ++i)
		{
			winning_node = find_best_matching_node(data[i]);
			umat.push_back(winning_node->id);
		}

		return umat;
		
	}

	vector<vector<float>> get_umat()
	{
		vector<float> um;
		vector<vector<vector<float>>> neurons(map_x, vector<vector<float>>(map_y, vector<float>(dimension_neurons)));

		int count = 0;

		for (int i = 0; i < map_x; ++i)
		{
			for (int j = 0; j < map_y; j++)
			{
				neurons[i][j] = SOM[count].get_weigths();
				count++;
			}
		}

		int X = 2 * map_x - 1;
		int Y = 2 * map_y - 1;

		vector<vector<float>> U(X, vector<float>(Y));

		for (int i = 0; i < map_x; ++i)
		{
			for (int j = 0; j < map_y; ++j)
			{
				if (j < (map_y - 1))
				{
					float dx = get_distance(neurons[i][j], neurons[i][j + 1]);
					U[2 * i][2 * j + 1] = dx;
					um.push_back(dx);
				}

				if (i < (map_x - 1))
				{
					float dy = get_distance(neurons[i][j], neurons[i + 1][j]);
					U[2 * i + 1][2 * j] = dy;
					um.push_back(dy);
				}

				if (i < (map_x - 1) && j < (map_y - 1))
				{
					float dz1 = get_distance(neurons[i][j], neurons[i+1][j+1]);
					float dz2 = get_distance(neurons[i+1][j], neurons[i][j+1]);
					U[2 * i + 1][2 * j + 1] = (dz1 + dz2) / (2 * sqrt(2));
				}
			}

			vector<float> a;

			for (int i = 0; i < X; i += 2)
			{
				for (int j = 0; j < Y; j += 2)
				{
					if (j > 0 && i > 0 && i < (X - 1) && j < (Y - 1))
					{
						a.push_back(U[i][j-1]);
						a.push_back(U[i][j+1]);
						a.push_back(U[i-1][j]);
						a.push_back(U[i+1][j]);
					}
					else if (i == 0 && j > 0 && j < (Y - 1))
					{
						a.push_back(U[i][j-1]);
						a.push_back(U[i][j+1]);
						a.push_back(U[i+1][j]);
					}
					else if (i == (X - 1) && j > 0 && j < (Y - 1))
					{
						a.push_back(U[i][j - 1]);
						a.push_back(U[i][j + 1]);
						a.push_back(U[i - 1][j]);
					}
					else if (j == 0 && i > 0 && i < (X - 1))
					{
						a.push_back(U[i][j + 1]);
						a.push_back(U[i - 1][j]);
						a.push_back(U[i + 1][j]);
					}
					else if (j == (Y - 1) && i > 0 && i < (X - 1))
					{
						a.push_back(U[i][j - 1]);
						a.push_back(U[i - 1][j]);
						a.push_back(U[i + 1][j]);
					}
					else if (j == 0 && i == 0)
					{
						a.push_back(U[i][j + 1]);
						a.push_back(U[i + 1][j]);
					}
					else if (i == 0 && j > 0 && j < (Y - 1))
					{
						a.push_back(U[i][j]);
						a.push_back(U[i][j]);
						a.push_back(U[i][j]);
					}
					else if (j == (X - 1) && i == 0)
					{
						a.push_back(U[i][j - 1]);
						a.push_back(U[i + 1][j]);
					}
					else if (j == 0 && i < (X - 1))
					{
						a.push_back(U[i][j + 1]);
						a.push_back(U[i - 1][j]);
					}
					else if (j == (Y - 1) && i == (X - 1))
					{
						a.push_back(U[i][j - 1]);
						a.push_back(U[i - 1][j]);
					}
					else
					{
						a.push_back(0.0f);
					}

					U[i][j] = calculate_median(a);
				}
			}
		}

		return U;
	}


};

#endif
