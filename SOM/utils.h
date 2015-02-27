#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <random>

#include "node.h"

using namespace std;
//----------------------------------------------------------------------------
//	some random number functions.
//----------------------------------------------------------------------------

//returns a random integer between x and y
inline int	  RandInt(int x,int y) {return rand()%(y-x+1)+x;}

//returns a random float between zero and 1
inline float RandFloat()		   {

	random_device rd;
	mt19937 eng(rd());
	uniform_real_distribution<> dis(-1.0, 1.0);

	return dis(eng);

	//return (rand())/(RAND_MAX+RandInt(-1,1));

}

//returns a random bool
inline bool   RandBool()
{
	if (RandInt(0,1)) return true;

	else return false;
}

inline int min_index(vector<float> distances)
{
	int indexOfMin = 0;
	float smallDist = distances[0];

	for (int k = 0; k < distances.size(); ++k)
	{
		if (distances[k] < smallDist)
		{
			smallDist = distances[k];
			indexOfMin = k;
		}
	}
	return indexOfMin;
}

inline float dist(vector<float> tuple, vector<float> mean)
{
    // Euclidean distance between two vectors for UpdateClustering()
    // consider alternatives such as Manhattan distance
    float sumSquaredDiffs = 0.0;

    for (int j = 0; j < tuple.size(); ++j)
		sumSquaredDiffs += powf((tuple[j] - mean[j]), 2.0f);
	return sqrtf(sumSquaredDiffs);
}

void mapminmax(vector<vector<float>> &data)
{
	float min = 0.0f;
	float max = 0.0f;

	float scale_range = 1.0f - (-1.0f);

	for (int i = 0; i < data.size(); ++i)
	{
		min = *min_element(data[i].begin(), data[i].end());
		max = *max_element(data[i].begin(), data[i].end());

		float value_range = max - min;

		for (int j = 0; j < data[i].size(); ++j)
		{
			//data[i][j] = (data[i][j] - min) / (max - min);
			//data[i][j] = (((scale_range * (data[i][j] - min)) / value_range));
			data[i][j] = (data[i][j] - min) / value_range * scale_range + (-1.0f);
		}
	}
}

vector<vector<float>> transpose(vector<vector<float>> &data)
{
	vector<vector<float>> new_data(data[0].size(),vector<float>(data.size()));

	for (int i = 0; i < data.size(); ++i)
	{
		for (int j = 0; j < data[i].size(); ++j)
		{
			new_data[j][i] = data[i][j];
		}
	}

	return new_data;
}

bool is_par(int num)
{
	if (num % 2 == 0)
		return true;
	return false;
}

float get_distance(const vector<float> &vec1, const vector<float> &vec2)
{
	float distance = 0;

	for (int i = 0; i < vec2.size(); i++)
	{
		distance += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	}

	//return sqrt(distance);
	return distance;
}

vector<vector<float>> get_vizinhos(int x, int y, const vector<vector<vector<float>>> &neurons)
{
	vector<vector<float>> vizinhos;

	if (x == 0)
	{
		if (y == 0)
		{
			vizinhos.push_back(neurons[x][y + 1]);//direita
			vizinhos.push_back(neurons[x][neurons.size() - 1]);//esquerda
			vizinhos.push_back(neurons[x + 1][y]);//abaixo
			vizinhos.push_back(neurons[neurons.size() - 1][y]);//acima
		}
		else if (y == (neurons.size() - 1))
		{
			vizinhos.push_back(neurons[x][y - (neurons.size() - 1)]);//direita
			vizinhos.push_back(neurons[x][y - 1]);//esquerda
			vizinhos.push_back(neurons[x + 1][y]);//abaixo
			vizinhos.push_back(neurons[neurons.size() - 1][y]);//acima
		}
		else
		{
			vizinhos.push_back(neurons[x][y + 1]);//direita
			vizinhos.push_back(neurons[x][y - 1]);//esquerda
			vizinhos.push_back(neurons[x + 1][y]);//abaixo
			vizinhos.push_back(neurons[neurons.size() - 1][y]);//acima
		}
	}
	else if (x == (neurons.size() - 1))
	{
		if (y == 0)
		{
			vizinhos.push_back(neurons[x][y + 1]);//direita
			vizinhos.push_back(neurons[x][neurons.size() - 1]);//esquerda
			vizinhos.push_back(neurons[x - (neurons.size() - 1)][y]);//abaixo
			vizinhos.push_back(neurons[x - 1][y]);//acima
		}
		else if (y == (neurons.size() - 1))
		{
			vizinhos.push_back(neurons[x][y - (neurons.size() - 1)]);//direita
			vizinhos.push_back(neurons[x][y-1]);//esquerda
			vizinhos.push_back(neurons[x - (neurons.size() - 1)][y]);//abaixo
			vizinhos.push_back(neurons[x - 1][y]);//acima
		}
		else
		{
			vizinhos.push_back(neurons[x][y + 1]);//direita
			vizinhos.push_back(neurons[x][y - 1]);//esquerda
			vizinhos.push_back(neurons[x - (neurons.size() - 1)][y]);//abaixo
			vizinhos.push_back(neurons[x - 1][y]);//acima
		}
	}
	else if (y == 0)
	{
		vizinhos.push_back(neurons[x][y + 1]);//direita
		vizinhos.push_back(neurons[x][neurons.size() - 1]);//esquerda
		vizinhos.push_back(neurons[x + 1][y]);//abaixo
		vizinhos.push_back(neurons[x - 1][y]);//acima
	}
	else if (y == (neurons.size() - 1))
	{
		vizinhos.push_back(neurons[x][y - (neurons.size() - 1)]);//direita
		vizinhos.push_back(neurons[x][y - 1]);//esquerda
		vizinhos.push_back(neurons[x + 1][y]);//abaixo
		vizinhos.push_back(neurons[x - 1][y]);//acima
	}
	else
	{
		vizinhos.push_back(neurons[x][y + 1]);
		vizinhos.push_back(neurons[x][y - 1]);
		vizinhos.push_back(neurons[x + 1][y]);
		vizinhos.push_back(neurons[x - 1][y]);
	}

	return vizinhos;
}
#endif