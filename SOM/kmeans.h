#pragma once
#ifndef KMEANS_H_
#define KMEANS_H_

#include <iostream>
#include <vector>
#include <time.h>

#include "utils.h"

using namespace std;

class KMeans
{

private:
	//vector<vector<float>> data;
	//int num_clusters;

public:

	KMeans(void){}

	vector<int> cluster(vector<vector<float>> data_set, int num_cluster)
	{
		data_set = normalized(data_set);

		bool changed = true; 
        bool success = true;

		vector<int> clustering = init_clustering(data_set.size(), num_cluster);
		vector<vector<float>> means = allocate(num_cluster, data_set[0].size());

		int max_count = data_set.size() * 10;
		int ct = 0;

		while (changed ==  true && success == true && ct < max_count)
		{
			++ct;
			success = update_means(data_set, clustering, means);
			changed = update_clustering(data_set, clustering, means);
		}

		return clustering;
	}

	vector<vector<float>> normalized(vector<vector<float>> data)
	{
		vector<vector<float>> result;

		for (int i = 0; i < data.size(); ++i)
		{
			result.push_back(data[i]);
		}

		for(int j = 0; j < data[0].size(); ++j)
		{
			float colSum = 0;

			for(int i = 0; i < result.size(); ++i)
			{
				colSum += result[i][j];
			}

			float mean = colSum / result.size();
			float sum = 0.0;

			for(int i = 0; i < result.size(); ++i)
			{
				sum += (result[i][j] - mean) * (result[i][j] - mean);
			}

			float sd = sum / result.size();

			for (int i = 0; i < result.size(); ++i)
			{
				result[i][j] = (result[i][j] - mean) /sd;
			}
		}

		return result;
	}

	vector<int> init_clustering(int num_tuples, int num_cluster)
	{
		vector<int> clustering;

		for(int i = 0; i < num_tuples; ++i)
		{
			clustering.push_back(0);
		}

		for(int i = 0; i < num_cluster; ++i)
		{
			clustering[i] = i;
		}

		srand( (unsigned)0 );

		for(int i = num_cluster; i < clustering.size(); ++i)
		{
			clustering[i] = rand() % num_cluster;
		}

		return clustering;
	}

	vector<vector<float>> allocate( int num_clusters, int num_columns)
	{
		vector<vector<float>> result;

		for(int i = 0; i < num_clusters; ++i)
		{
			vector<float> temp(num_columns);
			result.push_back(temp);
		}

		return result;
	}

	bool update_means(vector<vector<float>> data_set, vector<int> clustering, vector<vector<float>> &means)
	{
		int num_clusters = means.size();

		vector<int> cluster_counts(num_clusters);

		for(int i = 0; i < data_set.size(); ++i)
		{
			int cluster = clustering[i];
			++cluster_counts[cluster];
			++cluster_counts[cluster];
		}

		for(int k = 0; k < num_clusters; ++k)
		{
			if(cluster_counts[k] == 0)
			{
				return false;
			}
		}

		for(int k = 0; k < means.size(); ++k)
		{
			for(int j = 0; j<means[k].size(); ++j)
			{
				means[k][j] = 0.0;
			}
		}

		for(int i = 0; i < data_set.size(); ++i)
		{
			int cluster = clustering[i];
			for(int j = 0; j < data_set[i].size(); ++j)
			{
				means[cluster][j] += data_set[i][j];
			}
		}

		for(int k = 0; k < means.size(); ++k)
		{
			for(int j = 0; j < means[k].size(); ++j)
			{
				means[k][j] /= cluster_counts[k];
			}
		}

		return true;
	}

	bool update_clustering(vector<vector<float>> data, vector<int> clustering, vector<vector<float>> means)
	{
		int num_clusters = means.size();
		bool changed = false;

		vector<int> new_clustering;

		for(int i = 0; i < clustering.size(); ++i)
		{
			new_clustering.push_back(clustering[i]);
		}

		vector<float> distances;

		for(int i = 0; i < num_clusters; i++)
		{
			distances.push_back(0.0);
		}

		for(int i = 0; i < data.size(); ++i)
		{
			for(int k = 0; k < num_clusters; ++k)
			{
				distances[k] = dist(data[i], means[k]);
			}

			int new_cluster_id = min_index(distances);

			if(new_cluster_id != new_clustering[i])
			{
				changed = true;
				new_clustering[i] = new_cluster_id;
			}
		}

		if(changed == false)
			return false;

		vector<int> cluster_counts;

		for(int i = 0; i < num_clusters; ++i)
		{
			cluster_counts.push_back(0);
		}
		
		for(int i = 0; i < data.size(); ++i)
		{
			int cluster = new_clustering[i];
			++cluster_counts[cluster];
		}

		for(int k = 0; k < num_clusters; ++k)
		{
			if(cluster_counts[k] == 0)
			{
				return false;
			}
		}

		for(int i = 0; i < new_clustering.size(); ++i)
		{
			clustering[i] = new_clustering[i];
		}
		return true;
	}
	~KMeans(void);
};

#endif

