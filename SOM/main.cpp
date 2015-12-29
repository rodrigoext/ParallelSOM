#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm> 
#include <Eigen/Dense>

//#include <KMeansRexCore.h>

#include "som.h"
#include "node_hex.h"
#include "kmeans.h"
#include "utils.h"

using std::cout;
using std::endl;
using std::ifstream;
using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic> data_set_eigen;

MatrixXf read_data_eigen(const char * file_name)
{
	vector<vector<string>> data;
	MatrixXf data_float;

	ifstream infile;

	infile.open(file_name);

	cerr << "Reading data..." << endl;

	if (infile.is_open())
	{

		int lines = std::count(std::istreambuf_iterator<char>(infile),
			std::istreambuf_iterator<char>(), '\n');

		while (infile)
		{
			string s;

			if (!getline(infile, s))
				break;

			istringstream ss(s);

			vector<float> record;

			Eigen::VectorXf rec;

			int cols = 0;

			while (ss)
			{
				string s2;
				if (!getline(ss, s2, ','))
					break;

				float temp = 0.0f;

				istringstream iss(s2);

				iss >> temp;

				record.push_back(temp);

				//				data_float()

				//			cols += 1;
			}

			lines += 1;
		}
	}

	return data_float;
}

void load_data_set(const char * file_name)
{
	ifstream infile;
	string line;

	infile.open(file_name);

	cerr << "Reading data..." << endl;

	int num_lines = 0;

	while (getline(infile, line))
	{
		++num_lines;
	}

	cerr << "Numeber of lines: " << num_lines << endl;
}

vector<vector<float>> read_data(const char * file_name)
{
	vector<vector<string>> data;
	vector<vector<float>> data_float;

	ifstream infile;

	infile.open(file_name);

	cerr << "Reading data..." << endl;

	int i, j = 0;

	if (infile.is_open())
	{
		while (infile)
		{
			string s;

			if (!getline(infile, s))
				break;

			istringstream ss(s);
			vector<float> record;

			while (ss)
			{
				string s2;
				if (!getline(ss, s2, ','))
					break;

				float temp = 0;

				istringstream iss(s2);

				iss >> temp;

				record.push_back(temp);

			}

			data_float.push_back(record);
		}
	}

	if (!infile.eof())
	{
		cerr << "Fooey!\n";
	}

	/*for(int i = 0; i < data_float.size(); i++)
	{
	for(int j = 0; j < data_float[i].size(); j++)
	{
	cerr << data_float[i][j] << " ";
	}
	cerr << endl;
	}*/

	return data_float;
}

void show_clustered(vector<vector<float>> data, vector<int> clustering, int num_clusters, int decimals)
{
	for (int k = 0; k < num_clusters; ++k)
	{
		cout << "=======================" << endl;
		for (int i = 0; i < data.size(); ++i)
		{
			int cluster_id = clustering[i];

			if (cluster_id != k)
				continue;

			cout << i << "  ";

			for (int j = 0; j < data[i].size(); ++j)
			{
				if (data[i][j] >= 0.0)
					cout << "  ";
				cout << data[i][j] << "  ";
			}
			cout << endl;
		}

		cout << "=======================" << endl;
	}
}

int main()
{
	vector<vector<float>> data_set = read_data("data_seis.csv");

	//load_data_set("data_seis.csv");

	mapminmax(data_set);

	data_set = transpose(data_set);

	print_data_normalized(data_set);

	Som* som = new Som();


	//KMeans* km = new KMeans();

	//vector<int> result =  km->cluster(data_set, 3);

	//cout << endl;

	//show_clustered(data_set_test, result, 3, 0);

	cerr << "Creating Map..." << endl;

	int msize = 100;

	som->create(msize, msize, msize, msize, 1000, data_set[0].size());

	int count = 0;

	cerr << "Training..." << endl;

	while (!som->finished_training())
	{
		som->epoch(data_set);
		count++;
	}

	cerr << "Printing results..." << endl;
	//print_weights(msize, som);
	//print_hits(msize, som);
	print_best_data_neuron(msize, som, data_set);
	//print_umat(msize, som);

	cout << "Num Iterations: " << count << endl;
	cout << "Learning Rate: " << som->get_learning_rate() << endl;

	return 0;
}