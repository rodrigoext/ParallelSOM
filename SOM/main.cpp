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

using std::cout;
using std::endl;
using std::ifstream;
using namespace std;
using namespace Eigen;

vector<vector<float>> read_data(const char * file_name)
{
	vector<vector<string>> data;
	vector<vector<float>> data_float;

	ifstream infile;
   
    infile.open(file_name);

	cerr << "Reading data..." << endl;

	if(infile.is_open())
	{
		while(infile)
		{
	 		string s;
		
			if(!getline(infile, s)) 
				break;

			istringstream ss(s);
			vector<float> record;

			while(ss)
			{
				string s2;
				if(!getline(ss, s2, ','))
					break;

				float temp = 0;

				istringstream iss(s2);

				iss >> temp;

				record.push_back(temp);
			}

			data_float.push_back(record);
		}
	}

	if(!infile.eof())
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

vector<vector<float>> read_data_test()
{
	vector<vector<float>> data;

	vector<float> ins;

	ins.push_back(65.0);
	ins.push_back(220.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(73.0);
	ins.push_back(160.0);
	data.push_back(ins);
	ins.clear();

	ins.push_back(59.0);
	ins.push_back(110.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(61.0);
	ins.push_back(120.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(75.0);
	ins.push_back(150.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(67.0);
	ins.push_back(240.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(68.0);
	ins.push_back(230.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(70.0);
	ins.push_back(220.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(62.0);
	ins.push_back(130.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(66.0);
	ins.push_back(210.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(77.0);
	ins.push_back(190.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(75.0);
	ins.push_back(180.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(74.0);
	ins.push_back(170.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(70.0);
	ins.push_back(210.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(61.0);
	ins.push_back(110.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(58.0);
	ins.push_back(100.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(66.0);
	ins.push_back(230.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(59.0);
	ins.push_back(120.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(68.0);
	ins.push_back(210.0);
	data.push_back(ins);
	ins.clear();
	
	ins.push_back(61.0);
	ins.push_back(130.0);
	data.push_back(ins);
	ins.clear();

	return data;
}

void show_clustered(vector<vector<float>> data, vector<int> clustering, int num_clusters, int decimals)
{
	for (int k = 0; k < num_clusters; ++k)
	{
		cout << "=======================" << endl;
		for(int i = 0; i < data.size(); ++i)
		{
			int cluster_id = clustering[i];
			
			if(cluster_id != k)
				continue;

			cout << i << "  ";

			for(int j = 0; j < data[i].size(); ++j)
			{
				if(data[i][j] >= 0.0)
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
	
	/*
	
	-- No final obter os pesos dos neur�nios que seriam as posi��es deles no mapa, e n�o x e y como eu estava pensando.

	-- A rede esta com o tereinamento muito r�pido e tenho que ver melhor o por que disso.
	
	*/

	//Carrega dados
	vector<vector<float>> data_set = read_data("data_simple.csv");

	//cout << data_set.pop_back.pop_back() << endl;

	//VectorXf vector_data;
	//MatrixXf data;
	////data = data_set;

	//MatrixXd m = MatrixXd::Random(3,3);
	//m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
	//cout << "m =" << endl << m << endl;
	//VectorXd v(3);
	//v << 1, 2, 3;
	//cout << "m * v =" << endl << m * v << endl;


	//NodeHex* n = new NodeHex(10);


	//cout << "vetoor = " << n->wv << endl;

	//vector<vector<float>> data_set_test = read_data_test();

	Som* som = new Som();
	

	//KMeans* km = new KMeans();
	
	//vector<int> result =  km->cluster(data_set, 3);

	//cout << endl;

	//show_clustered(data_set_test, result, 3, 0);

	int msize = 4;

	som->create(msize, msize, msize, msize, 1000);

	int count = 0;

	while (!som->finished_training())
	{
		som->epoch(data_set);
		count ++;
	}
		
	//ofstream resultado_final;
	ofstream pesos_neuronios;
	//resultado_final.open ("data_seis_result.txt");
	pesos_neuronios.open("pesos_neuronios.csv");
	
	//som->calculate_result_bmu(data_set);

	//reverse(som->result.begin(),som->result.end());

	for (int i = 0; i < msize*msize; ++i)
	{
		pesos_neuronios << som->SOM[i].pesoX() << "," << som->SOM[i].pesoY() << endl;
	}

	/*for(int i = 0; i < data_set.size(); ++i)
	{
		resultado_final << som->result.back() << endl;
		som->result.pop_back();
	}
		
	resultado_final.close();*/
	pesos_neuronios.close();

	cout << "Num: " << count << endl;

	system("pause");

    return 0;
}