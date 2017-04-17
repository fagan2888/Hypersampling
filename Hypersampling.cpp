#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "FastMat.h"
using namespace std;

using namespace std;

const int N = 20000;
const int dd = 16;

class Gibb {

public:
	int 		number_of_clusters;
	Matrix 		data;
	Vector		assignment = zeros(N,1);
	int			mm = dd + 2;
	double		alpha;
	Vector		hyper_mean = zeros(dd, 1);
	double		kappa0 ;
	Matrix		psi ;  
	Vector		point;
	Matrix		data_std = zeros(dd, dd);


	vector<int>				suffstats_number_of_points;
	//vector<double>			suffstats_mean_of_points_1;
	//vector<double>			suffstats_mean_of_points_2;
	vector<Vector>			suffstats_mean_of_points; // for vectorization in d-dim
	vector<Matrix>          Scatter;
	Matrix					scatter_mat = zeros(dd, dd);

	void	 	initialize(int initial_number_of_clusters, double alpha);
	double 		log_predictive_likelihood(int cluster_id, int data_id);
	double		new_logandCRP_likelihood(int data_id);
	//vector<double>	cluster_assignment_distribution(int data_id);
	double		log_cluster_assign_score(int cid);
	//double		log_predictive_likelihood_dp(int data_id, int cid);
	int			create_cluster();
	void 		destroy_cluster(int cid);
	void 		prune_cluster();
	int 		cluster_assignment(int data_id);
	void 		add_datapoint_to_suffstats(int data_id, int cid);
	void 		remove_datapoint_from_suffstats(int data_id, int cid);

	//Hypersampling functions
	//double Hypersampling_k0(int cluster_id, int data_id);
};




/********************************   initialize function *******************************************************/


void Gibb::initialize(int initial_number_of_clusters, double initial_alpha)
{
	number_of_clusters = initial_number_of_clusters;
	alpha = initial_alpha;
	// Vectorizing to d-dim:
	
	// Done
	for (int i = 0; i<N; i++)
	{
		int x = rand() % number_of_clusters;				// data pointlere table/cluster assign etdik random sekilde
		assignment[i] = x;
	}

	for (int i = 0; i<number_of_clusters; i++)
	{
		suffstats_number_of_points.push_back(0);
		//suffstats_mean_of_points_1.push_back(0.0);
		//suffstats_mean_of_points_2.push_back(0.0);
		suffstats_mean_of_points.push_back(zeros(dd,1));

	}

	for (int i = 0; i<N; i++)
	{
		suffstats_number_of_points[assignment[i]] += 1;			// her bir table'daki elementlerin sayini tapiriq
	}

	//for (int i = 0; i<N; i++)
	//{
	//	suffstats_mean_of_points_1[assignment[i]] += data[i][0] / suffstats_number_of_points[assignment[i]];	// table'daki mean_1'i hesabla
	//	suffstats_mean_of_points_2[assignment[i]] += data[i][1] / suffstats_number_of_points[assignment[i]];	// table'daki mean_2'i hesabla
	//}

	// Vectorization:
	//for (int i = 0; i < N; i++) {
	//	for (int j = 0; j < d; j++) {
	//		suffstats_mean_of_points[assignment[i]][j] += data[i][j] / suffstats_number_of_points[assignment[i]];
	//	}
	//}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < dd; j++) {
			suffstats_mean_of_points[assignment[i]][j] += data[i][j] / suffstats_number_of_points[assignment[i]];
		}
	}

	for (int  i= 0; i < N; i++) {
		for (int j = 0; j < dd; j++) {
			hyper_mean[j] += data[i][j] / N;
		}
	
	}

}




/****************************** 	Predictive Likelihood 		*************************************************************/

// New cluster Log Likelihoods function ::

double Gibb::new_logandCRP_likelihood(int data_id)
{
	Matrix   pred_sigmanew = zeros(dd, dd);
	
	pred_sigmanew = psi; 
	Stut t_distribution(hyper_mean, pred_sigmanew*(kappa0 + 1) / (kappa0*(mm - dd + 1)), mm - dd + 1);  //   ??? why d+2
																							//cout << "new cluster: " << '\n';
	return (t_distribution.likelihood(data[data_id]) + log(alpha));
}






double Gibb::log_predictive_likelihood(int cluster_id, int data_id)
{

	Matrix   pred_sigma = zeros(dd, dd); 
	Vector   cluster_mean = zeros(dd, 1);
	Vector   pred_mean ; // = zeros(d);



	//if (cluster_id >= number_of_clusters)
	//{

	//	pred_sigma = psi; //+ scatter_mat;     //Ask this part from Halid abi
	//	


	//	Stut t_distribution(hyper_mean, pred_sigma*(kappa0 + 1) / (kappa0*(mm - dd + 1)), mm - dd + 1);  //   ??? why d+2
	//	//cout << "new cluster: " << '\n';
	//	return t_distribution.likelihood(data[data_id]);

	//}  // ??? Ask Halid again here about the issue of N-1 or N

	 

	cluster_mean = suffstats_mean_of_points[cluster_id]; //v({ suffstats_mean_of_points_1[cluster_id], suffstats_mean_of_points_2[cluster_id] });

	pred_mean = (cluster_mean * (suffstats_number_of_points[cluster_id] - 1) + hyper_mean*kappa0) / (kappa0 + suffstats_number_of_points[cluster_id] -1);
	pred_sigma = psi + Scatter[cluster_id] + ((cluster_mean - hyper_mean) >> (cluster_mean - hyper_mean)) *(kappa0*(suffstats_number_of_points[cluster_id] - 1) / (kappa0 + suffstats_number_of_points[cluster_id] - 1));

	Stut t_distribution(pred_mean, pred_sigma*(kappa0 + suffstats_number_of_points[cluster_id] ) / ((kappa0 + suffstats_number_of_points[cluster_id]-1)*(mm + suffstats_number_of_points[cluster_id] - dd )), mm + suffstats_number_of_points[cluster_id] - dd);
	//cout << "old cluster: " << '\n';
	return t_distribution.likelihood(data[data_id]);


};



/*******************************	Log cluster assign score	**********************************************************/


double	Gibb::log_cluster_assign_score(int cid)
{
	//if (cid >= number_of_clusters)
	//	return log(alpha);

	return 	log(suffstats_number_of_points[cid]);
};



/******************************      Cluster Assignment Distribution 	*************************************************************/
// Adjust this code to unify the cluster_assignment_distribution and cluster_assignment, by using samplefrom log!!!!!!! 



int	Gibb::cluster_assignment(int data_id)
{

	Vector		scores(number_of_clusters+1);

	for (int i = 0; i < number_of_clusters; i++)
	{
		scores[i] = log_predictive_likelihood(i, data_id);    
		scores[i] += log_cluster_assign_score(i);
		//cout << "Lh: " << log_predictive_likelihood(i, data_id) << '\n';
		//cout << "CRP lh: " << log_cluster_assign_score(i) << '\n';
		//cout << "cid : " << i << '\n';
		//if (isnan(log_predictive_likelihood(i, data_id))) {
		//	cout << "Lh: " << log_predictive_likelihood(i, data_id) << '\n';
		//	cout << "CRP lh: " << log_cluster_assign_score(i) << '\n';
		//	cout << "Index : " << data_id << '\n';
		//}
	}
	
	scores[number_of_clusters] = new_logandCRP_likelihood(data_id);

	int cid = sampleFromLog(scores);
	//cout << id_x; 
	//for (int i = 0; i <  number_of_clusters+1; i++)
	//	cout << "scores "<< i << ": " << scores[i] << '\n';



	//double sum_of_scores = 0;

	//for (int i = 0; i <  number_of_clusters; i++)
	//	sum_of_scores += scores[i];

	//for (int i = 0; i < number_of_clusters; i++)
	//	cout << "scores: " << scores[i] << '\n';
	//cout << "cluster id: " << cid << '\n';

	if (cid > number_of_clusters-1)
	{
		return create_cluster();
	}
	else
		return cid;

};



/***********************************	Log Predictive Interface	***********************************************************/


/***********************************	Create Cluster			**********************************************************/
int	Gibb::create_cluster()
{
	number_of_clusters++;
	suffstats_number_of_points.push_back(0);
	//suffstats_mean_of_points_1.push_back(0.0);
	//suffstats_mean_of_points_2.push_back(0.0);
	// Vectorization:
	/*vector<double>   zeros;

	for (int i = 0; i < d; i++) {
		zeros.push_back(0.0);
	}*/
	suffstats_mean_of_points.push_back(zeros(dd,1));
	Scatter.push_back(zeros(dd, dd));

	return (number_of_clusters - 1);
};





/*********************************	Destroy Cluster			**********************************************************/
void 	Gibb::destroy_cluster(int cid)
{	// Continue from here!!!!
	for (int i = cid; i<number_of_clusters-1 ; i++)
	{
		suffstats_number_of_points[i] = suffstats_number_of_points[i + 1];
		//suffstats_mean_of_points_1[i] = suffstats_mean_of_points_1[i + 1];
		//suffstats_mean_of_points_2[i] = suffstats_mean_of_points_2[i + 1];
		/*
		for (int j = 0; j < dd; j++) {
			suffstats_mean_of_points[i][j] = suffstats_mean_of_points[i+1][j];
		}*/
		suffstats_mean_of_points[i] = suffstats_mean_of_points[i + 1];
		Scatter[i] = Scatter[i + 1];
	
		//for (int j = 0; j < N; j++)
		//{
		//	if (assignment[j] > cid)
		//		assignment[j] -= 1;
		//}
	}

	//This guy is true

	for (int j = 0; j < N; j++)
	{
		if (assignment[j] > cid)
			assignment[j] -= 1;
	}

	suffstats_number_of_points.erase(suffstats_number_of_points.begin() + (number_of_clusters - 1));
	//suffstats_mean_of_points_1.erase(suffstats_mean_of_points_1.begin() + (number_of_clusters - 1));
	//suffstats_mean_of_points_2.erase(suffstats_mean_of_points_2.begin() + (number_of_clusters - 1));

	
	suffstats_mean_of_points.erase(suffstats_mean_of_points.begin() + (number_of_clusters - 1));
	Scatter.erase(Scatter.begin() + (number_of_clusters - 1));

	number_of_clusters--;
};



/**********************************	 Prune Cluster			*********************************************************/
void 	Gibb::prune_cluster()
{
	for (int i = 0; i < number_of_clusters; i++) {
		if (suffstats_number_of_points[i] == 0) {
			destroy_cluster(i);
		}
	}
};




/**********************************	Sample Assignment 		*********************************************************/
//int 	Gibb::sample_assignment(int data_id)
//{
//	vector<double>	scores = cluster_assignment_distribution(data_id);
//
//	double 	random_ = ((double)rand()) / (RAND_MAX + 0.0000001);
//	//double ilkrandom = random_;
//	int 		cid = 0;
//
//	//for (int i=0;i<number_of_clusters;i++)
//	//	cout <<"score "<<i<<" = "<< scores[i] <<" " ;
//	//cout << '\n';
//	try {
//		while (1)
//		{
//			if (random_ > scores[cid])
//			{
//				random_ -= scores[cid];
//				cid++;
//			}
//			else
//				break;
//		}
//	}
//	catch (int e)
//	{
//		system("pause");
//	}
//	/*cout << "cid: " << cid << '\n';
//	cout << "NofC: " << number_of_clusters << '\n';
//	*/
//	/*if (cid >= number_of_clusters-1)
//	{
//	cout << "hell yeah" << '\n';
//	}*/
//
//	if (cid >= number_of_clusters)
//	{
//		return create_cluster();
//	}
//	else
//		return cid;
//};


void 	Gibb::add_datapoint_to_suffstats(int data_id, int cid)
{
	
	Scatter[cid] += (data[data_id] >> data[data_id]) + (suffstats_mean_of_points[cid] >> suffstats_mean_of_points[cid])*suffstats_number_of_points[cid];

	suffstats_mean_of_points[cid] = (suffstats_mean_of_points[cid] *suffstats_number_of_points[cid] + data[data_id]) / (suffstats_number_of_points[cid] + 1);

	suffstats_number_of_points[cid] += 1;
	Scatter[cid] -= (suffstats_mean_of_points[cid] >> suffstats_mean_of_points[cid])*suffstats_number_of_points[cid];
}



void 	Gibb::remove_datapoint_from_suffstats(int data_id, int cid)
{
	/*if (suffstats_number_of_points[cid] == 1) {
		suffstats_number_of_points[cid] -= 1;
	}
	else {*/
		Scatter[cid] = Scatter[cid] + (suffstats_mean_of_points[cid] >> suffstats_mean_of_points[cid])*suffstats_number_of_points[cid] - (data[data_id] >> data[data_id]);
		//suffstats_mean_of_points_1[cid] = (suffstats_number_of_points[cid] * suffstats_mean_of_points_1[cid] - data[data_id][0]) / (suffstats_number_of_points[cid] - 1);
		//suffstats_mean_of_points_2[cid] = (suffstats_number_of_points[cid] * suffstats_mean_of_points_2[cid] - data[data_id][1]) / (suffstats_number_of_points[cid] - 1);
		//
		//for (int j = 0; j < dd; j++) {
		//	suffstats_mean_of_points[cid][j] = (suffstats_number_of_points[cid] * suffstats_mean_of_points[cid][j] - data[data_id][j]) / (suffstats_number_of_points[cid] - 1);
		//}

		suffstats_mean_of_points[cid] = (suffstats_mean_of_points[cid] * suffstats_number_of_points[cid] - data[data_id]) / (suffstats_number_of_points[cid] - 1);


		suffstats_number_of_points[cid] -= 1;
		Scatter[cid] -= (suffstats_mean_of_points[cid] >> suffstats_mean_of_points[cid])*suffstats_number_of_points[cid];
	//}
}



int main()
{
	d = dd;
	init_buffer(dd*dd, dd);
	srand(time(NULL));
	Gibb	example;
	precomputegamLn(dd + N);

	
	ifstream file;
	example.data.resize(N, dd);
	
	file.open("scaled_letters.txt");
	if (file.fail()) {
		cerr << "file open fail" << endl;
	}
	else {
		while (!file.eof())
		{
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < dd; j++) {
					file >> example.data[i][j];
					//cout << example.data[i][j] << '\n';
				}
			}
		}
	}


	Vector		true_labels(N);
	ifstream fil;

	fil.open("letter_labels.txt");
	//cout << "here is Ok !!" << '\n';
	if (fil.fail()) {
		cerr << "file open fail" << endl;
	}
	else {
		while (!fil.eof())
		{
			for (int i = 0; i < N; i++) {
				fil >> true_labels[i];
				//cout << example.data[i][j] << '\n';
			}
		}
	}

	cout << "data's read" << '\n';
	example.initialize(1, 1);



	//Hypersampling Parametrrs//
	double						total_lh;
	//double						total_sum;
	//vector<double>				total_lh_kappa0;
	double						k0_min = 0.025;
	double						k0_max = 1;
	Vector						total_lh_kappa0 = zeros(int((k0_max - k0_min) / 0.025) + 1, 1);
	double						hypersampled_kappa0;
	int							m_min = dd + 2;
	int							m_max = dd + 2 + 100*dd;
	Vector						total_lh_m = zeros(int((m_max - m_min)/(2*dd)), 1);
	int							hypersampled_m;
	double						diagval_min = 0.001;
	double						diagval_max = 1;
	//const int 					dvdim = (diagval_max - diagval_min) / 0.1;
	//Matrix					total_lh_psimatrix(dd, int((diagval_max - diagval_min) / 0.1) + 1);
	double						total_lh_psimatrix[dd][19 + 1];
	Vector						total_lh_psi = zeros(int((diagval_max - diagval_min) / 0.05) + 1, 1);
	Matrix						hypersampled_psi = eye(dd);
	//vector<double>				total_lh_alpha;
	double						alpha_min = 0.1;
	double						alpha_max = 3;
	Vector						total_lh_alpha = zeros(int((alpha_max - alpha_min) / 0.1) + 1, 1);
	double						hypersampled_alpha;
	Vector						f1_scores = zeros(10, 1);
	int							idx = 0;
	int							index = 0;
	int							c_id;



	for (int i = 0; i < example.number_of_clusters; i++) {
		example.Scatter.push_back(std::move(example.scatter_mat));
	}


	for (int i = 0; i < N; i++)
	{
		example.Scatter[example.assignment[i]] += (example.data[i] - example.suffstats_mean_of_points[example.assignment[i]])>> (example.data[i] - example.suffstats_mean_of_points[example.assignment[i]]);
	}

	example.psi = eye(dd) ;  // eye(dd)*20 ;
	example.mm = dd + 2  ;  // example.mm = dd + 2 +100 ;
	example.kappa0 = 0.05 ; // example.kappa0 = 0.1 ;

	for (int i = 0; i < example.number_of_clusters; i++)
		cout << "Table " << i << " : " << example.suffstats_number_of_points[i] << '\n';

	cout << '\n';

	for (int fold = 0; fold < 10; fold++) {


		// Number of sweeps
		for (int number_of_sweeps = 0; number_of_sweeps < 1500; number_of_sweeps++)
		{

			////////////     Gibb Sampling      ////////////
			for (int i = 0; i < N; i++)
			{
				c_id = -1;
				example.remove_datapoint_from_suffstats(i, example.assignment[i]);
				example.prune_cluster();  // Halit abiden sorus, sadece cluster i prune etsek de olur degil mi??
				c_id = example.cluster_assignment(i);
				example.assignment[i] = c_id;
				example.add_datapoint_to_suffstats(i, c_id);

			}

			/*for (int i = 0; i < example.number_of_clusters; i++) {
				cout << "Table " << i << " : " << example.suffstats_number_of_points[i] << '\n';
			}*/












			
				
			
			/*********************************    Hypersampling the kappa0    *********************/

//
//
//			//example.psi = eye(d);//*(m-d-1);
//			//example.kappa0 = 0.01;
//			//Note:  this for loop will include the gibb sampler for sure
			
			index = 0;
			for (double k = k0_min; k < k0_max; k += 0.025) {
				example.kappa0 = k;
				total_lh = 0;
				for (int i = 0; i < N; i++)
				{
					total_lh += example.log_predictive_likelihood(example.assignment[i], i);
				}

				total_lh_kappa0[index] = total_lh / N;
				index++;
			}

			//for (int i = 0; i < total_lh_kappa0.size(); i++) {
			//	total_lh_kappa0[i] = exp(total_lh_kappa0[i]);
			//	
			//}

			//total_sum = 0;

			//for (int i = 0; i < total_lh_kappa0.size(); i++) {
			//	total_sum += total_lh_kappa0[i];
			//}

			//for (int i = 0; i < total_lh_kappa0.size(); i++) {
			//	total_lh_kappa0[i] /= total_sum;
			//	
			//}


			// Sampling the kappa0 from the scores.
			/*double 	rv = ((double)rand()) / (RAND_MAX + 0.000001);
			idx = 0;

			while (1) {
				if (rv > total_lh_kappa0[idx])
				{
					rv -= total_lh_kappa0[idx];
					idx++;
				}
				else
					break;
			}*/

			idx = sampleFromLog(total_lh_kappa0);
			hypersampled_kappa0 = (idx)*0.025 + k0_min;
			//cout << "hypersampled kappa0 : " << hypersampled_kappa0 << '\n';


			/****************************************************   Hypersampling m   ***********************************************************/
				// Using this hypersampled kappa0 now lets sample m:

			example.kappa0 = hypersampled_kappa0;
			//Matrix	Psi = eye(d);
			//example.psi = eye(d)*(m - d - 1);


			//example.psi = hypersampled_psi;
			index = 0;
			for (int s = m_min; s < m_max; s += (2*dd)) {
				example.mm = s;
				total_lh = 0;
				for (int i = 0; i < N; i++)
				{
					total_lh += example.log_predictive_likelihood(example.assignment[i], i);
				}
				total_lh_m[index] = total_lh/N ;
				index++;
			}

			//for (int i = 0; i < total_lh_m.size(); i++) {
			//	//cout << "scores of m: " << total_lh_m[i] << '\n';
			//	total_lh_m[i] = exp(total_lh_m[i]);
			//	//cout << "scores of exp(m): " << total_lh_m[i] << '\n';
			//}

			//total_sum = 0;
			//for (int i = 0; i < total_lh_m.size(); i++) {
			//	total_sum += total_lh_m[i];
			//}

			//for (int i = 0; i < total_lh_m.size(); i++) {
			//	total_lh_m[i] /= total_sum;
			//}

			//// Sampling the m from  scores.
			//double 	rv_m = ((double)rand()) / (RAND_MAX + 0.000001);
			//idx = 0;
			//cout << "size tm: " << total_lh_m.size() << '\n';
			//while (1) {
			//	if (rv_m > total_lh_m[idx])
			//	{
			//		rv_m -= total_lh_m[idx];
			//		idx++;
			//	}
			//	else
			//		break;
			//}

			//for (m = d + 2; m < (d + 2 + 100 * d); m += (2 * d)) {
			//	total_lh_m.pop_back();
			//} 

			idx = sampleFromLog(total_lh_m);
			//cout << "idx: " << idx << '\n';
			hypersampled_m = m_min + idx*2*dd;
			//cout << "hypersampled m : " << hypersampled_m << '\n';

			/*for (int i = 0; i < total_lh_m.size(); i++) {
				cout << "scores of m: " << total_lh_m[i] << '\n';
			}*/

			/************************************************   Hypersampling  Psi   ************************************************************/

			////Matrix   total_lh_psimatrix;
			//example.mm = hypersampled_m;
			//////  example.psi = eye(d)*(m - d - 1);
			////for (int dim = 0; dim < dd; dim++) {
			////	index = 0;
			////	//Vector				total_lh_psi(int((diagval_max - diagval_min)/0.25));
			////	for (double diag_val = diagval_min; diag_val < diagval_max; diag_val += 0.01) {
			////		total_lh = 0;
			////		example.psi[dim][dim] = (example.mm - dd - 1)*diag_val;
			////		for (int i = 0; i < N; i++)
			////		{
			////			total_lh += example.log_predictive_likelihood(example.assignment[i], i) / N;
			////		}
			////		//total_lh_psi[index] = total_lh;
			////		total_lh_psimatrix[dim][index] = total_lh;
			////		index++;
			////	}
			////}



			////Vector		total_lh_psi = zeros(int((diagval_max - diagval_min) / 0.01) + 1, 1);
			////Vector		idx_psi = zeros(dd, 1);
			////
			////for (int i = 0; i < dd; i++) {
			////	index = 0;
			////	for (double j = diagval_min; j < diagval_max; j += 0.1) {
			////		total_lh_psi[index] = total_lh_psimatrix[i][index];
			////		index++;
			////	}
			////	idx_psi[i] = sampleFromLog(total_lh_psi);
			////}


			example.mm = hypersampled_m;
			
			//example.psi /= (hypersampled_m - dd - 1);
			for (int dim = 0; dim < dd; dim++) {
				index = 0;
				//Vector				total_lh_psi(int((diagval_max - diagval_min)/0.25));
				for (double diag_val = diagval_min; diag_val < diagval_max; diag_val += 0.025) {
					total_lh = 0;
					example.psi[dim][dim] = (example.mm - dd - 1)*diag_val;    
					for (int i = 0; i < N; i++)
					{
						total_lh += example.log_predictive_likelihood(example.assignment[i], i) ;
					}
					total_lh_psi[index] = total_lh/N;
					index++;
				}
				idx = sampleFromLog(total_lh_psi);
				hypersampled_psi[dim][dim] = (example.mm - dd - 1)*(idx*0.025 + diagval_min);
				example.psi[dim][dim] = hypersampled_psi[dim][dim];
			}



			//for (int i = 0; i < dd; i++) {
			//	//hypersampled_psi[i][i] = (idx_psi[i] * 0.1 + diagval_min)*(example.mm - dd - 1);
			//	cout << "hypersampled psi : " << hypersampled_psi[i][i] << '\n';
			//}


			/***************************************************   Hypersampling   alpha   ************************************************************/


			example.psi = hypersampled_psi;
			index = 0;
			for (double a = alpha_min; a < alpha_max; a += 0.1) {
				example.alpha = a;
				total_lh = 0;

				total_lh = log(example.alpha)*example.number_of_clusters;
				/*for (int i = 0; i < example.number_of_clusters; i++) {
					total_lh += gamln(example.suffstats_number_of_points[i]);
				}*/

				for (int j = 1; j < N; j++) {
					total_lh -= log(j - 1 + example.alpha);
				}
				//cout << "total_lh: " << total_lh << '\n';
				total_lh_alpha[index] = total_lh;
				index++;
			}

			//for (int i = 0; i < total_lh_alpha.n; i++) {
			//	total_lh_alpha[i] = exp(total_lh_alpha[i]);

			//}

			//total_sum = 0;

			//for (int i = 0; i < total_lh_alpha.size(); i++) {
			//	total_sum += total_lh_alpha[i];
			//}

			//for (int i = 0; i < total_lh_alpha.size(); i++) {
			//	total_lh_alpha[i] /= total_sum;

			//}



			////cout << "hey" << '\n';
			//// Sampling alpha from  scores.
			//double 	rv_a = ((double)rand()) / (RAND_MAX + 0.000001);
			//idx = 0;

			//while (1) {
			//	if (rv_a > total_lh_alpha[idx])
			//	{
			//		rv_a -= total_lh_alpha[idx];
			//		idx++;
			//	}
			//	else
			//		break;
			//}

			idx = sampleFromLog(total_lh_alpha);
			hypersampled_alpha = idx*0.1 + alpha_min;
			//cout << "hypersampled alpha : " << hypersampled_alpha << '\n';

			example.alpha = hypersampled_alpha;


/*			for (int i = 0; i < example.number_of_clusters; i++) {
				cout << "Table " << i << " : " << example.suffstats_number_of_points[i] << '\n';
			}
			cout << '\n'*/

		}













		//cout << "Are you trespassing??";

		/***************   Calculating     evaluating scores     ****************/
		
		Vector		true_classes;
		double		maxx = 0;
		int			max_id = 0;
		

		true_classes = zeros(true_labels.unique().n, 1);
		// To calculate the number of unique true labels
		for (int i = 0; i < N; i++) {
			true_classes[int(true_labels[i]) - 1] += 1;
		}

		Vector		precision(example.number_of_clusters);
		Vector		recall(example.number_of_clusters);
		Vector		f1score(example.number_of_clusters);
		Matrix		contingency_table = zeros(example.number_of_clusters, true_classes.n);

		for (int i = 0; i < N; i++) {
			contingency_table[example.assignment[i]][int(true_labels[i]) - 1] += 1.0;
		}

		for (int i = 0; i < example.number_of_clusters; i++) {

			maxx = -1;
			max_id = 0;
			for (int t = 0; t < true_classes.n; t++) {
				if (contingency_table[i][t] >= maxx) {
					maxx = contingency_table[i][t];
					max_id = t;
				}
			}

			precision[i] = maxx / example.suffstats_number_of_points[i];
			recall[i] = maxx / true_classes[max_id];
			f1score[i] = 2.0*precision[i] * recall[i] / (recall[i] + precision[i]);
		}

		f1_scores[fold] = f1score.mean();
		cout << "f1score:" << f1score.mean() << '\n';
		
	}
	for (int l = 0; l < 10; l++) {
		cout << "10-fold f1-score :" << f1_scores[l] << '\n';
	}
	cout << '\a';
	cout << "10-fold f1-score mean:" << f1_scores.mean() << '\n';
	Vector   st(10);
	st = (f1_scores - ones(10)*f1_scores.mean())/sqrt(10);

	
	cout << "10-fold f1-score std:" << st.norm() << '\n';

	system("pause");
 	return 0;
}
