#include "function.h"


void ProcessSingleImage(string path,int label,bool blur, package& pack, bool debug)
{
	cv::Mat img = cv::imread(path);
	cv::cvtColor(img,img,CV_RGB2GRAY);
	if (blur) 
	{
		cv::blur(img,img,cv::Size(3,3));
	}
	img.convertTo(img, CV_32F);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<float>(i, j) /= 255.0f;
		}
	}
	vector<float> ans;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			ans.push_back(img.at<float>(i, j));
		}
	}
	pack.data.push_back(ans);
	pack.label.push_back(label);
	if (debug) 
	{
		cout << "path: " << path << endl;
		cout << "rows: " << img.rows << " " << "cols: " << img.cols << " " << "label: " << label << endl;
		cout << "" << endl;
	}
	
}

package CreateData(bool blur,int N, int low, int high, bool debug) 
{
	package pack;
	for (int s = 1; s <= N; s++) 
	{
		for (int i = low; i <= high; i++)
		{
			string path = "./att_faces/s";
			path += to_string(s);
			path += "/";
			path += to_string(i);
			path += ".pgm";
			ProcessSingleImage(path, s, blur, pack, debug);
		}
	}
	return pack;
}

vector<float> ComputeAverageFace(package pack, bool debug) 
{
	vector<float> ans;
	for (int i = 0; i < pack.data[0].size(); i++) 
	{
		float sum = 0.0f;
		for (int s = 0; s < pack.data.size(); s++) 
		{	
			sum += pack.data[s][i];
		}
		sum /= (float)pack.data.size();
		ans.push_back(sum);
	}
	if (debug) 
	{
		cv::Mat img = cv::Mat(cv::Size(92,112),CV_8U);
		for (int row = 0; row < 112; row++) 
		{
			for (int col = 0; col < 92; col++) 
			{
				img.at<unsigned char>(row, col) = (unsigned char)(255.0f * ans[92 * row + col]);
			} 
		}
		cv::imshow("window", img);
		cv::waitKey(50);
	}
	return ans;
}

cv::Mat GenerateResidualImgs(package train_dat, vector<float> avg, bool debug) 
{
	int M = train_dat.data.size();
	int N = train_dat.data[0].size();
	cv::Mat ans = cv::Mat(cv::Size(N, M), CV_32F);
	for (int m = 0; m < M; m++) 
	{
		for (int i = 0; i < N; i++) 
		{
			ans.at<float>(m, i) = train_dat.data[m][i] - avg[i];
		}
		if (debug)
		{
			cv::Mat tmp = cv::Mat(cv::Size(92, 112), CV_8U);
			for (int row = 0; row < 112; row++)
			{
				for (int col = 0; col < 92; col++)
				{
					tmp.at<unsigned char>(row, col) = (unsigned char)(ans.at<float>(m, 92 * row + col) * 255.0f);
				}
			}
			cv::imshow("win", tmp);
			cv::waitKey(30);
		}
	}
	return ans;
}

vector<float> GenerateFeatureVector(cv::Mat eigenvalues, cv::Mat eigenvectors, vector<float> img, bool debug)
{
	vector<float> ans;
	int M = eigenvectors.rows;
	int N = eigenvectors.cols;
	for (int m = 0; m < M; m++) 
	{
		float sum = 0;
		for (int i = 0; i < N; i++) 
		{
			sum += eigenvectors.at<float>(m, i) * img[i];
		}
		ans.push_back(sum);
	}
	if (debug) 
	{
		for (int i = 0; i < M; i++) 
		{
			cout << ans[i] << " ";
		}
		cout << "" << endl;
	}
	return ans;
}

vector<cell> CvtDataSet2Cell(cv::Mat eigenvalues, cv::Mat eigenvectors, cv::Mat cluster,vector<int>label, bool debug) 
{
	cell c;
	vector<float> img;
	vector<cell> ans;
	int M = cluster.rows;
	int N = cluster.cols;
	for (int m = 0; m < M; m++) 
	{
		img.clear();
		for (int i = 0; i < N; i++) 
		{
			img.push_back(cluster.at<float>(m, i));
		}
		c.feature = GenerateFeatureVector(eigenvalues,eigenvectors,img,debug);
		c.label = label[m];
		ans.push_back(c);
	}
	return ans;
}

float L2Distance(vector<float> a, vector<float>b) 
{
	float sum = 0;
	for (int i = 0; i < a.size(); i++) 
	{
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	}
	sum = sqrt(sum);
	return sum;
}


float L1Distance(vector<float> a, vector<float>b) 
{
	float sum = 0;
	for (int i = 0; i < a.size(); i++) 
	{
		sum += abs(a[i] - b[i]);
	}
	return sum;
}


void GatherTestAccuracy(vector<cell> ref, vector<cell> unknown, int flag) 
{
	float accuracy;
	float numOfPos = 0.0f;
	for (int i = 0; i < unknown.size(); i++) 
	{
		int real_label = unknown[i].label;
		int label = ref[0].label;
		float minDis;
		float dis;
		if (flag == 0) 
		{
			minDis = L2Distance(ref[0].feature, unknown[i].feature);
		}
		else 
		{
			minDis = L1Distance(ref[0].feature, unknown[i].feature);
		}
		for (int j = 1; j < ref.size(); j++) 
		{
			if (flag == 0) 
			{
				dis = L2Distance(ref[j].feature, unknown[i].feature);
			}
			else 
			{
				dis = L1Distance(ref[j].feature, unknown[i].feature);
			}
			if (dis < minDis) 
			{
				minDis = dis;
				label = ref[j].label;
			}
		}
		cout << "real label: " << real_label << "  " << "predicted label: " << label << "  " << "distance: " << minDis <<endl ;
		if (real_label == label)
		{
			numOfPos += 1.0f;
		}
	}
	accuracy = numOfPos / ((float)unknown.size());
	cout << "Total number of test cases: " << unknown.size() << endl;
	cout << "# correct: " << (int)numOfPos << "  " << "accuracy: " << accuracy << endl;
}


