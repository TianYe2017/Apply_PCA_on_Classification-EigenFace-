#pragma once
#ifndef FUNCTION_H
#define FUNCTION_H

#include <opencv.hpp>
#include <highgui.hpp>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

struct package 
{
	vector<vector<float>> data;
	vector<int> label;
};

struct cell 
{
	vector<float> feature;
	int label;
};


package CreateData(bool blur, int N, int low, int high, bool debug);
vector<float> ComputeAverageFace(package pack, bool debug);
cv::Mat GenerateResidualImgs(package train_dat, vector<float> avg, bool debug);
vector<cell> CvtDataSet2Cell(cv::Mat eigenvalues, cv::Mat eigenvectors, cv::Mat cluster, vector<int>label, bool debug);
void GatherTestAccuracy(vector<cell> ref, vector<cell> unknown, int flag);






#endif // !FUNCTION_H
