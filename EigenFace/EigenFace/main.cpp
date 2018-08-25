#include "iostream"
#include "function.h"

using namespace std;

void main(void) 
{
	cout << "read train data...\n\n" << endl;
	package trainData = CreateData(true, 40, 1, 7, false);
	cout << "read test data...\n\n" << endl;
	package testData = CreateData(true, 40, 8, 10, false);
	cout << "compute avg face..." << endl;
	vector<float> avgFace =  ComputeAverageFace(trainData, false);
	cout << "compute residual faces..." << endl;
	cv::Mat clusterTrain = GenerateResidualImgs(trainData, avgFace, false);
	cv::Mat clusterTest = GenerateResidualImgs(testData, avgFace, false);
	cout << "PCA..." << endl;
	cv::PCA pca(clusterTrain, cv::Mat(), CV_PCA_DATA_AS_ROW, 40);
	cv::Mat eigenValues = pca.eigenvalues.clone();
	cv::Mat eigenVectors = pca.eigenvectors.clone();
	cout << "eigenValues: " << eigenValues.rows << " " << eigenValues.cols << endl;
	for (int i = 0; i < eigenValues.rows; i++)
	{
		cout << eigenValues.at<float>(i, 0) << endl;
	}
	/*for (int m = 0; m < 40; m++) 
	{
		for (int i = 5000; i < 5030; i++) 
		{
			cout << eigenVectors.at<float>(m, i) << " "; 
		}
	}
	cout << "" <<endl;*/
	cout << "eigenVectors: " << eigenVectors.rows << " " << eigenVectors.cols << endl;
	cout << "convert images to feature vectors..." << endl;
	vector<cell> convertedTrainData = CvtDataSet2Cell(eigenValues, eigenVectors, clusterTrain, trainData.label, false);
	vector<cell> convertedTestData = CvtDataSet2Cell(eigenValues, eigenVectors, clusterTest, testData.label, false);
	//cout << convertedTrainData.size() << " " << convertedTestData.size()<< endl;
	cout << "Matching..." << endl;
	GatherTestAccuracy(convertedTrainData, convertedTestData, 0);

	while (1) 
	{
		cv::waitKey(100);
	};
	/*srand((unsigned)time(NULL));

	cv::Mat data = cv::Mat(cv::Size(30,5),CV_32F);
	for (int i = 0; i < 5; i++) 
	{
		for (int j = 0; j < 30; j++)
		{
			data.at<float>(i, j) = (float)(rand()%(100-1+1)+1);
		}
	}
	
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 30; j++)
		{
			cout << data.at<float>(i, j) << " ";
		}
		cout << "" <<endl;
	}
	cout << "" << endl;

	
	
*/

}