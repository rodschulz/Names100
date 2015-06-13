#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "Helper.h"
#include "Codebook.h"
#include "Config.h"
#include "Svm.h"

using namespace std;
using namespace cv;

void calculateBoWs(const string &_inputFolder, const vector<string> &_classNames, const string &_set, vector<Codebook> &_codebooks, vector<Mat> &_BoWs)
{
	for (size_t i = 0; i < _classNames.size(); i++)
	{
		string className = _classNames[i];
		cout << "Calculating BoW for class " << className << " using set " << _set << "\n";

		vector<string> imageList;
		Helper::getContentsList(_inputFolder + className + "/" + className + "_" + _set + "/", imageList);
		Mat currentClassBoW = Mat::zeros(imageList.size(), _codebooks[i].getClusterNumber(), CV_32FC1);
		_BoWs.push_back(currentClassBoW);

		int j = 0;
		Mat documentCounter = Mat::zeros(1, _codebooks[i].getClusterNumber(), CV_32FC1);

		cout << "Calculating frequencies\n";
		for (string imageLocation : imageList)
		{
			Mat descriptors;
			Helper::calculateImageDescriptors(imageLocation, descriptors);
			Mat row = currentClassBoW.row(j++);
			_codebooks[i].getBoWTF(descriptors, row);

			for (int k = 0; k < currentClassBoW.cols; k++)
				documentCounter.at<float>(0, k) += (row.at<float>(0, k) > 0 ? 1 : 0);
		}

		cout << "Calculating tf-idf\n";
		// Calculate tf-idf logarithmic factor and then the tf-idf itself
		for (int k = 0; k < documentCounter.cols; k++)
			documentCounter.at<float>(0, k) = log((float) imageList.size() / documentCounter.at<float>(0, k));
		for (int p = 0; p < currentClassBoW.rows; p++)
		{
			for (int q = 0; q < currentClassBoW.cols; q++)
				currentClassBoW.at<float>(p, q) *= documentCounter.at<float>(0, q);
		}

		//Helper::printMatrix<float>(documentCounter, 3);
		//Helper::printMatrix<float>(currentClassBoW, 3);
	}
}

int main(int _nargs, char ** _vargs)
{
	if (_nargs < 2)
		cout << "Not enough arguments\n";

	string inputFolder = _vargs[1];
	cout << "Input folder: " << inputFolder << endl;
	Config::load("../config/config");

	// Create a new image sample
	if (Config::createImageSample())
	{
		cout << "Creating image sample\n";
		Helper::createImageSamples(inputFolder, Config::getSampleSize());
	}

	// Generate or load the codebooks
	vector<Codebook> codebooks;
	vector<string> classNames;
	Helper::getClassNames(inputFolder, classNames);
	for (string className : classNames)
	{
		if (!Codebook::loadCodebook(inputFolder + className + "/sample/", codebooks))
		{
			cout << "Codebook for class '" << className << "' not found in cache. Calculating new codebook\n";
			codebooks.push_back(Codebook(Config::getCodebookClustersNumber()));
			codebooks.back().calculateCodebook(inputFolder + className + "/sample/", 10000, 0.1);
			cout << "Saving codebook for class '" << className << "' to cache file\n";
			codebooks.back().saveToFile("../cache/");
		}
		else
			cout << "Codebook for class '" << className << "' read from cache\n";
	}

	// Calculate the BoW for each image in each set
	vector<Mat> trainBoWs, validationBoWs, testBoWs;
	calculateBoWs(inputFolder, classNames, "train", codebooks, trainBoWs);
	calculateBoWs(inputFolder, classNames, "val", codebooks, validationBoWs);
	calculateBoWs(inputFolder, classNames, "test", codebooks, testBoWs);

	//Preparing labels for training Set... there must be a better way to do this but I'm having problems with arrays
	int t = 0;
	int nClasses = trainBoWs.size();
	vector<int> trainLabels;
	for(int i = 0; i < nClasses ; i++){
		Mat currClass = trainBoWs[i];
		int rows = currClass.rows;
		for(int j = 0; j < rows; j++){
			trainLabels.push_back(i);
			t++;
		}
	}

	// Classification part
//	vector<Mat> dataToTrain;
//	Svm::setUpTrainData(trainBoWs, dataToTrain, trainLabels.size());
//
//	// Model SVM Parameters
//	CvSVMParams params;
//	params.svm_type = CvSVM::C_SVC;
//	params.kernel_type = CvSVM::RBF;
//	// Finishing criteria
//	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-6);
//	//svm::loadSVMParams(params, "../config/svmparams");
//	params.C = 100000;
//	params.gamma = 3;
//
//	// Model SVM
//	CvSVM SVM;
//	cout << "Training SVM model . . ." << endl;
//
//	// Training!
//	SVM.train(dataToTrain[0], dataToTrain[1], Mat(), Mat(), params);
//
//	Mat validationSet;
//	Helper::concatMats(validationBoWs, validationSet);
//
//	Mat testSet;
//	Helper::concatMats(testBoWs, testSet);
//
//	int class0, class1, class2;
//	class0 = 0;
//	class1 = 0;
//	class2 = 0;


	// UNCOMMENT THIS FOR VALIDATION RUNS
	// TEST RUNS CODE BELOW

	/*ofstream myfile;
	string outputfilename = "../validation/Cparam-" + to_string(params.C);

	if(!Helper::fileExists(outputfilename.c_str())) {
		myfile.open(outputfilename, ios::out | ios::app);
		myfile << "# " << validationSet.rows << " images in validation set" << endl;
		myfile << "# VALIDATION SET:" << endl;
		myfile << "# Class 0: " << validationBoWs[0].rows << " | Class 1: " << validationBoWs[1].rows << " | Class 2: " << validationBoWs[2].rows << endl;
		myfile << "# SampleSize \t NClusters \t Class0 \t Class1 \t Class2" << endl;
		myfile << "# CLASSIFICATION RESULTS: (first row is actual set)" << endl;
		myfile << "0" << "\t" << "0" << "\t" << validationBoWs[0].rows << "\t" << validationBoWs[1].rows << "\t" << validationBoWs[2].rows << endl;
		myfile.close();
	}

	//Preparing labels for validation Set... there must be a better way to do this but I'm having problems with arrays
	t = 0;
	nClasses = validationBoWs.size();
	vector<int> validationLabels;
	for(int i = 0; i < nClasses ; i++){
		Mat currClass = validationBoWs[i];
		int rows = currClass.rows;
		for(int j = 0; j < rows; j++){
			validationLabels.push_back(i);
			t++;
		}
	}

	cout << "Running classification for validation set" << endl;
	//int confusion[4*3]; // TP,FP,TN,FP for each class
	int tp0 = 0;
	int fp0 = 0;
	int tn0 = 0;
	int fn0 = 0;
	int tp1 = 0;
	int fp1 = 0;
	int tn1 = 0;
	int fn1 = 0;
	int tp2 = 0;
	int fp2 = 0;
	int tn2 = 0;
	int fn2 = 0;

	for (int k = 0; k < validationSet.rows; k++)
	{
		float res = SVM.predict(validationSet.row(k));
		int realClass = validationLabels[k];
		if (res == 0)
		{
			class0++;
			switch(realClass) {
				case 0:
					tp0++;
					tn1++;
					tn2++;
				case 1:
					fp0++;
					fn1++;
					tn2++;
				case 2:
					fp0++;
					tn1++;
					fn2++;
			}
		}
		else if (res == 1)
		{
			class1++;
			switch(realClass){
				case 0:
					fn0++;
					fp1++;
					tn2++;
				case 1:
					tn0++;
					tp1++;
					tn2++;
				case 2:
					tn0++;
					fp1++;
					fn2++;
			}
		}
		else
		{
			class2++;
			switch(realClass){
				case 0:
					fn0++;
					tn1++;
					fp2++;
				case 1:
					tn0++;
					fn1++;
					fp2++;
				case 2:
					tn0++;
					tn1++;
					tp2++;
			}
		}

	}

	myfile.open(outputfilename, ios::out | ios::app);
	myfile << to_string(Config::getSampleSize()) << "\t" << to_string(Config::getCodebookClustersNumber()) << "\t" << class0 << "\t" << class1 << "\t" << class2 << endl;
	myfile.close();*/

	// UNCOMMENT THIS FOR TEST RUNS
//	ofstream myfile;
//	string outputfilename = "../results/Cparam" + to_string(params.C) + "-NClusters" + to_string(Config::getCodebookClustersNumber()) + "-SampleSize" + to_string(Config::getSampleSize()) + "-gamma" + to_string(params.gamma);
//
//	if(!Helper::fileExists(outputfilename.c_str())) {
//		myfile.open(outputfilename, ios::out | ios::app);
//		myfile << "# " << testSet.rows << " images in test set" << endl;
//		myfile << "# TEST SET:" << endl;
//		myfile << "# Class 0: " << testBoWs[0].rows << " | Class 1: " << testBoWs[1].rows << " | Class 2: " << testBoWs[2].rows << endl;
//		myfile << "# CLASSIFICATION RESULTS: (first row is actual set)" << endl;
//		myfile << "# Class \t RealValue \t SVM \t Precision \t Recall \t Accuracy"<< endl;
//		myfile.close();
//	}
//
//	//Preparing labels for test Set... there must be a better way to do this but I'm having problems with arrays
//	t = 0;
//	nClasses = testBoWs.size();
//	vector<int> testLabels;
//	for(int i = 0; i < nClasses ; i++){
//		Mat currClass = testBoWs[i];
//		int rows = currClass.rows;
//		for(int j = 0; j < rows; j++){
//			testLabels.push_back(i);
//			t++;
//		}
//	}
//
//	cout << "Running classification for test set" << endl;
//	//int confusion[4*3]; // TP,FP,TN,FP for each class
//	int class0mat[3] = {0, 0, 0}; //0 as 0, 0 as 1, 0 as 2
//	int class1mat[3] = {0, 0, 0}; //1 as 0, 1 as 1, 1 as 2
//	int class2mat[3] = {0, 0, 0}; //2 as 0, 2 as 1, 2 as 2
//	int tp0 = 0;
//	int fp0 = 0;
//	int tn0 = 0;
//	int fn0 = 0;
//	int tp1 = 0;
//	int fp1 = 0;
//	int tn1 = 0;
//	int fn1 = 0;
//	int tp2 = 0;
//	int fp2 = 0;
//	int tn2 = 0;
//	int fn2 = 0;
//
//	for (int k = 0; k < testSet.rows; k++)
//	{
//		float res = SVM.predict(testSet.row(k));
//		int realClass = testLabels[k];
//		if (res == 0) {
//			class0++;
//			if (realClass == 0) {
//				tp0++;
//				tn1++;
//				tn2++;
//				class0mat[0]++;
//			}else if (realClass == 1) {
//				fp0++;
//				fn1++;
//				tn2++;
//				class1mat[0]++;
//			}else if (realClass == 2){
//					fp0++;
//					tn1++;
//					fn2++;
//					class2mat[0]++;
//			}
//		}
//		else if (res == 1) {
//			class1++;
//			if(realClass == 0) {
//				fn0++;
//				fp1++;
//				tn2++;
//				class0mat[1]++;
//			}else if (realClass == 1) {
//				tn0++;
//				tp1++;
//				tn2++;
//				class1mat[1]++;
//			}else if (realClass == 2){
//					tn0++;
//					fp1++;
//					fn2++;
//					class2mat[1]++;
//			}
//		}
//		else {
//			class2++;
//			if (realClass == 0) {
//				fn0++;
//				tn1++;
//				fp2++;
//				class0mat[2]++;
//			}else if (realClass == 1) {
//				tn0++;
//				fn1++;
//				fp2++;
//				class1mat[2]++;
//			}else if (realClass == 2){
//					tn0++;
//					tn1++;
//					tp2++;
//					class2mat[2]++;
//			}
//		}
//	}
//
//	float precision0 = (float)tp0 / (tp0 + fp0);
//	float precision1 = (float)tp1 / (tp1 + fp1);
//	float precision2 = (float)tp2 / (tp2 + fp2);
//
//	float recall0 = (float)tp0 / (tp0 + fn0);
//	float recall1 = (float)tp1 / (tp1 + fn1);
//	float recall2 = (float)tp2 / (tp2 + fn2);
//
//	float accu0 = (float)(tp0 + tn0) / (tp0 + tn0 + fp0 + fn0);
//	float accu1 = (float)(tp1 + tn1) / (tp1 + tn1 + fp1 + fn1);
//	float accu2 = (float)(tp2 + tn2) / (tp2 + tn2 + fp2 + fn2);
//
//	myfile.open(outputfilename, ios::out | ios::app);
//	myfile << classNames[0] << "\t" << testBoWs[0].rows << "\t" << class0 << "\t" << precision0 << "\t" << recall0 << "\t" << accu0 << endl;
//	myfile << classNames[1] << "\t" << testBoWs[1].rows << "\t" << class1 << "\t" << precision1 << "\t" << recall1 << "\t" << accu1 << endl;
//	myfile << classNames[2] << "\t" << testBoWs[2].rows << "\t" << class2 << "\t" << precision2 << "\t" << recall2 << "\t" << accu2 << endl;
//	myfile.close();
//
//	ofstream confMat;
//	string confMatName = "../results/confusionMatrix";
//
//	confMat.open(confMatName, ios::out | ios::app);
//	confMat << "\t\tReal" << endl;
//	confMat << "\tClass0\tClass1\tClass2" << endl;
//	confMat << "Pred Class 0\t" << class0mat[0] << "\t" << class1mat[0] << "\t" << class2mat[0] << endl;
//	confMat << "Pred Class 1\t" << class0mat[1] << "\t" << class1mat[1] << "\t" << class2mat[1] << endl;
//	confMat << "Pred Class 2\t" << class0mat[2] << "\t" << class1mat[2] << "\t" << class2mat[2] << endl;
//	confMat.close();

	cout << "Finished\n";
	return EXIT_SUCCESS;
}
