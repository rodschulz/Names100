//
// Created by fran on 5/20/15.
//

#include "Svm.h"

#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

Svm::Svm()
{
}

Svm::~Svm()
{
}

// Set up training data
void Svm::setUpTrainData(const vector<Mat> &_bows, vector<Mat> &_toTrain)
{

	int totalImages = 0;
	int totalFeatures = _bows[0].cols;
	int nClasses = _bows.size();

	for (int k = 0; k < nClasses; k++)
	{
		totalImages += _bows[k].rows;
	}

	int labels[435] =
	{ }; //TODO ojo con esto
	Mat trainingData(totalImages, totalFeatures, CV_32FC1);

	int t = 0;
	for (int i = 0; i < nClasses; i++)
	{
		Mat currClass = _bows[i];
		int rows = currClass.rows;
		for (int j = 0; j < rows; j++)
		{
			//labels.push_back(i); // Add the "class label", same for all the class data
			labels[t] = i;
			currClass.row(j).copyTo(trainingData.row(t));
			t++;
		}
	}

	Mat labelsMat(435, 1, CV_32SC1, labels);

	vector<Mat> dataToTrain;
	_toTrain.push_back(trainingData);
	_toTrain.push_back(labelsMat);
}

// Load SVM parameters
void Svm::loadSVMParams(CvSVMParams &_params, const string &_filename)
{
	string line;
	ifstream inputFile;
	inputFile.open(_filename.c_str(), fstream::in);
	if (inputFile.is_open())
	{
		while (getline(inputFile, line))
		{
			vector<string> tokens;
			istringstream iss(line);
			copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(tokens));

			if (tokens[0].compare("C") == 0)
				_params.svm_type = atoi(tokens[1].c_str());
			else if (tokens[0].compare("gamma") == 0)
				_params.kernel_type = atoi(tokens[1].c_str());
		}
		inputFile.close();
	}
	else
		cout << "Unable to open SVM params input: " << _filename;
}
