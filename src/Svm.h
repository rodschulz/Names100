//
// Created by fran on 5/20/15.
//

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

class Svm
{
public:
	static void setUpTrainData(const vector<Mat> &_bows, vector<Mat> &_toTrain);
	static void loadSVMParams(CvSVMParams &_params, const string &_filename);

private:
	Svm();
	~Svm();
};
