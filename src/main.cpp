#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "Helper.h"
#include "Codebook.h"
#include "Config.h"
#include "Svm.h"

using namespace std;
using namespace std::chrono;
using namespace cv;

void getCodebooks(const string &_inputFolder, const vector<string> &_classNames, vector<Codebook> &_codebooks)
{
	double kmeansStop = Config::getKMeansThreshold();
	int kmeansMaxIter = Config::getKMeansMaxIterations();

	for (string className : _classNames)
	{
		if (!Codebook::loadCodebook(_inputFolder + className + "/sample/", Config::getCacheLocation(), _codebooks))
		{
			cout << "Codebook for class '" << className << "' not found in cache.\n\tCalculating new codebook\n";
			_codebooks.push_back(Codebook(Config::getCodebookSize(), Config::useDenseSampling(), Config::getGridSize()));
			_codebooks.back().calculateCodebook(_inputFolder + className + "/sample/", kmeansMaxIter, kmeansStop);
			cout << "\tSaving codebook for class '" << className << "' to cache file\n";
			_codebooks.back().saveToFile(Config::getCacheLocation());
		}
		else
			cout << "Codebook for class '" << className << "' read from cache\n";
	}
}

void calculateBoWs(const string &_inputFolder, const vector<string> &_classNames, const string &_set, vector<Codebook> &_codebooks, vector<Mat> &_BoWs)
{
	BoWType bowType = Config::getBoWType();
	int totalBins = Helper::getSqrSum(Helper::generateLevels(Config::getPyramidLevelNumber()));

	// Create the set of BoWs
	_BoWs.resize(_classNames.size());

	// Iterate over each class
#pragma omp parallel for
	for (size_t i = 0; i < _classNames.size(); i++)
	{
		string className = _classNames[i];
		cout << "Calculating BoW for class " << className << " using set " << _set << endl;

		int bowCols = bowType == LLC_SPM ? totalBins * _codebooks[i].getClusterNumber() : _codebooks[i].getClusterNumber();

		// Get the list of images to process
		vector<string> imageList;
		Helper::getContentsList(_inputFolder + className + "/" + className + "_" + _set + "/", imageList);

		// Create the matrix to hold the bows
		Mat currentClassBoW = Mat::zeros(imageList.size(), bowCols, CV_32FC1);
		_BoWs[i] = currentClassBoW;

		int j = 0;
		Mat documentCounter = Mat::zeros(1, _codebooks[i].getClusterNumber(), CV_32FC1);

		for (string imageLocation : imageList)
		{
			int imgWidth, imgHeight;
			Mat descriptors;
			vector<KeyPoint> keypoints;

			// Get descriptors
			Helper::calculateImageDescriptors(imageLocation, descriptors, keypoints, imgWidth, imgHeight, _codebooks[i].usingDenseSampling(), _codebooks[i].getGridSize());

			// Calculate BoW
			Mat row = currentClassBoW.row(j++);
			if (bowType == FREQUENCIES)
				_codebooks[i].getBoWTF(descriptors, row);
			else
				_codebooks[i].calculateLLC(descriptors, keypoints, Config::getLLCNeighbors(), row, imgWidth, imgHeight, Config::getPyramidLevelNumber());

			if (bowType == FREQUENCIES)
			{
				cout << "\tCalculating frequencies\n";
				for (int k = 0; k < currentClassBoW.cols; k++)
					documentCounter.at<float>(0, k) += (row.at<float>(0, k) > 0 ? 1 : 0);
			}
		}

		// Calculate TF-IDF if it is the right bow
		if (bowType == FREQUENCIES && Config::calculateTFIDF())
		{
			cout << "\tCalculating tf-idf\n";

			// Calculate tf-idf logarithmic factor and then the tf-idf itself
			for (int k = 0; k < documentCounter.cols; k++)
				documentCounter.at<float>(0, k) = log((float) imageList.size() / documentCounter.at<float>(0, k));

			for (int p = 0; p < currentClassBoW.rows; p++)
			{
				for (int q = 0; q < currentClassBoW.cols; q++)
					currentClassBoW.at<float>(p, q) *= documentCounter.at<float>(0, q);
			}
		}
	}
}

void writeResults(const vector<string> &_classNames, const vector<vector<float>> &_scores, vector<int> &_correctGuesses, const int _totalGuesses)
{
	// Create results folder
	string cmd = "mkdir -p ../results/";
	if (system(cmd.c_str()) != 0)
		cout << "WARNING: can't create results folder\n";

	FILE *resultsFile;
	string outputfilename = "../results/results";
	resultsFile = fopen(outputfilename.c_str(), "w");

	fprintf(resultsFile, "# Name\tRightGuesses\n");
	fprintf(resultsFile, "# Total Guesses = %d\n", _totalGuesses);

	int totalSize = _scores.size();
	for (int k = 0; k < totalSize; k++)
		fprintf(resultsFile, "%s \t %d \t %.2f\n", _classNames[k].c_str(), _correctGuesses[k], ((float) _correctGuesses[k]) / ((float) _totalGuesses));
	fclose(resultsFile);
}

int main(int _nargs, char ** _vargs)
{
	if (_nargs < 2)
	{
		cout << "Not enough arguments\n";
		return EXIT_FAILURE;
	}

	string inputFolder = _vargs[1];
	cout << "Input folder: " << inputFolder << endl;

	Config::load("./config/config");

	// Create a new image sample
	if (Config::createImageSample())
		Helper::createImageSamples(inputFolder, Config::getSampleSize());

	// Get class names
	vector<string> classNames;
	Helper::getClassNames(inputFolder, classNames);

	// Generate/load the codebook for each class
	vector<Codebook> codebooks;
	getCodebooks(inputFolder, classNames, codebooks);

	// Calculate the BoW for each image in each set
	vector<Mat> trainBoWs, validationBoWs, testBoWs;
	calculateBoWs(inputFolder, classNames, "train", codebooks, trainBoWs);
	calculateBoWs(inputFolder, classNames, "val", codebooks, validationBoWs);
	calculateBoWs(inputFolder, classNames, "test", codebooks, testBoWs);

	int totalGuesses = 0;
	int totalNames = trainBoWs.size();
	vector<vector<float>> allScores;
	vector<int> rightGuesses = vector<int>(totalNames, 0);

	// 1 vs. 1 SVM Classification
	for (int n = 0; n < totalNames; n++)
	{
		// n == current name
		vector<float> nScores;
		for (int m = 0; m < totalNames; m++)
		{
			// m == name for pair, comparison is n vs. m
			if (m != n)
			{
				cout << "Comparing: " << classNames[n] << " vs. " << classNames[m] << endl;

				// Training for n vs. m comparison
				vector<Mat> tempTrain;
				tempTrain.push_back(trainBoWs[n]);
				tempTrain.push_back(trainBoWs[m]);

				Mat trainnm;
				Helper::concatMats(tempTrain, trainnm);

				const int totalLabels = trainBoWs[n].rows + trainBoWs[m].rows;
				int *trainLabels;
				trainLabels = (int *) malloc(totalLabels * sizeof(int));

				for (int k = 0; k < trainBoWs[n].rows; k++)
					trainLabels[k] = 1;
				for (int k = trainBoWs[n].rows; k < totalLabels; k++)
					trainLabels[k] = -1;

				Mat labelsMat = Mat(totalLabels, 1, CV_32SC1, trainLabels);

				// Model SVM Parameters
				CvSVMParams params;
				params.svm_type = CvSVM::C_SVC;
				params.kernel_type = CvSVM::LINEAR;

				// Finishing criteria
				params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-6);
				params.C = 10000;

				// Model SVM
				CvSVM SVM;
				cout << "Training SVM model . . ." << endl;

				// Training!
				SVM.train(trainnm, labelsMat, Mat(), Mat(), params);

				//This changes for validation/test runs
				vector<Mat> tempVal;
				tempVal.push_back(validationBoWs[n]);
				tempVal.push_back(validationBoWs[m]);

				Mat validationSet;
				Helper::concatMats(tempVal, validationSet);

				cout << "Running SVM . . . Comparing " << validationSet.rows << " images." << endl;

				for (int k = 0; k < validationSet.rows; k++)
				{
					float res = SVM.predict(validationSet.row(k));
					nScores.push_back(res);
					if (k < validationBoWs[n].rows && res == 1)
						rightGuesses[n]++;
					else if (k >= validationBoWs[n].rows && res == -1)
						rightGuesses[m]++;
					totalGuesses++;
				}

				free(trainLabels);
				cout << "Done." << endl;

			}
			else
				nScores.push_back(NAN); //NAN result for comparison between same names
		}

		allScores.push_back(nScores);
	}

	// Write results
	writeResults(classNames, allScores, rightGuesses, totalGuesses);

	cout << "Finished\n";
	return EXIT_SUCCESS;
}
