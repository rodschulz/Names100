/**
 * Author: rodrigo
 * 2015
 */
#include "Helper.h"
#include <stdlib.h>
#include <dirent.h>
#include <cstring>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;

Helper::Helper()
{
}

Helper::~Helper()
{
}

int Helper::getRandomNumber(const int _min, const int _max)
{
	srand(rand());
	srand(rand());
	int number = _min + (rand() % (int) (_max - _min + 1));
	return number;
}

void Helper::getContentsList(const string &_folder, vector<string> &_contents, const bool _appendToList)
{
	DIR *folder;
	struct dirent *epdf;

	if (!_appendToList)
		_contents.clear();

	if ((folder = opendir(_folder.c_str())) != NULL)
	{
		while ((epdf = readdir(folder)) != NULL)
		{
			if (strcmp(epdf->d_name, ".") == 0 || strcmp(epdf->d_name, "..") == 0)
				continue;

			_contents.push_back(_folder + epdf->d_name);
		}
		closedir(folder);
	}
	else
	{
		cout << "ERROR: can't open folder " << _folder << "\n";
	}

	std::sort(_contents.begin(), _contents.end());
}

void Helper::createImageSamples(const string &_inputFolder, const double _sampleSize, const long _seed)
{
	cout << "Creating image sample\n";

	vector<string> folderList;
	Helper::getContentsList(_inputFolder, folderList);

	if (_seed != -1)
		srand(_seed);
	else
	{
		srand(time(NULL));
		srand(rand());
		srand(rand());
	}

	double percent = (double) folderList.size() / 100.0;
	double total = 0.0;

	string cmd;
	for (string classFolder : folderList)
	{
		cout << "\t" << (total += percent) << "% done" << endl;

		string className = classFolder.substr(classFolder.find_last_of('/'));

		vector<string> classContents;
		Helper::getContentsList(classFolder + className + "_train/", classContents, true);
		Helper::getContentsList(classFolder + className + "_val/", classContents, true);

		string sampleFolder = classFolder + "/sample/";
		cmd = "rm -rf " + sampleFolder;
		system(cmd.c_str());
		cmd = "mkdir " + sampleFolder;
		system(cmd.c_str());

		vector<string> classSample;
		int sampleSize = classContents.size() * _sampleSize;
		for (int k = 0; k < sampleSize; k++)
		{
			int sampleIndex = (rand() % (int) classContents.size());
			string origin = *(classContents.begin() + sampleIndex);
			string destination = sampleFolder + origin.substr(origin.find_last_of('/') + 1);
			cmd = "cp " + origin + " " + destination;
			system(cmd.c_str());
			classContents.erase(classContents.begin() + sampleIndex);
		}
	}
}

void Helper::getClassNames(const string &_inputFolder, vector<string> &_classNames)
{
	_classNames.clear();
	Helper::getContentsList(_inputFolder, _classNames);
	for (size_t i = 0; i < _classNames.size(); i++)
		_classNames[i] = _classNames[i].substr(_classNames[i].find_last_of('/') + 1);
}

bool Helper::calculateImageDescriptors(const string &_imageLocation, Mat &_descriptors, vector<KeyPoint> &_keypoints, const bool _denseSaming, const int _gridSize)
{
	bool statusOk = true;
	Mat image = imread(_imageLocation, CV_LOAD_IMAGE_GRAYSCALE);

	Ptr<FeatureDetector> featureExtractor;
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

	if (_denseSaming)
	{
		DenseFeatureDetector detector(1, 1, 0.1, _gridSize);
		detector.detect(image, _keypoints);
		descriptorExtractor->compute(image, _keypoints, _descriptors);
	}
	else
	{
		initModule_nonfree();
		featureExtractor = FeatureDetector::create("HARRIS");
		featureExtractor->detect(image, _keypoints);
		descriptorExtractor->compute(image, _keypoints, _descriptors);
	}

	if (_keypoints.empty() || _descriptors.empty())
		statusOk = false;

	return statusOk;
}

template<>
void Helper::printMatrix<int>(const Mat &_matrix, const int _precision, const string &_name)
{
	string format = "%" + to_string(_precision) + "d\t";

	printf("%s\n", _name.c_str());
	for (int i = 0; i < _matrix.rows; i++)
	{
		for (int j = 0; j < _matrix.cols; j++)
		{
			printf(format.c_str(), _matrix.at<int>(i, j));
		}
		printf("\n");
	}
}

string Helper::calculateHash(const vector<string> &_imageLocationList, const string &_extra)
{
	string names = "";
	for (string location : _imageLocationList)
		names += location.substr(location.find_last_of('/') + 1);

	names += ("-" + _extra);

	hash<string> strHash;
	return toHexString(strHash(names));
}

void Helper::concatMats(vector<Mat> &_vec, Mat &_res)
{
	int s = _vec.size();
	Mat aux;
	vconcat(_vec[0], _vec[1], aux);

	for (int i = 2; i < s - 1; i++)
	{
		vconcat(aux, _vec[i], aux);
	}

	vconcat(aux, _vec[s - 1], _res);
}
