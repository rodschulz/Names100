/**
 * Author: rodrigo
 * 2015
 */
#include "Codebook.h"
#include "Helper.h"
#include "Config.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <iostream>

Codebook::Codebook(const int _clusterNumber, const bool _useDenseSampling, const int _gridSize)
{
	centers = Mat::zeros(1, 1, CV_32FC1);
	clusterNumber = _clusterNumber;
	denseSampling = _useDenseSampling;
	gridSize = _gridSize;
	dataHash = "";
}

Codebook::Codebook(const Codebook &_other)
{
	centers = _other.centers.clone();
	clusterNumber = _other.clusterNumber;
	denseSampling = _other.denseSampling;
	gridSize = _other.gridSize;
	dataHash = _other.dataHash;
	index = _other.index;
}

Codebook::Codebook()
{
	centers = Mat::zeros(1, 1, CV_32FC1);
	clusterNumber = 1;
	denseSampling = false;
	gridSize = 1;
	dataHash = "";
}

Codebook::~Codebook()
{
}

Codebook &Codebook::operator=(const Codebook &_other)
{
	if (this != &_other)
	{
		centers = _other.centers.clone();
		clusterNumber = _other.clusterNumber;
		denseSampling = _other.denseSampling;
		gridSize = _other.gridSize;
		dataHash = _other.dataHash;
		index = _other.index;
	}

	return *this;
}

ostream &operator<<(ostream &_stream, const Codebook &_codebook)
{
	int rows = _codebook.centers.rows;
	int cols = _codebook.centers.cols;
	int dataType = _codebook.centers.type();

	_stream << boolalpha << _codebook.dataHash << " " << rows << " " << cols << " " << _codebook.denseSampling << " " << _codebook.gridSize << "\n";
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (dataType == CV_64FC1)
				_stream << _codebook.centers.at<double>(i, j) << " ";
			else if (dataType == CV_32FC1)
				_stream << _codebook.centers.at<float>(i, j) << " ";
		}
		_stream << "\n";
	}

	return _stream;
}

void Codebook::calculateCodebook(const string &_dataLocation, const int _maxInterationNumber, const double _stopThreshold)
{
	vector<string> imageLocationList;
	Helper::getContentsList(_dataLocation, imageLocationList);

	vector<Mat> descriptors;
	descriptors.reserve(imageLocationList.size());

	Mat samples;
	for (string imageLocation : imageLocationList)
	{
		// Calculate image's descriptors
		vector<KeyPoint> keypoints;
		descriptors.push_back(Mat());
		if (!Helper::calculateImageDescriptors(imageLocation, descriptors.back(), keypoints, denseSampling, gridSize))
			continue;

		if (samples.rows == 0)
			descriptors.back().copyTo(samples);
		else
			vconcat(samples, descriptors.back(), samples);
	}

	int attempts = 5;
	Mat labels;
	kmeans(samples, clusterNumber, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, _maxInterationNumber, _stopThreshold), attempts, KMEANS_PP_CENTERS, centers);

	// Hash of the files used for the codebook (just the names for now)
	dataHash = Helper::calculateHash(imageLocationList, Config::getConfigHash());
}

void Codebook::saveToFile(const string &_destinationFolder) const
{
	// Attempt to create folder in case if doesn't exists
	string cmd = "mkdir -p " + _destinationFolder;
	if (system(cmd.c_str()) != 0)
		cout << "WARNING: wrong command while saving cache\n";

	fstream cacheFile;
	cacheFile.open(_destinationFolder + dataHash + ".dat", fstream::out);
	cacheFile << *this;
	cacheFile.close();
}

void Codebook::getBoWTF(const Mat &_descriptors, Mat &_BoW)
{
	index.build(centers, flann::KDTreeIndexParams(4));

	if (_BoW.cols != centers.rows)
	{
		cout << "ERROR: wrong dimensions on BoW calculation\n";
		return;
	}

	if (_descriptors.rows > 0)
	{
		// Get frequencies of each word
		for (int i = 0; i < _descriptors.rows; i++)
		{
			Mat indices, distances, currentRow;
			_descriptors.row(i).copyTo(currentRow);

			index.knnSearch(currentRow, indices, distances, 1);
			_BoW.at<float>(0, indices.at<int>(0, 0)) += 1;
		}
		// Normalize using the total of word to get the TF
		_BoW *= (1 / (float) _descriptors.rows);
	}
}

bool Codebook::loadCodebook(const string &_sampleLocation, const string &_cacheLocation, vector<Codebook> &_codebooks)
{
	// Calculate hash of data in sample
	vector<string> imageLocationList;
	Helper::getContentsList(_sampleLocation, imageLocationList);

	// Hash of the files used for the codebook (just the names for now)
	string sampleHash = Helper::calculateHash(imageLocationList, Config::getConfigHash());
	string filename = _cacheLocation + sampleHash + ".dat";

	bool codebookRead = false;
	string line;
	ifstream inputFile;
	inputFile.open(filename.c_str(), fstream::in);
	if (inputFile.is_open())
	{
		int rows = -1;
		int cols = -1;
		int i = 0;
		while (getline(inputFile, line))
		{
			vector<string> tokens;
			istringstream iss(line);
			copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(tokens));

			if (rows == -1)
			{
				if (tokens.size() != 5)
					break;

				rows = stoi(tokens[1]);
				cols = stoi(tokens[2]);
				bool useDenseSampling = tokens[3].compare("true") == 0;
				int samplingGridSize = stoi(tokens[4]);

				_codebooks.push_back(Codebook(rows, useDenseSampling, samplingGridSize));
				_codebooks.back().dataHash = sampleHash;
				_codebooks.back().centers = Mat::zeros(rows, cols, CV_32FC1);
			}
			else
			{
				int j = 0;
				for (string value : tokens)
				{
					_codebooks.back().centers.at<float>(i, j++) = stof(value);
				}
				i++;
			}
		}
		inputFile.close();
		codebookRead = rows != -1 ? true : false;
	}

	return codebookRead;
}
