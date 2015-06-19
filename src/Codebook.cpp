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
		int width, height;
		vector<KeyPoint> keypoints;
		descriptors.push_back(Mat());
		if (!Helper::calculateImageDescriptors(imageLocation, descriptors.back(), keypoints, width, height, denseSampling, gridSize))
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
	if (_BoW.cols != centers.rows)
	{
		cout << "ERROR: wrong dimensions on BoW calculation\n";
		return;
	}

	if (_descriptors.rows > 0)
	{
		index.build(centers, flann::KDTreeIndexParams(4));

		// Get frequencies of each word
		for (int i = 0; i < _descriptors.rows; i++)
		{
			Mat indices, distances, currentRow;
			_descriptors.row(i).copyTo(currentRow);

			index.knnSearch(currentRow, indices, distances, 1);
			_BoW.at<float>(0, indices.at<int>(0, 0)) += 1;
		}
		// Normalize using the total of words to get the TF
		_BoW *= (1 / (float) _descriptors.rows);
	}
}

void Codebook::calculateLLC(const Mat &_descriptors, const vector<KeyPoint> &_keypoints, const int _neighborhood, Mat &_BoW, const int _imgWidth, const int _imgHeight, const int _levels)
{
	if (_descriptors.rows > 0)
	{
		index.build(centers, flann::KDTreeIndexParams(4));

		/**
		 * Make the LLC discretization (using the KNN approximation)
		 */

		// Find the code for each descriptor (1 row == 1 descriptor)
		Mat codes = Mat::zeros(_descriptors.rows, centers.rows, CV_32FC1);
		float beta = 1E-4;
		for (int i = 0; i < _descriptors.rows; i++)
		{
			// Find the closest neighbors
			Mat indices, distances, currentRow;
			_descriptors.row(i).copyTo(currentRow);
			index.knnSearch(currentRow, indices, distances, _neighborhood);

			// Extract the closer centers
			Mat B;
			copyIndexedRows(centers, indices, B);

			// Shift target descriptor to the origin
			Mat z = Mat::zeros(B.rows, B.cols, CV_32FC1);
			for (int j = 0; j < B.rows; j++)
				((Mat) (B.row(j) - currentRow)).copyTo(z.row(j));

			Mat C = z * z.t();
			C = C + (Mat::eye(B.rows, B.rows, CV_32FC1) * beta * trace(C)[0]);
			Mat w = C.inv() * Mat::ones(B.rows, 1, CV_32FC1);
			w /= sum(w)[0];

			// Copy data to the right positions
			copyToIndex(w, indices, codes.row(i));
		}
		codes = codes.t();

		/**
		 * Max pooling
		 */
		vector<int> levels = Helper::generateLevels(_levels);
		int totalBins = Helper::getSqrSum<int>(levels);
		vector<bool> initialized(totalBins, false);

		// Create a pool with data from the coding phase
		Mat pool = Mat::zeros(centers.rows, totalBins, CV_32FC1);
		int acc = 0;
		for (int level : levels)
		{
			double binWidth = (double) _imgWidth / level;
			double binHeight = (double) _imgHeight / level;

			// If level is 1, then all the keypoints are in the same bin
			if (level == 1)
			{
				for (int i = 0; i < codes.rows; i++)
				{
					double minCode, maxCode;
					minMaxIdx(codes.row(i), &minCode, &maxCode);
					pool.at<float>(i, 0) = maxCode;
				}
				initialized[0] = true;
				acc += level;
			}
			else
			{
				// Iterate over each keypoint updating the pooling according to
				// the bin it belongs to
				for (size_t k = 0; k < _keypoints.size(); k++)
				{
					// Get the bin where the current keypoint is
					int binX = _keypoints[k].pt.x / binWidth;
					int binY = _keypoints[k].pt.y / binHeight;
					int bin = binY * level + binX + acc;

					Mat poolBin = pool.col(bin);
					if (!initialized[bin])
					{
						codes.col(k).copyTo(poolBin);
						initialized[bin] = true;
					}
					else
						cv::max(codes.col(k), pool.col(bin), poolBin);
				}
				acc += (level * level);
			}
		}

		// Finally unravel the pooling matrix into a single vector as the LLC encoding with pyramid
		if (_BoW.isContinuous())
		{
			Mat transposedPool = pool.t();
			memcpy(_BoW.data, transposedPool.data, sizeof(float) * _BoW.cols);
		}
		else
		{
			for (int i = 0; i < pool.cols; i++)
				for (int j = 0; j < pool.rows; j++)
					_BoW.at<float>(0, i * pool.rows + j) = pool.at<float>(j, i);
		}

		Mat sqrSum = _BoW * _BoW.t();
		_BoW /= sqrt(sqrSum.at<float>(0, 0));
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
