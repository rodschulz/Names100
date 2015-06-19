/**
 * Author: rodrigo
 * 2015
 */
#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cv;

class Codebook
{
public:
	Codebook(const int _clusterNumber, const bool _useDenseSampling, const int _gridSize);
	Codebook(const Codebook &_other);
	Codebook();
	~Codebook();

	Codebook &operator=(const Codebook &_other);
	friend ostream &operator<<(ostream &_stream, const Codebook &_codebook);

	void calculateCodebook(const string &_dataLocation, const int _maxInterationNumber, const double _stopThreshold);
	void saveToFile(const string &_destinationFolder) const;
	void getBoWTF(const Mat &_descriptor, Mat &_BoW);
	void calculateLLC(const Mat &_descriptors, const vector<KeyPoint> &_keypoints, const int _neighborhood, Mat &_BoW);

	inline int getClusterNumber() const
	{
		return clusterNumber;
	}
	inline bool usingDenseSampling()
	{
		return denseSampling;
	}
	inline int getGridSize()
	{
		return gridSize;
	}

	static bool loadCodebook(const string &_imageSampleLocation, const string &_cacheLocation, vector<Codebook> &_codebooks);

private:
	void buildIndex();
	Mat getGradient(const Mat &_diff, const Mat &_B);

	inline void copyIndexedRows(const Mat &_input, const Mat &_indices, Mat &_output)
	{
		_output = Mat();
		for (int j = 0; j < _indices.cols; j++)
		{
			if (_output.empty())
				_input.row(_indices.at<int>(j)).copyTo(_output);
			else
				vconcat(_output, _input.row(_indices.at<int>(j)), _output);
		}
	}

	inline void copyToIndex(const Mat &_input, const Mat &_indices, Mat _output)
	{
		for (int j = 0; j < _indices.cols; j++)
			_output.at<float>(0, _indices.at<int>(j)) = _input.at<float>(j, 0);
	}

	inline int getBinTotal(const vector<int> &_levels)
	{
		int total = 0;
		for (int level : _levels)
			total += (level * level);
		return total;
	}

	Mat centers;
	int clusterNumber;
	bool denseSampling;
	int gridSize;

	string dataHash;
	flann::Index index;
};

