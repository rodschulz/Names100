/**
 * Author: rodrigo
 * 2015
 */
#include "Config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <math.h>

Config::Config()
{
	sampleImageSet = false;
	codebookClusters = 5;
	sampleSize = 0.05;
}

Config::~Config()
{
}

void Config::load(const string &_filename)
{
	string line;
	ifstream inputFile;
	inputFile.open(_filename.c_str(), fstream::in);
	if (inputFile.is_open())
	{
		while (getline(inputFile, line))
		{
			// Parse string line
			vector<string> tokens;
			istringstream iss(line);
			copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(tokens));

			parse(tokens[0], tokens[1]);
		}
		inputFile.close();
	}
	else
		cout << "Unable to open input: " << _filename;
}

void Config::parse(const string _key, const string _value)
{
	if (_key.compare("sampleImageSet") == 0)
		getInstance()->sampleImageSet = _value.compare("true") == 0;
	else if (_key.compare("sampleSize") == 0)
		getInstance()->sampleSize = atof(_value.c_str());
	else if (_key.compare("codebookClusters") == 0)
		getInstance()->codebookClusters = atoi(_value.c_str());
}

bool Config::createImageSample()
{
	return getInstance()->sampleImageSet;
}

int Config::getCodebookClustersNumber()
{
	return getInstance()->codebookClusters;
}

double Config::getSampleSize()
{
	return getInstance()->sampleSize;
}
