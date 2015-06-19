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
	cacheLocation = "";

	createSample = false;
	sampleSize = 0.05;

	codebookSize = 5;
	kmeansMaxIter = 10000;
	kmeansThres = 0.1;

	denseSampling = false;
	gridSize = 0;

	bowType = FREQUENCIES;

	useTFIDF = false;

	levels = 2;
	neighbors = 2;

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
			if (line.empty() || line[0] == '#')
				continue;

			// Parse string line
			vector<string> tokens;
			istringstream iss(line);
			copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(tokens));

			parse(tokens[0], tokens[1]);
		}
		inputFile.close();
	}
	else
		cout << "Unable to open input: " << _filename << "\n";
}

void Config::parse(const string _key, const string _value)
{
	// cache read/write location
	if (_key.compare("cacheLocation") == 0)
		getInstance()->cacheLocation = _value;

	// sample options
	else if (_key.compare("createSample") == 0)
		getInstance()->createSample = _value.compare("true") == 0;

	else if (_key.compare("sampleSize") == 0)
		getInstance()->sampleSize = atof(_value.c_str());

	// codebook options
	else if (_key.compare("codebookSize") == 0)
		getInstance()->codebookSize = atoi(_value.c_str());

	else if (_key.compare("kmeansMaxIter") == 0)
		getInstance()->kmeansMaxIter = atoi(_value.c_str());

	else if (_key.compare("kmeansThres") == 0)
		getInstance()->kmeansThres = atof(_value.c_str());

	// Sampling options
	else if (_key.compare("denseSampling") == 0)
		getInstance()->denseSampling = _value.compare("true") == 0;

	else if (_key.compare("gridSize") == 0)
		getInstance()->gridSize = atoi(_value.c_str());

	// BoW type options
	else if (_key.compare("bowType") == 0)
		getInstance()->bowType = parseBowType(_value);

	// BoW freq options
	else if (_key.compare("useTFIDF") == 0)
		getInstance()->useTFIDF = _value.compare("true") == 0;

	// BoW LLC options
	else if (_key.compare("levels") == 0)
		getInstance()->levels = atoi(_value.c_str());

	else if (_key.compare("neighbors") == 0)
		getInstance()->neighbors = atoi(_value.c_str());
}
