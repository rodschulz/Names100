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
	createSample = false;
	codebookSize = 5;
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
		cout << "Unable to open input: " << _filename << "\n";
}

void Config::parse(const string _key, const string _value)
{
	if (_key.compare("cacheLocation") == 0)
		getInstance()->cacheLocation = _value;

	else if (_key.compare("createSample") == 0)
		getInstance()->createSample = _value.compare("true") == 0;

	else if (_key.compare("sampleSize") == 0)
		getInstance()->sampleSize = atof(_value.c_str());

	else if (_key.compare("codebookSize") == 0)
		getInstance()->codebookSize = atoi(_value.c_str());
}

string Config::getCacheLocation()
{
	return getInstance()->cacheLocation;
}

bool Config::createImageSample()
{
	return getInstance()->createSample;
}

int Config::getCodebookSize()
{
	return getInstance()->codebookSize;
}

double Config::getSampleSize()
{
	return getInstance()->sampleSize;
}
