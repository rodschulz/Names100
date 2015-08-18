/**
 * Author: rodrigo
 * 2015
 */
#pragma once

#include <string>
#include <functional>
#include <sstream>
#include <iomanip>
#include "Helper.h"

using namespace std;

enum BoWType
{
	FREQUENCIES, LLC_SPM
};

class Config
{
public:
	~Config();

	// Returns the instance of the singleton
	static Config *getInstance()
	{
		static Config *instance = new Config();
		return instance;
	}
	// Loads the configuration file
	static void load(const string &_filename);

	static string getCacheLocation()
	{
		return getInstance()->cacheLocation;
	}

	static bool createImageSample()
	{
		return getInstance()->createSample;
	}
	static double getSampleSize()
	{
		return getInstance()->sampleSize;
	}

	static int getCodebookSize()
	{
		return getInstance()->codebookSize;
	}
	static int getKMeansMaxIterations()
	{
		return getInstance()->kmeansMaxIter;
	}
	static double getKMeansThreshold()
	{
		return getInstance()->kmeansThres;
	}

	static bool useDenseSampling()
	{
		return getInstance()->denseSampling;
	}
	static int getGridSize()
	{
		return getInstance()->gridSize;
	}
	static BoWType getBoWType()
	{
		return getInstance()->bowType;
	}
	static bool calculateTFIDF()
	{
		return getInstance()->useTFIDF;
	}
	static int getPyramidLevelNumber()
	{
		return getInstance()->levels;
	}
	static int getLLCNeighbors()
	{
		return getInstance()->neighbors;
	}
	static bool testRandomClassification()
	{
		return getInstance()->testRandom;
	}
	static int getRandomClassificationIterations()
	{
		return getInstance()->testIterations;
	}
	static string getConfigHash()
	{
		string str = "";
		str += "codebookSize=" + to_string(getInstance()->codebookSize);
		str += "-denseSampling=" + to_string(getInstance()->denseSampling);
		str += "-gridSize=" + to_string(getInstance()->denseSampling ? getInstance()->gridSize : 0);

		hash<string> strHash;
		return Helper::toHexString(strHash(str));
	}

private:
	Config();

	// Parses the given value according to the given key
	static void parse(const string _key, const string _value);

	static inline BoWType parseBowType(const string &_value)
	{
		if (_value.compare("LLC") == 0)
			return LLC_SPM;
		else
			return FREQUENCIES;
	}

	string cacheLocation;

	bool createSample;
	double sampleSize;

	int codebookSize;
	int kmeansMaxIter;
	double kmeansThres;

	bool denseSampling;
	int gridSize;

	BoWType bowType;

	bool useTFIDF;

	int levels;
	int neighbors;

	bool testRandom;
	int testIterations;
};

