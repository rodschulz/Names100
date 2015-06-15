/**
 * Author: rodrigo
 * 2015
 */
#pragma once

#include <string>
#include <functional>
#include <sstream>

using namespace std;

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
	static bool calculateTFIDF()
	{
		return getInstance()->useTFIDF;
	}
	static string getConfigHash()
	{
		string str = "";
		str += "codebookSize=" + to_string(getInstance()->codebookSize);
		str += "-denseSampling=" + to_string(getInstance()->denseSampling);
		str += "-gridSize=" + to_string(getInstance()->denseSampling ? getInstance()->gridSize : 0);

		hash<string> strHash;
		return to_string(strHash(str));
	}

private:
	Config();

	// Parses the given value according to the given key
	static void parse(const string _key, const string _value);

	string cacheLocation;

	bool createSample;
	double sampleSize;

	int codebookSize;
	int kmeansMaxIter;
	double kmeansThres;

	bool denseSampling;
	int gridSize;
	bool useTFIDF;
};

