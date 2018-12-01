#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <iostream>
#include <ctime>
#include <random>
#include <cmath>

#include <Matrix.h>

using namespace std;

class NeuralNetwork{
public:
	NeuralNetwork(vector<int> nodes);
	NeuralNetwork(vector<int> nodes, double lambda);
	~NeuralNetwork();
	void printWeights();

	Matrix* forward(Matrix* input);
	double cost(Matrix* input, Matrix* expected);
	vector<Matrix*> getDeltas(Matrix* x, Matrix* p);
	vector<Matrix*> dCost(Matrix* x, Matrix* p);
	vector<Matrix*> interpolateDCost(Matrix* x, Matrix* p, double e);
	vector<double> evaluateGradientDifference(Matrix* x, Matrix* p, double e);
	Matrix* wrapVector(vector<Matrix*> v);

	void bfgsOptim(Matrix* x, Matrix* p);
	void printA(Matrix* input);

private:
	vector<Matrix*> weightmap;
	vector<int> nodestruct;
	double lambda;

	double cost(double input, double expected);
	Matrix* getZ(Matrix* input, int n);
	vector<Matrix*> getZTrail(Matrix* input);
	Matrix* getA(Matrix* input, int n);
	vector<Matrix*> getATrail(Matrix* input);
	double activation(double i);
	Matrix* activation(Matrix* i);
	double dActivation(double i);
	Matrix* dActivation(Matrix* i);
	double d2Activation(double i);
	Matrix* wrapWeights();
	void setWeights(vector<double> w);
	void setWeights(Matrix* w);

};

#endif

