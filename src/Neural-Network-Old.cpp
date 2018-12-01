#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <ctime>
#include <random>
#include <cmath>

#include <Matrix.h>
#include <NeuralNetwork.h>

using namespace std;

int main(int argc, char** argv){

	char cat;
	/*
		Inputs and outputs, X is hours (max 24), Y is in percent (max 100)
		As long as inputs are between 0 and 1, it should be fine
	*/
	Matrix* X = new Matrix(3, 2, { 3, 5, 5, 1, 10, 2 });
	Matrix* Y = new Matrix(3, 1, { 75, 82, 93 });
	//Matrix* X = new Matrix(3, 2, { 8, 2, 5, 7, 4, 8 });
	//Matrix* Y = new Matrix(3, 1, { 63, 95, 81 });
	X = X->scaleDown(24);
	Y = Y->scaleDown(100);
	cout << endl;

	/*
		Creating the network
		Passes a vector of the depths of layers,
		eg layer 0 (input layer) is of depth 2 (requires 2 arguments per input) aka row of size 2
		final layer is output layer of depth 1 (only want 1 output per input) aka row size of 1
		Any layers in between are hidden layers and their depth does not really matter but more gives more precision
	*/
	NeuralNetwork net({ 2, 3, 3, 1}, 1e-6);

	cout << "Gradients" << endl;
	vector<Matrix*> v = net.dCost(X, Y);
	for (Matrix* x : v)
		x->printMatrix();

	cout << "Gradients (int)" << endl;
	v = net.interpolateDCost(X, Y, 1e-4);
	for (Matrix* x : v)
		x->printMatrix();
	vector<double> d = net.evaluateGradientDifference(X, Y, 1e-4);
	for (double x : d)
		cout << x << endl;

	cout << "Results: " << endl;
	net.forward(X)->printMatrix();

	cout << "\nPress anything to continue: ";
	scanf("%c", &cat);

	return 0;
}

