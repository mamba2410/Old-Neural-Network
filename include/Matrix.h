#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <ctime>
#include <random>
#include <cmath>

using namespace std;

class Matrix {
public:
	Matrix(int r, int c);
	Matrix(int r, int c, vector<double> cont);
	~Matrix();
	void setContents(vector<double> cont);
	vector<double> getContents();
	void fillDepth(vector<double> cont);
	void fillBreadth(vector<double> cont);
	int	getRows();
	int	getColumns();
	double getCell(int r, int c);
	void setCell(int r, int c, double value);
	void printMatrix();
	void printContents();

	Matrix*	vectorMult(Matrix* n);
	Matrix*	elementMult(Matrix* n);
	Matrix* elementDiv(Matrix* n);
	Matrix*	add(Matrix* n);
	Matrix*	subtract(Matrix* n);
	Matrix*	getT();
	Matrix* scaleUp(double s);
	Matrix* scaleDown(double s);
	double norm();
	double mean();
	double absMean();
	double getSquareSum();

	void makeNoise();
	void zeroFill();
	void fillWith(double v);
	void fillIdentity();
	
private:
	int rows, columns;
	vector<double>	contents;

};

#endif
