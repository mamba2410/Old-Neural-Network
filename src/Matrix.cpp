#include <Matrix.h>

using namespace std;

/*
	Creates a matrix of dimensions 
	row, column and initialises a vector
	array for the contents with zero
*/
Matrix::Matrix(int r, int c) : rows(r), columns(c) {
	contents = vector<double>(r*c);
	this->zeroFill();
}

/*
	Creates a matrix of dimensions 
	row, columns and initialises it with the 
	vector array given
*/
Matrix::Matrix(int r, int c, vector<double> cont) : rows(r), columns(c), contents(cont) {}

Matrix::~Matrix() {}

/*
	Returns the number of  rows of the matrix
*/
int Matrix::getRows() {
	return this->rows;
}

/*
	Returns the number of columns of the matrix
*/
int Matrix::getColumns() {
	return this->columns;
}

/*
	Returns the vector array of the contents
	of the matrix
	Read breadth first
*/
vector<double> Matrix::getContents() {
	return this->contents;
}

/*
	Setd the contents of the matrix to the 
	given vector array
*/
void Matrix::setContents(vector<double> cont) {
	if (cont.size() >= contents.size()) {
		for (int i = 0; i < this->contents.size(); i++)
			contents[i] = cont[i];
	} else cout << "Contents supplied are insufficient for the matrix " << this << endl;
}

/*
	Returns the value in the specified cell
	of the matrix
*/
double Matrix::getCell(int r, int c) {
	return this->contents[(r*this->columns) + c];
}

/*
	Sets the value in the specified cell to the 
	value given
*/
void Matrix::setCell(int r, int c, double value) {
	this->contents[(r*this->columns) + c] = value;
}

/*
	Fills the matrix with the vector array given
	Contents is filled depth-first
*/
void Matrix::fillDepth(vector<double> cont) {
	if (cont.size() >= contents.size()) {
		for (int c = 0; c < this->getColumns(); c++)
			for (int r = 0; r < this->getRows(); r++) {
				this->setCell(r, c, cont[this->getRows()*c + r]);
			}
	} else {
		cout << "Contents supplied are insufficient for the matrix " << this << endl;
	}
}

/*
	Fills the matrix with the contents given
	Contents is filled breadth-first
*/
void Matrix::fillBreadth(vector<double> cont) {
	this->setContents(cont);
}

/*
	Prints the matrix formatted so that it
	looks like a matrix
*/
void Matrix::printMatrix() {
	cout << this->getRows() << "x" << this->getColumns() << " Matrix: " << this;
	for (int i = 0; i < this->rows*this->columns; i++) {
		if (i%this->columns == 0) cout << endl << "\t";
		cout << this->contents[i] << "  ";
	}
	cout << endl << endl;
}

/*
	Prints the vector array contents in a single line
	element by element
*/
void Matrix::printContents() {
	cout << "Matrix: " << this << endl;
	for (int i = 0; i < this->rows*this->columns; i++) cout << contents[i] << " ";
	cout << endl;
}

/*
	Performs a vector multiplication of the matrices given
	It obeys matrix rules and returns the appropriate
	output
*/
Matrix* Matrix::vectorMult(Matrix* n) {
	if (this->columns == n->getRows()) {
		Matrix* fin = new Matrix(this->rows, n->getColumns());
		double temp = 0;

		for (int cn = 0; cn < n->getColumns(); cn++) {
			for (int rm = 0; rm < this->getRows(); rm++) {
				for (int rn = 0; rn < n->getRows(); rn++)
					temp += n->getCell(rn, cn) * (this->getCell(rm, rn));
				fin->setCell(rm, cn, temp);
				temp = 0;
			}
		}
		return fin;
	} else {
		cout << "Matrices " << this << " and " << n << " are not compatible for multiplication: ";
		printf("%dx%d and %dx%d\n", this->getRows(), this->getColumns(), n->getRows(), n->getColumns());
		return this;
	}

}

/*
	Performs a scalar multiplication of
	the matrices cell by cell, much like
	addition, but multiplying instead
*/
Matrix* Matrix::elementMult(Matrix* n) {
	//cout << "Element multiplying  " << this->getRows() << "x" << this->getColumns() << " by " << n->getRows() << "x" << n->getColumns() << endl;
	int sr = this->getRows() < n->getRows() ? this->getRows() : n->getRows();
	int sc = this->getColumns() < n->getColumns() ? this->getColumns() : n->getColumns();
	Matrix* p = new Matrix(sr, sc);

	for (int r = 0; r < sr; r++) {
		for (int c = 0; c < sc; c++)
			p->setCell(r, c, this->getCell(r, c)*n->getCell(r, c));
	}
		return p;
}

/*
	Performs a scalar division by dividing
	each element by the corresponding element
	in the other matrix
*/
Matrix* Matrix::elementDiv(Matrix* n) {
	int sr = this->getRows() < n->getRows() ? this->getRows() : n->getRows();
	int sc = this->getColumns() < n->getColumns() ? this->getColumns() : n->getColumns();
	Matrix* p = new Matrix(sr, sc);

	for (int r = 0; r < sr; r++) {
		for (int c = 0; c < sc; c++)
			p->setCell(r, c, this->getCell(r, c)/n->getCell(r, c));
	}
	return p;
}

/*
	Performs a matrix addition of the matrices
	where each cell is added to the corresponding
	cell of the other matrix
*/
Matrix* Matrix::add(Matrix* n) {
	Matrix* fin = new Matrix(this->getRows(), this->getColumns());
	if ((this->getRows() == n->getRows()) && (this->getColumns() == n->getColumns())) {
		for(int r = 0; r < fin->getRows(); r++)
			for (int c = 0; c < fin->getColumns(); c++)
				fin->setCell(r, c, this->getCell(r, c)+n->getCell(r, c));
		return fin;
	} else {
		cout << "Matrices " << this << " and " << n << " are incompatible for addition" << endl;
		return this;
	}
}

/*
	Perfors a matrix subtraction of the matrices
	where each cell is subtracted from the 
	corresponding cell of the other matrix
*/
Matrix* Matrix::subtract(Matrix* n) {
	Matrix* fin = new Matrix(this->getRows(), this->getColumns());
	if ((this->getRows() == n->getRows()) && (this->getColumns() == n->getColumns())) {
		for (int r = 0; r < fin->getRows(); r++)
			for (int c = 0; c < fin->getColumns(); c++)
				fin->setCell(r, c, this->getCell(r, c) - n->getCell(r, c));
		return fin;
	} else {
		cout << "Matrices " << this << " and " << n << " are incompatible for subtraction" << endl;
		return this;
	}
}

/*
	Returns the transposition of the matrix
	As if the matrix was reflected in the
	line y = -x
*/
Matrix* Matrix::getT() {
	Matrix* fin = new Matrix(this->getColumns(), this->getRows());
	fin->fillDepth(this->getContents());
	return fin;
}

/*
	Returns the matrix being multiplied
	by the scalar given and performs the 
	operatino on itself
*/
Matrix* Matrix::scaleUp(double s) {
	Matrix* fin = new Matrix(this->getRows(), this->getColumns());
	vector<double> fc(fin->getRows()*fin->getColumns());
	for (int i = 0; i < fc.size(); i++)
		fc[i] = this->contents[i] * s;
	//this->setContents(fc);
	fin->setContents(fc);
	return fin;
}

/*
	Returns the matrix being divided
	by the scalar given and performs the
	operatino on itself
*/
Matrix* Matrix::scaleDown(double s) {
	Matrix* fin = new Matrix(this->getRows(), this->getColumns());
	vector<double> fc(fin->getRows()*fin->getColumns());
	for (int i = 0; i < fc.size(); i++)
		fc[i] = this->contents[i] / s;
	//this->setContents(fc);
	fin->setContents(fc);
	return fin;
}

/*
	Returns the Frodian norm of the matrix
*/
double Matrix::norm() {
	double tot = 0;
	for (double i : this->contents)
		tot += i*i;
	return sqrt(tot);
}

/*
	Returns the mean of all of the elements of the matrix
*/
double Matrix::mean() {
	double sum = 0;
	for (double x : this->contents)
		sum += x;
	return (sum / (double)(this->getContents().size()));
}

/*
	Returns the mean of the absolute values of the matrix
*/
double Matrix::absMean() {
	double sum = 0;
	for (double x : this->contents)
		if(x > 0) sum += x;
		else sum -= x;
	return (sum / (double)(this->getContents().size()));
}

/*
	Returns the sum of the squares of the matrix
*/
double Matrix::getSquareSum() {
	double f = 0;
	for (double x : this->contents)
		f += (x*x);
	return f;
}

/*
	Fills the matrix with random numbers 
	between zero and 1, seeded with the time
*/
void Matrix::makeNoise() {
	srand(time(0));
	for (int i = 0; i < this->rows*this->columns; i++) {
		this->contents[i] = (double)rand() / RAND_MAX;
	}
}

/*
	Fills the matrix with the specified double
*/
void Matrix::fillWith(double v) {
	for (int i = 0; i < this->contents.size(); i++)
		this->contents[i] = v;
}

/*
	Fills the matrix with zeros to initialise
	the contents vector array
*/
void Matrix::zeroFill() {
	this->fillWith(0);
}

/*
	Turns the matrix into the identity matrix
	of its size. Must be square
*/
void Matrix::fillIdentity() {
	if (this->getRows() == this->getColumns()) {
		this->zeroFill();
		for (int i = 0; i < (this->getRows()); i++)
			this->setCell(i, i, 1);
	} else cout << "Matrix is not square" << endl;
}


