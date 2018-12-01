
#include <NeuralNetwork.h>
#include <Matrix.h>

#define PHI 1.61803398875

using namespace std;

/*
	Creates a new neural network
	Given the number of nodes on each layer, it generates a vector of weight 
	matrices and randomises the values inside the matrices
	It also takes a learning rate paramater, if left blank, it defaults to zero
*/
NeuralNetwork::NeuralNetwork(vector<int> nodes, double lambda) : nodestruct(nodes), lambda(lambda){
	this->weightmap = vector<Matrix*>(nodes.size() - 1);
	for (int i = 0; i < (nodes.size() - 1); i++) {
		weightmap[i] = new Matrix(nodes[i], nodes[i + 1]);
		weightmap[i]->makeNoise();
	}
}

/*
	If no learning rate is set, it defaults to zero
*/
NeuralNetwork::NeuralNetwork(vector<int> nodes) {
	NeuralNetwork(nodes, 0);
}

NeuralNetwork::~NeuralNetwork() {}

/*
	Prints all of the weights in weightmap fully
*/
void NeuralNetwork::printWeights() {
	for (Matrix* i : weightmap)
		i->printMatrix();
}

/*
	Turns the matrix vector into a single nx1 matrix
*/
Matrix* NeuralNetwork::wrapVector(vector<Matrix*> v) {
	vector<double> fc;
	for (Matrix* x : v) {
		for (double d : x->getContents())
			fc.push_back(d);
	}
	return new Matrix(fc.size(), 1, fc);
}

/*
	A function to return the weights in an nx1 matrix
*/
Matrix* NeuralNetwork::wrapWeights() {
	return wrapVector(this->weightmap);
}

/*
A function to set the weights from a double vector
*/
void NeuralNetwork::setWeights(vector<double> c) {
	vector<double> t;
	int i, p = 0, t2;
	for (Matrix* x : weightmap) {
		t = {};
		t2 = (x->getRows())*(x->getColumns());
		for (i = 0; i < t2; i++) {
			//cout << "Attempting to access " << i+p << " of a " << c.size() << endl;
			t.push_back(c[i + p]);
		}
		x->setContents(t);
		p += i;
	}
}

/*
	A function to set the weights from an nx1 matrix
*/
void NeuralNetwork::setWeights(Matrix* w) {
	this->setWeights(w->getContents());
}

/*
	Activation function for the network
	Ranges from 0 to 1
*/
double NeuralNetwork::activation(double z) {
	return (double)(1.0 / (1.0 + exp(-z)));
}

/*
	Performs the activation function element-wise to the matrix given
*/
Matrix* NeuralNetwork::activation(Matrix* m) {
	Matrix* fin = new Matrix(m->getRows(), m->getColumns());
	for (int r = 0; r < m->getRows(); r++) {
		for (int c = 0; c < m->getColumns(); c++)
			fin->setCell(r, c, activation(m->getCell(r, c)));
	}
	return fin;
}

/*
	The differential of the activation function with respect to z
*/
double NeuralNetwork::dActivation(double z) {
	return this->activation(z)*(1 - this->activation(z));
}

/*
	The differential of the activation function applied element-wise
	to the matrix given
*/
Matrix* NeuralNetwork::dActivation(Matrix* m) {
	Matrix* fin = new Matrix(m->getRows(), m->getColumns());
	for (int r = 0; r < m->getRows(); r++) {
		for (int c = 0; c < m->getColumns(); c++) {
			fin->setCell(r, c, this->dActivation(m->getCell(r, c)));
		}
	}
	return fin;
}

/*
	The second differential of the activation
	function with respect to z
*/
double NeuralNetwork::d2Activation(double z) {
	return exp(-z)*(exp(-z) - 1)*pow(this->activation(z), 3);
}

/*
	The second differential of the activation function applied 
	element-wise to the matrix given

Matrix* NeuralNetwork::d2Activation(Matrix* m) {
	Matrix* fin = new Matrix(m->getRows(), m->getColumns());
	for (int r = 0; r < m->getRows(); r++) {
		for (int c = 0; c < m->getColumns(); c++) {
			fin->setCell(r, c, this->d2Activation(m->getCell(r, c)));
		}
	}
	return fin;
}
*/

/*
	Gets the z matrix of the given index
	Working forwards from input to output
*/
Matrix* NeuralNetwork::getZ(Matrix* x, int an) {
	return this->getZTrail(x)[an];
}

/*
	Gets a vector of all of the z values
	Working forwards from input to output
*/
vector<Matrix*> NeuralNetwork::getZTrail(Matrix* x) {
	vector<Matrix*> finTrail;
	Matrix* fin = new Matrix(x->getRows(), x->getColumns());
	*fin = *x;
	finTrail.push_back(fin);
	for (int i = 0; i < this->weightmap.size(); i++) {
		fin = fin->vectorMult(this->weightmap[i]);
		finTrail.push_back(fin);
		fin = this->activation(fin);
	}
	return finTrail;
}

/*
	Gets the a matrix of the given index
	Working forwards from input to output
*/
Matrix* NeuralNetwork::getA(Matrix* x, int an) {
	return this->getATrail(x)[an];
}

/*
	Gets a vector of all of the a values of a
	Working forwards from input to output
*/
vector<Matrix*> NeuralNetwork::getATrail(Matrix* x) {
	vector<Matrix*> finTrail;
	Matrix* fin = new Matrix(x->getRows(), x->getColumns());
	*fin = *x;
	finTrail.push_back(fin);
	for (int i = 0; i < this->weightmap.size(); i++) {
		fin = fin->vectorMult(this->weightmap[i]);
		fin = this->activation(fin);
		finTrail.push_back(fin);
	}
	return finTrail;
}

/*
	Returns the final (output) value of the forward propagation
	Working forwards from input to output
*/
Matrix* NeuralNetwork::forward(Matrix* x) {
	return getA(x, this->weightmap.size());
}

/*
	The cost function of the network,
	Returns the cost of the given values
*/
double NeuralNetwork::cost(double y, double p) {
	return (double)((p - y)*(p - y) / 2);
}

/*
	The function called to get the matrix of costs for each input
	One cost across all inputs
*/
double NeuralNetwork::cost(Matrix* x, Matrix* p) {
	Matrix* y = this->forward(x);
	double c = 0, w = 0;
	vector<double> pc = p->getContents();
	vector<double> yc = y->getContents();
	for (int i = 0; i < pc.size(); i++) {
		c += (pc[i] - yc[i])*(pc[i] - yc[i]);
	}
	for (Matrix* x : weightmap)
		w += (this->lambda * x->getSquareSum())/2;
	w = c*0.5 + w;
	return w;
}

/*
	Returns the delta values for each weight
	Working from output to input
	Returns deltas in ascending order
*/
vector<Matrix*> NeuralNetwork::getDeltas(Matrix* x, Matrix* p) {
	vector<Matrix*> bd, fd, zt, at;
	Matrix* y = this->forward(x);
	zt = this->getZTrail(x);
	int i = this->weightmap.size()-1;
	Matrix* t, *adz, *wt;
	// -(y-yHat)
	t = (p->subtract(y));
	t = t->scaleUp(-1);
	
	// f'(z)
	adz = this->dActivation(zt[i+1]);
	// -(y-yHat)...f'(z)
	t = t->elementMult(adz);
	bd.push_back(t);
	i--;
	for (; i >= 0; i--) {
		// d*wt
		wt = this->weightmap[i+1]->getT();
		t = bd.back()->vectorMult(wt);
		// f'(z)
		adz = this->dActivation(zt[i+1]);
		// (d*wt)...f'(z)
		t = t->elementMult(adz);
		bd.push_back(t);
	}
	for (int i = bd.size() - 1; i >= 0; i--)
		fd.push_back(bd[i]);
	return fd;
}

/*
	Returns a list of the differential of the cost 
	function with respect to the weights
*/
vector<Matrix*> NeuralNetwork::dCost(Matrix* x, Matrix* p) {
	vector<Matrix*> a = this->getATrail(x);
	vector<Matrix*> z = this->getZTrail(x);
	vector<Matrix*> d = this->getDeltas(x, p);
	Matrix* t;
	a.pop_back();

	for (int i = a.size() - 1; i > 0; i--) {
		a[i] = (a[i]->getT())->vectorMult(d[i]);
		t = this->weightmap[i];
		t = t->scaleUp(this->lambda);
		a[i] = a[i]->add(t);
	}
	a[0] = (z[0]->getT())->vectorMult(d[0]);
	t = this->weightmap[0];
	t = t->scaleUp(this->lambda);
	a[0] = a[0]->add(t);

	//cout << "Lambda is " << this->lambda << endl;
	
	return a;
}

/*
	Computes dcdw by linear interpolation 
	then returns a vector of each dcdw 
	matrix for each weight cell
*/
vector<Matrix*> NeuralNetwork::interpolateDCost(Matrix* x, Matrix* p, double e) {
	vector<Matrix*> dcdw;
	double orig, wpe, wme;
	for (Matrix* m : this->weightmap) {
		Matrix* cpw = new Matrix(m->getRows(), m->getColumns());
		for (int r = 0; r < m->getRows(); r++) {
			for (int c = 0; c < m->getColumns(); c++) {
				orig = m->getCell(r, c);
				m->setCell(r, c, orig+e);
				wpe = this->cost(x, p);
				m->setCell(r, c, orig-e);
				wme = this->cost(x, p);
				wpe = (wpe - wme) / (2*e);
				m->setCell(r, c, orig);
				cpw->setCell(r, c, wpe);
			}
		}
		dcdw.push_back(cpw);
	}
	return dcdw;
}

/*
	Quantifies the difference of the interpolated 
	gradients and the gradients calculated by function
*/
vector<double> NeuralNetwork::evaluateGradientDifference(Matrix* x, Matrix* p, double e) {
	vector<Matrix*> computed, interpolated;
	vector<double> difference;
	Matrix* t, *y = this->forward(x);
	double nm, np;
	computed = dCost(x, p);
	interpolated = interpolateDCost(x, p, e);
	for (int i = 0; i < computed.size(); i++) {
		t = computed[i]->subtract(interpolated[i]);
		nm = t->norm();
		t = computed[i]->add(interpolated[i]);
		np = t->norm();
		cout << "Difference: " << nm << "\tSum: " << np << endl;
		difference.push_back(nm / np);
	}
	return difference;
}

/*
	Performs BFGS gradient descent on the network
	to optimise the weights and train the network
*/
void NeuralNetwork::bfgsOptim(Matrix* training, Matrix* predicted) {
	int nit = 0, n;
	double gradNorm = 1;
	
	
	cout << "Starting BFGS" << endl;
	Matrix* Bi, *p, *s, *y, *x, *yt1, *yt2;
	vector<Matrix*> fd;
	x = this->wrapWeights();
	n = x->getContents().size();
	s = new Matrix(n, 1);
	s->zeroFill();
	//s->printMatrix();
	Bi = new Matrix(n, n);
	Bi->fillIdentity();
	
	for (; /*(nit < 100) && */(gradNorm > 1e-10); nit++) {
		// 1. Get the direction pk
		fd = this->dCost(training, predicted);
		p = this->wrapVector(fd);
		gradNorm = p->absMean();
		p = p->scaleUp(-1);
		p = Bi->vectorMult(p);

		// cout << "Direction ";
		// p->printMatrix();

		// For step 5
		yt1 = this->wrapVector(this->dCost(training, predicted));

		// 2, 3, 4. Golden Section Search for alphak, get sk, xk +sk
		// Input is actually x + alpha*p, not alpha
		Matrix* pw = this->wrapWeights();
		vector<double> pwc = pw->getContents();
		double a = 0, b = 5, c, d, c1, c2, f;
		c = b - ((b - a) / PHI);
		d = a + ((b - a) / PHI);
		while ((c-d)*(c-d) > 1e-6) {
			this->setWeights(pw->add(p->scaleUp(c)));
			c1 = cost(training, predicted);
			this->setWeights(pw->add(p->scaleUp(d)));
			c2 = cost(training, predicted);
			if (c1 < c2) b = d;
			else a = c;
			c = b - ((b - a) / PHI);
			d = a + ((b - a) / PHI);
		}
		f = (b + a) / 2;
		//cout << "alpha " << f << endl;
		// s = alphak*pk
		s = p->scaleUp(f);
		//cout << "s ";
		//s->printMatrix();
		pw = pw->add(s);
		this->setWeights(pw);
		//cout << "Weights ";
		//pw->printMatrix();

		// 5. y = f'(xn+1) - f'(xn)
		yt2 = this->wrapVector(this->dCost(training, predicted));
		y = yt2->subtract(yt1);
		//cout << "y";
		//y->printMatrix();

		// 6. Update the inverse hessian
		double t61, t62, t63;
		Matrix* t64, *t65, *Bit;
		// st*y
		t61 = ((s->getT())->vectorMult(y))->norm();
		// yt * bi * y
		t62 = ((y->getT())->vectorMult(Bi->vectorMult(y))->norm());
		// (st*y + yt*bi*y) / (st*y)^2
		t63 = (t62+t61) / (t61*t61);
		// Bi += 
		Bit = Bi->add( (s->vectorMult(s->getT()))->scaleUp(t63) );
		t64 = Bi->vectorMult(y->vectorMult(s->getT()));
		t65 = s->vectorMult((y->getT())->vectorMult(Bi));
		Bi = Bit->subtract((t64->add(t65))->scaleDown(t61));

		//if (nit > 500) {
			//pw->makeNoise();
			//this->setWeights(pw);
		//	this->bfgsOptim(training, predicted);
		//	return;
		//}

		// Tadaa...
		///cout << "\tCompleted " << nit << " iterations of BFGS" << endl << endl;

	}
	cout << "Completed " << nit << " iterations of BFGS" << endl << endl;
}

/*
	Used for testing purposes,
	prints the values of a through
	forward propagation
*/
void NeuralNetwork::printA(Matrix* x) {
	Matrix* fin = new Matrix(x->getRows(), x->getColumns());
	*fin = *x;
	for (int i = 0; i < this->weightmap.size(); i++) {
		fin = fin->vectorMult(weightmap[i]);
		fin = this->activation(fin);
		cout << "Printing a " << i << " ";
		fin->printMatrix();
	}
}


