#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <thread>

using namespace std;
using namespace Eigen;

class MyData
{
public:
	vector<VectorXd> dataBody;
	VectorXd Mean;
	VectorXd Sig;
	MatrixXd Cov;
	MatrixXd CoR;

	MyData(int dim)
	{
		this->dim = dim;
		Mean.resize(dim), Mean.setZero();
		Cov.resize(dim, dim), Cov.setZero();
		Sig.resize(dim), Sig.setZero();
		CoR.resize(dim, dim), CoR.setZero();
	}
	void changeDim(int dim)
	{
		this->dim = dim;
		Mean.resize(dim), Mean.setZero();
		Cov.resize(dim, dim), Cov.setZero();
		Sig.resize(dim), Sig.setZero();
		CoR.resize(dim, dim), CoR.setZero();
	}
	void readData(const char* path)
	{
		fstream fs(path, ios::in);
		while (1)
		{
			VectorXd buf(dim);
			for (auto& D : buf)
			{
				fs >> D;
			}
			if (fs.eof()) break;
			dataBody.push_back(buf);
		}
		fs.close();
		init();
	}
	void wirteData(const char* path)
	{
		fstream fs(path, ios::out);
		for (auto& V : dataBody)
		{
			for (auto& D : V)
			{
				fs << " " << D;
			}
			fs << endl;
		}
		fs.close();
	}
	void init()
	{
		Mean.setZero(); Cov.setZero();
		calcMean();
		calcCov();
		calcSigma();
		calcCoR();
	}

private:
	int dim;
	void calcMean()
	{
		Mean.setZero();
		for (auto& v : dataBody)
		{
			Mean += v;
		}
		Mean /= dataBody.size();
	}
	void calcCov()
	{
		Cov.setZero();
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				for (auto& v : dataBody)
				{
					Cov(i, j) += (v(i) - Mean(i)) * (v(j) - Mean(j));
				}
			}
		}
		Cov /= dataBody.size();
	}
	void calcSigma()
	{
		Sig.setZero();
		for (int i = 0; i < dim; i++)
		{
			Sig(i) = sqrt(Cov(i, i));
		}
	}
	void calcCoR()
	{
		CoR.setZero();
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				CoR(i, j) = Cov(i, j) / Sig(i) / Sig(j);
			}
		}
	}
};
class Gauss2D {
public:
	Gauss2D() {
		S.Identity();
		Si.Identity();
		det = 1.0;
		mu.Zero();
	};
	Gauss2D(Matrix2d SS) {
		S = SS;
		Si = S.inverse();
		det = S.determinant();
		mu.Zero();
	}
	Gauss2D(Matrix2d SS, Vector2d mumu) {
		S = SS;
		Si = S.inverse();
		det = S.determinant();
		mu = mumu;
	}
	Matrix2d S, Si;
	Vector2d mu;
	const double pi = 3.141592;
	double det;
	double dist2d(const double x1, const double x2) {
		Vector2d x2d = Vector2d(x1, x2);
		float ex = (double)((x2d - mu).transpose() * Si * (x2d - mu));
		return exp(-ex / 2.0) / 2.0 / pi / sqrt(det);
	}
};

int main(int argc, char* argv[])
{
	MyData cloudData(3);
	EigenSolver<MatrixXd> es;
	vector<thread> tDrawer;
	
	cout << "Read Cloud Data..." << endl << endl;
	cloudData.readData("planeCloud.txt");

	cout << "Mean of cloud : " << endl;
	cout << cloudData.Mean << endl << endl;
	cout << "Covariance of cloud : " << endl;
	cout << cloudData.Cov << endl << endl;

	es.compute(cloudData.Cov);
	cout << "EigenValues of cloud : " << endl;
	cout << es.eigenvalues() << endl << endl;
	cout << "EigenVectors of cloud : " << endl;
	cout << es.eigenvectors() << endl << endl;

	cout << "Plot Cloud With Eigen Vectors..." << endl << endl;

	fstream fio("CloudEigenValue.txt", ios::out);
	fio << es.eigenvalues().real() << endl;
	fio.close();
	fio.open("CloudEigenVectors.txt", ios::out);
	fio << es.eigenvectors().real() << endl;
	fio.close();
	fio.open("CloudMeans.txt", ios::out);
	fio << cloudData.Mean << endl;
	fio.close();

	tDrawer.push_back(thread([] {system("python ..\\Covariance_Plotter\\cloudPlot.py"); }));

	cout << "Change Data into 2D..." << endl << endl;

	MyData cloud2D(2);
	for (auto& V : cloudData.dataBody)
	{
		VectorXd buf(2);
		buf(0) = - V.dot(es.eigenvectors().real().col(0));
		// 2nd Vector is Negligible
		buf(1) = V.dot(es.eigenvectors().real().col(2));
		cloud2D.dataBody.push_back(buf);
	}
	cloud2D.init();
	cloud2D.wirteData("cloud2D.txt");

	cout << "Plotting 2D converted Data..." << endl << endl;
	tDrawer.push_back(thread([] {system("python ..\\Covariance_Plotter\\cloud2Dplot.py"); }));

	Gauss2D gaussian(cloud2D.Cov, cloud2D.Mean);

	cout << "Mean of cloud 2D : " << endl;
	cout << cloud2D.Mean << endl << endl;
	cout << "Covariance of cloud 2D : " << endl;
	cout << cloud2D.Cov << endl << endl;


	cout << "Plotting Gaussian Graph..." << endl << endl;
	fio.open("cloudGauss.txt", ios::out);
	for (double i = 50.; i < 249.999999; i += 0.1)
	{
		for (double j = -50.; j < 50; j += 0.1)
		{
			fio << i << " " << j << " " << gaussian.dist2d(i, j) << endl;
		}
	}
	fio.close();
	tDrawer.push_back(thread([] {system("python ..\\Covariance_Plotter\\gauss2Dplot.py"); }));

	cout << "Reading Iris Data..." << endl << endl;
	MyData irisData(5);
	irisData.readData("iris.txt");
	cout << irisData.Mean << endl << endl;
	cout << irisData.Cov << endl << endl;

	MyData Iris(4);
	for (auto& V : irisData.dataBody)
	{
		VectorXd buf(4);
		buf(0) = V(0), buf(1) = V(1), buf(2) = V(2), buf(3) = V(3);
		Iris.dataBody.push_back(buf);
	}
	Iris.init();
	
	cout << endl;

	cout << "Mean of iris : " << endl;
	cout << Iris.Mean << endl << endl;
	cout << "Covariance of iris : " << endl;
	cout << Iris.Cov << endl << endl;

	es.compute(Iris.Cov);
	cout << "EigenValues of iris : " << endl;
	cout << es.eigenvalues() << endl;
	cout << "EigenVectors of iris : " << endl;
	cout << es.eigenvectors() << endl << endl;

	cout << "Coverting Iris Data with the principle component..." << endl << endl;
	MyData iris2D(2);
	for (auto& V : Iris.dataBody)
	{
		VectorXd buf(2);
		buf(0) = -V.dot(es.eigenvectors().real().col(0));
		buf(1) = V.dot(es.eigenvectors().real().col(1));
		// 3rd, 4th Vector is Negligible
		iris2D.dataBody.push_back(buf);
	}
	iris2D.init();
	MyData iris2D_withClass(3);
	for (int i = 0; i < iris2D.dataBody.size(); i++)
	{
		VectorXd buf(3);
		buf(0) = iris2D.dataBody[i](0);
		buf(1) = iris2D.dataBody[i](1);
		buf(2) = irisData.dataBody[i](4);

		iris2D_withClass.dataBody.push_back(buf);
	}


	cout << "Plot Iris Data..." << endl << endl;
	iris2D_withClass.wirteData("iris2D.txt");
	tDrawer.push_back(thread([] {system("python ..\\Covariance_Plotter\\iris2Dplot.py"); }));


	cout << "Reading Stock market Data..." << endl << endl;
	MyData stockTMP(6), stockData(12);
	stockTMP.readData("6_Portfolios_2x3_weekly.csv");
	for (int j = 0; j < (stockTMP.dataBody.size() - 1); j++)
	{
		VectorXd buf(12);
		for (int i = 0; i < 6; i++)
		{
			buf(i + 0) = stockTMP.dataBody[j + 0](i);
			buf(i + 6) = stockTMP.dataBody[j + 1](i);
		}
		stockData.dataBody.push_back(buf);
	}


	cout << "Correlation Between Stockdata & future data..." << endl << endl;
	stockData.init();
	cout << stockData.CoR << endl << endl;

	cout << "Terminating...";
	for (auto& T : tDrawer) 
	{
		T.join();
	}
	cout << "Done!" << endl;
	return 0;
}