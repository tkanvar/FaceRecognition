#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "core.hpp"
#include <string>
#include <stack>
#include <windows.h>
#include <string>
#include <vector>
#include <stack>
#include <iostream>
#include <atlstr.h> 
#include <fstream>

using namespace std;
using namespace cv;

#define THRESHOLD_VARIATION 0.8						// Percentage of sigma considered relevant
#define TVSVD 3000
#define TVLDA 2100

void svd(int noOfTrainingData, string traningData, string testingData);
void fisherDiscriminant(int noOfTrainingData, string traningData, string testingData);

int main(int argc, char** argv) 
{
	int noOfTrainingData = stoi(argv[1]);
    string trainingData = argv[2];
	string testingData = argv[3];

	svd(noOfTrainingData, trainingData, testingData);
	fisherDiscriminant(noOfTrainingData, trainingData, testingData);
    
    return 0;
}

void getDirectoriesAndFiles(int noOfDir, string path, vector<string> & files, vector<int> & labels, int directoryNo)
{
	string mask = "*";
	HANDLE hFind = INVALID_HANDLE_VALUE;
    WIN32_FIND_DATA ffd;
    string spec;
    vector<string> directories;

    spec = path + "\\" + mask;

    hFind = FindFirstFile(spec.c_str(), &ffd);
    if (hFind == INVALID_HANDLE_VALUE)  {
        return;
    } 

    do {
        if (strcmp(ffd.cFileName, ".") != 0 && 
            strcmp(ffd.cFileName, "..") != 0) 
		{
            if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                directories.push_back(path + "\\" + ffd.cFileName);
            }
			else
			{
				files.push_back(path + "\\" + ffd.cFileName);
				labels.push_back(directoryNo);
			}
        }
    } while ((noOfDir == -1 || directories.size() < noOfDir) && FindNextFile(hFind, &ffd) != 0);

    FindClose(hFind);
    hFind = INVALID_HANDLE_VALUE;

	for (int i = 0; i < directories.size(); i++)
	{
		getDirectoriesAndFiles(-1, directories[i], files, labels, i);
	}
}

void getDirectoriesAndFiles(int noOfDir, string path, vector<string> & files)
{
	vector<int> label;
	getDirectoriesAndFiles(noOfDir, path, files, label, -1);
}

void matrixMultiplication(float ** mat1, float ** mat2, float ** mat3, int dim1, int dim2, int dim3)
{
	for (int i = 0; i < dim1; i++)
	{
		for (int j = 0; j < dim3; j++)
		{
			mat3[i][j] = 0;
		}
	}

	for(int i = 0; i < dim1; ++i)
	{
        for(int j = 0; j < dim3; ++j)
		{
            for(int k = 0; k < dim2; ++k)
            {
                float val = mat3[i][j];
				val += mat1[i][k] * mat2[k][j];
				mat3[i][j] = val;
            }
		}
	}
}

void matrixAddition(float ** mat1, float ** mat2, float ** mat3, int dim1, int dim3)
{
	for(int i = 0; i < dim1; ++i)
	{
        for(int j = 0; j < dim3; ++j)
		{
			mat3[i][j] = mat1[i][j] + mat2[i][j];
		}
	}
}

void matrixSubtraction(float ** mat1, float ** mat2, float ** mat3, int dim1, int dim3)
{
	for(int i = 0; i < dim1; ++i)
	{
        for(int j = 0; j < dim3; ++j)
		{
			mat3[i][j] = mat1[i][j] - mat2[i][j];
		}
	}
}

void matrixScalarDivision(float ** mat1, float val, int dim1, int dim2)
{
	for (int i = 0; i < dim1; i++)
	{
		for (int j = 0; j < dim2; j++)
		{
			mat1[i][j] /= val;
		}
	}
}

void matrixTranspose(float ** mat1, float ** mat2, int dim1, int dim3)
{
	for(int i = 0; i < dim1; ++i)
	{
        for(int j = 0; j < dim3; ++j)
		{
			mat2[j][i] = mat1[i][j];
		}
	}
}

void matrixCopy(float ** mat1, float ** mat2, int col, int row, int dim1)
{
	for (int i = 0; i < dim1; i++)
	{
		if (col != -1)
		{
			mat2[i][0] = mat1[i][col];
		}
		else if (row != -1)
		{
			mat2[0][i] = mat1[row][i];
		}
	}
}

void matrixDelete(float ** mat1, int dim1, int dim2)
{
	for (int i = 0; i < dim1; i++)
	{
		delete[] mat1[i];
	}
	delete[] mat1;
}

float ** createFloatMatrix(int dim1, int dim2, float defaultValue=0)
{
	float ** n = new float * [dim1];
	for (int i = 0; i < dim1; i++)
	{
		n[i] = new float[dim2];
		for (int j = 0; j < dim2; j++)
		{
			n[i][j] = defaultValue;
		}
	}

	return n;
}

void svd(int noOfTrainingData, string traningData, string testingData)
{
	// Get Files and Directories
	vector<string> trainingFiles;
	getDirectoriesAndFiles(noOfTrainingData, traningData, trainingFiles);

	vector<string> testingFiles;
	getDirectoriesAndFiles(-1, testingData, testingFiles);

	////////// TRAIN THE RECOGNIZER ////////////

	Mat image = imread(trainingFiles[0], CV_LOAD_IMAGE_GRAYSCALE);
	int noOfCols = trainingFiles.size();
	int noOfRows = image.rows * image.cols;
	float ** trainingSetMat = new float * [noOfRows];
	for (int i = 0; i < noOfRows; i++)
	{
		trainingSetMat[i] = new float[noOfCols];
	}

	int noOfimageRows, noOfimageCols;
	noOfimageCols = image.cols;
	noOfimageRows = image.rows;

	for (int i = 0; i < trainingFiles.size(); i++)
	{
		string filePath = trainingFiles[i];
		image = imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
		
		if(!image.data )                              // Check for invalid input
		{
			cout <<  "Could not open or find the image" << std::endl ;
			return;
		} 

		// Arrange image in Nx1 column vectors
		for (int j = 0; j < image.rows; j++)
		{
			for (int k = 0; k < image.cols; k++)
			{
				trainingSetMat[j * image.cols + k][i] = image.at<uchar>(j,k);
			}
		}

		imshow("image", image);
		waitKey(200);
	}

	cout << "Recognizer training Started\n";

	// Find mean of training data
	float ** meanMat = new float * [noOfRows];
	for (int i = 0; i < noOfRows; i++)
	{
		meanMat[i] = new float[1];
	}

	for (int i = 0; i < noOfRows; i++)
	{
		float mean = 0;
		for (int j = 0; j < noOfCols; j++)
		{
			mean += trainingSetMat[i][j];
		}
		mean = mean / noOfCols;

		meanMat[i][0] = mean;
	}

	// Subtract mean from all training data
	for (int i = 0; i < noOfRows; i++)
	{
		for (int j = 0; j < noOfCols; j++)
		{
			trainingSetMat[i][j] = trainingSetMat[i][j] - meanMat[i][0];
		}
	}

	// Display mean subtracted data
	for (int j = 0; j < noOfRows; j++)
	{
		int imageR = j / noOfimageCols;
		int imageC = j % noOfimageCols;
		image.at<uchar>(imageR, imageC) = meanMat[j][0];
	}
	imshow("MeanSubImage", image);
	waitKey(200);

	// Apply SVD on trainingSetMat
	Mat data = Mat::zeros(noOfRows, noOfCols, CV_32F);
	for (int i = 0; i < noOfRows; i++)
	{
		for (int j = 0; j < noOfCols; j++)
		{
			data.at<float>(i,j) = trainingSetMat[i][j];
		}
	}

	Mat uData;// = Mat::zeros(noOfRows, noOfRows, CV_32F);
	Mat sigmaData;// = Mat::zeros(noOfRows, noOfCols, CV_32F);
	Mat vtData;// = Mat::zeros(noOfCols, noOfCols, CV_32F);
	SVD::compute(data, sigmaData, uData, vtData);

	// Select Singular Value Range
	int noOfReducedCols = 0;
	float sigVal = 0, totalSigVal = 0;
	for (int i = 0; i < sigmaData.rows; i++)
	{
		totalSigVal += sigmaData.at<float>(i,0);
	}
	for (int i = 0; i < sigmaData.rows; i++, noOfReducedCols++)
	{
		sigVal += sigmaData.at<float>(i,0);
		if ((sigVal / totalSigVal) > THRESHOLD_VARIATION)
		{
			break;
		}
	}

	// Create new uData transpose matrix using noOfReducedCols
	float ** uDataReducedTrns = new float*[noOfReducedCols];
	for (int i = 0; i < noOfReducedCols; i++)
	{
		uDataReducedTrns[i] = new float[noOfRows];
	}
	for (int i = 0; i < noOfRows; i++)
	{
		for (int j = 0; j < noOfReducedCols; j++)
		{
			uDataReducedTrns[j][i] = uData.at<float>(i,j);
		}
	}

	/////////// DISPLAY NEW REDUCED IMAGE ///////////////

	// Construct new sigmaData matrix using noOfCols
	float ** sigmaDataReduced = new float * [noOfReducedCols];
	for (int i = 0; i < noOfReducedCols; i++)
	{
		sigmaDataReduced[i] = new float[noOfCols];
	}
	for (int i = 0; i < noOfReducedCols; i++)
	{
		for (int j = 0; j < noOfCols; j++)
		{
			if (i == j)
			{
				sigmaDataReduced[i][j] = sigmaData.at<float>(i,0);
			}
			else
			{
				sigmaDataReduced[i][j] = 0;
			}
		}
	}

	// Construct new uData matrix using noOfCols
	float ** uDataReduced = new float * [noOfRows];
	for (int i = 0; i < noOfRows; i++)
	{
		uDataReduced[i] = new float[noOfReducedCols];
	}
	for (int i = 0; i < noOfRows; i++)
	{
		for (int j = 0; j < noOfReducedCols; j++)
		{
			uDataReduced[i][j] = uData.at<float>(i,j);
		}
	}

	// Display reduced images
	float ** intermedImage = new float * [noOfReducedCols];
	for (int i = 0; i < noOfReducedCols; i++)
	{
		intermedImage[i] = new float[noOfCols];
	}
	float ** finalNewTrainingData = new float * [noOfRows];
	for (int i = 0; i < noOfRows; i++)
	{
		finalNewTrainingData[i] = new float[noOfCols];
	}
	float ** vtDataMat = new float * [noOfCols];
	for (int i = 0; i < noOfCols; i++)
	{
		vtDataMat[i] = new float[noOfCols];
	}
	for (int i = 0; i < noOfCols; i++)
	{
		for (int j = 0; j < noOfCols; j++)
		{
			vtDataMat[i][j] = vtData.at<float>(i,j);
		}
	}
	matrixMultiplication(sigmaDataReduced, vtDataMat, intermedImage, noOfReducedCols, noOfCols, noOfCols);
	matrixMultiplication(uDataReduced, intermedImage, finalNewTrainingData, noOfRows, noOfReducedCols, noOfCols);

	float minVal = 0;
	for (int i = 0; i < noOfCols; i++)
	{
		for (int j = 0; j < noOfRows; j++)
		{
			if (minVal > finalNewTrainingData[j][i])
			{
				minVal = finalNewTrainingData[j][i];
			}
		}
	}
	for (int i = 0; i < noOfCols; i++)
	{
		for (int j = 0; j < noOfRows; j++)
		{
			int imageR = j / noOfimageCols;
			int imageC = j % noOfimageCols;
			image.at<uchar>(imageR, imageC) = -1 * minVal + finalNewTrainingData[j][i];
		}
		//imshow("MeanSubImage", image);
		//waitKey(200);
	}

	// Multiply uDataReducedTrns and trainingSetMat - Ut * A = Sigma * Vt
	float ** sigmaVtMat = new float * [noOfReducedCols];
	for (int i = 0; i < noOfReducedCols; i++)
	{
		sigmaVtMat[i] = new float[noOfCols];
	}
	matrixMultiplication(uDataReducedTrns, trainingSetMat, sigmaVtMat, noOfReducedCols, noOfRows, noOfCols);

	// Divide data into clusters using k-means
	Mat trainingDataKMeans(noOfCols, noOfReducedCols, CV_32F);
	for (int i = 0; i < noOfCols; i++)
	{
		for (int j = 0; j < noOfReducedCols; j++)
		{
			trainingDataKMeans.at<float>(i,j) = sigmaVtMat[j][i];
		}
	}

	int clusterCount = noOfTrainingData;
	Mat trainingLabelsMat;
	Mat trainingCentersMat(clusterCount, noOfReducedCols, trainingDataKMeans.type());

	kmeans(trainingDataKMeans, clusterCount, trainingLabelsMat, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, trainingCentersMat);

	// Cluster as follows
	vector<vector<int> > trainingClustersImageNo;
	trainingClustersImageNo.resize(noOfTrainingData);
	for (int i = 0; i < noOfCols; i++)
	{
		int label = trainingLabelsMat.at<int>(i, 0);
		trainingClustersImageNo[label].push_back(i);
	}
	for (int i = 0; i < trainingClustersImageNo.size(); i++)
	{
		cout << "Cluster: " << i << "\n";

		vector<int> * lTemp = &trainingClustersImageNo[i];
		for (int j = 0; j < lTemp->size(); j++)
		{
			int ImageNo = lTemp->at(j);
			Mat im = imread(trainingFiles[ImageNo]);
			imshow("Cluster", im);
			waitKey(200);
		}

		cout << "Enter any key to continue\n";
		char s;
		cin >> s;
	}

	// Prepare cluster centers
	float ** trainingClusterMean = new float * [noOfReducedCols];
	for (int i = 0; i < noOfReducedCols; i++)
	{
		trainingClusterMean[i] = new float[noOfTrainingData];
	}
	for (int i = 0; i < noOfTrainingData; i++)
	{
		for (int j = 0; j < noOfReducedCols; j++)
		{
			trainingClusterMean[j][i] = trainingCentersMat.at<float>(i,j);
		}
	}

	cout << "Recognizer training ended\n";

	///////////// TESTING THE TESTING DATA ////////////

	for (int testingCntr = 0; testingCntr < testingFiles.size(); testingCntr++)
	{
		string testingFilePath = testingFiles[testingCntr];

		Mat imageTesting = imread(testingFilePath, CV_LOAD_IMAGE_GRAYSCALE);
		int noOfColsTesting = 1;
		int noOfRowsTesting = imageTesting.rows * imageTesting.cols;
		float ** testingSetMat = new float * [noOfRowsTesting];
		for (int i = 0; i < noOfRowsTesting; i++)
		{
			testingSetMat[i] = new float[noOfColsTesting];
		}

		int noOfimageRows, noOfimageCols;
		noOfimageCols = imageTesting.cols;
		noOfimageRows = imageTesting.rows;

		// Arrange image in Nx1 column vectors
		for (int j = 0; j < imageTesting.rows; j++)
		{
			for (int k = 0; k < imageTesting.cols; k++)
			{
				testingSetMat[j * imageTesting.cols + k][0] = imageTesting.at<uchar>(j,k);
			}
		}

		imshow("image", imageTesting);
		waitKey(200);

		// Subtract mean from all testing data
		for (int i = 0; i < noOfRowsTesting; i++)
		{
			testingSetMat[i][0] = testingSetMat[i][0] - meanMat[i][0];
		}

		// Display Diff image
		float minValue = 0;
		for (int i = 0; i < noOfRowsTesting; i++)
		{
			if (minValue > testingSetMat[i][0])
			{
				minVal = testingSetMat[i][0];
			}
		}
		for (int j = 0; j < noOfRowsTesting; j++)
		{
			int imageR = j / noOfimageCols;
			int imageC = j % noOfimageCols;
			image.at<uchar>(imageR, imageC) = -1 * minValue + testingSetMat[j][0];
		}
		//imshow("DiffImage", image);
		//waitKey(200);

		// Multiply uDataReducedTrns and testingSetMat - Ut * A = Sigma * Vt
		float ** sigmaVtMatTesting = new float * [noOfReducedCols];
		for (int i = 0; i < noOfReducedCols; i++)
		{
			sigmaVtMatTesting[i] = new float[noOfColsTesting];
		}
		matrixMultiplication(uDataReducedTrns, testingSetMat, sigmaVtMatTesting, noOfReducedCols, noOfRowsTesting, noOfColsTesting);

		int clusterClosestToTestingImage = -1;
		float minVariance = 1000000;
		for (int j = 0; j < noOfTrainingData; j++)
		{
			// Find Diff matrix
			float ** diffMat = new float * [noOfReducedCols];
			float ** diffTransMat = new float * [1];
			for (int l = 0; l < noOfReducedCols; l++)
			{
				diffMat[l] = new float[1];
			}
			for (int l = 0; l < 1; l++)
			{
				diffTransMat[l] = new float[noOfReducedCols];
			}
			for (int l = 0; l < noOfReducedCols; l++)
			{
				diffMat[l][0] = trainingClusterMean[l][j] - sigmaVtMatTesting[l][0];
				diffTransMat[0][l] = diffMat[l][0];
			}
		
			// Find Diagonal matrix / covariance matrix between the testing image and current training image and check if the minimum value along the diagonal is less than THRESHOLD_SIMILARITY
			float ** covarianceMat = new float * [1];
			covarianceMat[0] = new float[1];
			matrixMultiplication(diffTransMat, diffMat, covarianceMat, 1, noOfReducedCols, 1);

			// Sqr root of diagonal elements
			covarianceMat[0][0] = sqrt(covarianceMat[0][0]);

			// Compare
			if (minVariance > covarianceMat[0][0])
			{
				minVariance = covarianceMat[0][0];
				clusterClosestToTestingImage = j;
			}

			delete[] covarianceMat[0];
			delete[] covarianceMat;

			for (int l = 0; l < noOfReducedCols; l++)
			{
				delete[] diffMat[l];
			}
			delete[] diffMat;

			delete[] diffTransMat[0];
			delete[] diffTransMat;
		}

		// Display cluster images
		if (clusterClosestToTestingImage != -1 && minVariance < TVSVD)
		{
			vector<int> * lTemp = &trainingClustersImageNo[clusterClosestToTestingImage];
			for (int l = 0; l < lTemp->size(); l++)
			{
				string filePath = trainingFiles[lTemp->at(l)];
				image = imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
				imshow("Found", image);
				waitKey(0);
			}
		}
		else
		{
			cout << "No matching image\n";
		}
	
		char s;
		cout << "Press any key to continue: ";
		cin >> s;

		// Delete matrixes
		for (int l = 0; l < noOfReducedCols; l++)
		{
			delete[] sigmaVtMatTesting[l];
		}
		delete[] sigmaVtMatTesting;
	}

	for (int l = 0; l < noOfRows; l++)
	{
		delete[] meanMat[l];
		delete[] trainingSetMat[l];
	}
	delete[] trainingSetMat;
	delete[] meanMat;

	for (int l = 0; l < noOfReducedCols; l++)
	{
		delete[] sigmaVtMat[l];
		delete[] uDataReducedTrns[l];
	}
	delete[] uDataReducedTrns;
	delete[] sigmaVtMat;

	for (int l = 0; l < noOfReducedCols; l++)
	{
		delete[] trainingClusterMean[l];
	}
	delete[] trainingClusterMean;

	char s;
	cout << "Press any key to continue to fisher: ";
	cin >> s;
}

void fisherDiscriminant(int noOfTrainingData, string trainingData, string testingData)
{
	// Get Files and Directories
	vector<string> trainingFiles;
	vector<int> trainingLabel;
	getDirectoriesAndFiles(noOfTrainingData, trainingData, trainingFiles, trainingLabel, -1);

	vector<string> testingFiles;
	getDirectoriesAndFiles(-1, testingData, testingFiles);

	////////// TRAIN THE RECOGNIZER ////////////

	Mat image = imread(trainingFiles[0], CV_LOAD_IMAGE_GRAYSCALE);
	int noOfRows = image.rows * image.cols;
	int noOfCols = trainingFiles.size();

	float ** trainingSet = new float * [noOfRows];
	for (int i = 0; i < noOfRows; i++)
	{
		trainingSet[i] = new float[noOfCols];
	}

	for (int i = 0; i < trainingFiles.size(); i++)
	{
		string filePath = trainingFiles[i];
		Mat image = imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);

		if (!image.data )                              // Check for invalid input
		{
			cout <<  "Could not open or find the image" << std::endl ;
			return;
		} 

		for (int j = 0; j < image.rows; j++)
		{
			for (int k = 0; k < image.cols; k++)
			{
				trainingSet[j * image.cols + k][i] = image.at<uchar>(j,k);
			}
		}

		imshow("image", image);
		waitKey(200);
	}

	// Calculate PCA of training data
	Mat data(noOfCols, noOfRows, CV_32F);
	for (int i = 0; i < noOfCols; i++)
	{
		for (int j = 0; j < noOfRows; j++)
		{
			data.at<float>(i,j) = trainingSet[j][i];
		}
	}
	PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, (int)trainingFiles.size());

	Mat eigenvecs = pca.eigenvectors.clone();
	float ** trainingEigenVecsTrns = createFloatMatrix(noOfCols, noOfRows);
	for (int i = 0; i < noOfRows; i++)
	{
		for (int j = 0; j < noOfCols; j++)
		{
			trainingEigenVecsTrns[j][i] = eigenvecs.at<float>(j,i);
		}
	}
	
	float ** trainingProjSet = createFloatMatrix(noOfCols, noOfCols);
	matrixMultiplication(trainingEigenVecsTrns, trainingSet, trainingProjSet, noOfCols, noOfRows, noOfCols);

	// LDA
	// Calculate mean of different trainingProjSet
	float ** trainingMean = createFloatMatrix(noOfCols, noOfTrainingData);

	int processingLabel = trainingLabel[0]-1;
	vector<int> noOfEleInEachSet;

	for (int i = 0, meanColCntr = -1; i <= trainingLabel.size(); i++)
	{
		if (i == trainingLabel.size() || trainingLabel[i] != processingLabel)
		{
			// Calculate mean till now
			if (meanColCntr != -1)
			{
				for (int j = 0; j < noOfCols; j++)
				{
					trainingMean[j][meanColCntr] /= noOfEleInEachSet[meanColCntr];
				}
			}

			// Prepare mean for new calculations
			if (i != trainingLabel.size())
			{
				meanColCntr++;

				for (int j = 0; j < noOfCols; j++)
				{
					trainingMean[j][meanColCntr] = 0;
				}
			
				noOfEleInEachSet.push_back(0);
				processingLabel = trainingLabel[i];
			}
			else
			{
				break;
			}
		}

		for (int j = 0; j < noOfCols; j++)
		{
			trainingMean[j][meanColCntr] += trainingProjSet[i][j];
		}
		noOfEleInEachSet[meanColCntr]++;
	}

	// Cluster as follows
	vector<vector<int> > trainingClustersImageNo;
	trainingClustersImageNo.resize(noOfTrainingData);
	int fileNo = 0;
	for (int i = 0; i < noOfTrainingData; i++)
	{
		for (int j = 0; j < noOfEleInEachSet[i]; j++, fileNo++)
		{
			trainingClustersImageNo[i].push_back(fileNo);
		}
	}

	// Calculate total mean
	float ** totalTrainingMean = new float * [noOfCols];
	for (int i = 0; i < noOfCols; i++)
	{
		totalTrainingMean[i] = new float[1];
		totalTrainingMean[i][0] = 0;
	}

	for (int i = 0; i < noOfCols; i++)
	{
		for(int j = 0; j < noOfTrainingData; j++)
		{
			totalTrainingMean[i][0] += trainingMean[i][j];
		}
		totalTrainingMean[i][0] /= noOfTrainingData;
	}

	// Calculate Sw
	float ** Sw = createFloatMatrix(noOfCols, noOfCols);
	float ** SwTemp = createFloatMatrix(noOfCols, noOfCols);
	float ** curMean = createFloatMatrix(noOfCols, 1);

	matrixCopy(trainingMean, curMean, 0, -1, noOfCols);

	processingLabel = trainingLabel[0];
	for (int i = 0, meanCntr = 0; i < noOfCols; i++)
	{
		if (processingLabel != trainingLabel[i])
		{
			processingLabel = trainingLabel[i];

			matrixScalarDivision(SwTemp, noOfEleInEachSet[meanCntr], noOfCols, noOfCols);
			matrixAddition(SwTemp, Sw, Sw, noOfCols, noOfCols);

			matrixCopy(trainingMean, curMean, meanCntr, -1, noOfCols);
			meanCntr++;
		}

		float ** curSet = createFloatMatrix(noOfCols, 1);
		matrixCopy(trainingProjSet, curSet, i, -1, noOfCols);

		float ** curSetTrns = createFloatMatrix(1, noOfCols);
		matrixSubtraction(curSet, curMean, curSet, noOfCols, 1);
		matrixTranspose(curSet, curSetTrns, noOfCols, 1);

		matrixMultiplication(curSet, curSetTrns, SwTemp, noOfCols, 1, noOfCols);

		matrixDelete(curSet, noOfCols, 1);
		matrixDelete(curSetTrns, 1, noOfCols);
	}

	matrixDelete(SwTemp, noOfCols, noOfCols);
	matrixDelete(curMean, noOfCols, 1);

	// Calculate Sb
	float ** Sb = createFloatMatrix(noOfCols, noOfCols);

	for (int i = 0; i < noOfTrainingData; i++)
	{
		float ** curMean = createFloatMatrix(noOfCols, 1);
		matrixCopy(trainingMean, curMean, i, -1, noOfCols);

		float ** lTemp = createFloatMatrix(noOfCols, 1);
		matrixSubtraction(curMean, totalTrainingMean, lTemp, noOfCols, 1);

		float ** lTempTrns = createFloatMatrix(1, noOfCols);
		matrixTranspose(lTemp, lTempTrns, noOfCols, 1);

		float ** lTemp1 = createFloatMatrix(noOfCols, noOfCols);
		matrixMultiplication(lTemp, lTempTrns, lTemp1, noOfCols, 1, noOfCols);

		matrixAddition(lTemp1, Sb, Sb, noOfCols, noOfCols);

		matrixDelete(curMean, noOfCols, 1);
		matrixDelete(lTemp, noOfCols, 1);
		matrixDelete(lTempTrns, 1, noOfCols);
		matrixDelete(lTemp1, noOfCols, noOfCols);
	}

	// Calculate W
	float ** SwInv = createFloatMatrix(noOfCols, noOfCols);
	Mat SwMat(noOfCols, noOfCols, CV_32F);
	Mat SwInvMat(noOfCols, noOfCols, CV_32F);
	for (int i = 0; i < noOfCols; i++)
	{
		for (int j = 0; j < noOfCols; j++)
		{
			SwMat.at<float>(i,j) = Sw[i][j];
		}
	}
	double det = determinant(SwMat);
	SwInvMat = SwMat.inv(DECOMP_LU);
	for (int i = 0; i < noOfCols; i++)
	{
		for (int j = 0; j < noOfCols; j++)
		{
			SwInv[i][j] = SwInvMat.at<float>(i,j);
		}
	}

	float ** w = createFloatMatrix(noOfCols, noOfCols);
	matrixMultiplication(SwInv, Sb, w, noOfCols, noOfCols, noOfCols);

	// Multiplier to w
	matrixScalarDivision(w, 10000000, noOfCols, noOfCols);

	// Find eigenvectors and eigenvalues of w
	Mat data1(noOfCols, noOfCols, CV_32F);
	for (int i = 0; i < noOfCols; i++)
	{
		for (int j = 0; j < noOfCols; j++)
		{
			data1.at<float>(i,j) = w[j][i];
		}
	}
	PCA pca1(data1, Mat(), CV_PCA_DATA_AS_ROW, noOfCols);

	Mat eigenvecs1 = pca1.eigenvectors.clone();
	Mat eigenvals1 = pca1.eigenvalues.clone();

	// Reduce dim
	int noOfReducedCols = noOfTrainingData - 1;
	float sigVal = 0, totalSigVal = 0;
	for (int i = 0; i < eigenvals1.rows; i++)
	{
		totalSigVal += eigenvals1.at<float>(i,0);
	}
	for (int i = 0; i < eigenvals1.rows; i++, noOfReducedCols++)
	{
		sigVal += eigenvals1.at<float>(i,0);
		if ((sigVal / totalSigVal) > THRESHOLD_VARIATION)
		{
			break;
		}
	}

	float ** wOpt = createFloatMatrix(noOfCols, noOfReducedCols);
	for (int i = 0; i < noOfCols; i++)
	{
		for (int j = 0; j < noOfReducedCols; j++)
		{
			wOpt[i][j] = eigenvecs1.at<float>(i,j);
		}
	}

	// Project mean points on wOpt
	float ** trainingProjectMean = createFloatMatrix(noOfReducedCols, noOfTrainingData);
	float ** wOptTrns = createFloatMatrix(noOfReducedCols, noOfCols);
	matrixTranspose(wOpt, wOptTrns, noOfCols, noOfReducedCols);
	matrixMultiplication(wOptTrns, trainingMean, trainingProjectMean, noOfReducedCols, noOfCols, noOfTrainingData);

	///////////// TESTING THE TESTING DATA ////////////

	for (int testingCntr = 0; testingCntr < testingFiles.size(); testingCntr++)
	{
		string testingFilePath = testingFiles[testingCntr];

		Mat imageTesting = imread(testingFilePath, CV_LOAD_IMAGE_GRAYSCALE);
		int noOfColsTesting = 1;
		int noOfRowsTesting = imageTesting.rows * imageTesting.cols;
		float ** testingSetMat = new float * [noOfRowsTesting];
		for (int i = 0; i < noOfRowsTesting; i++)
		{
			testingSetMat[i] = new float[noOfColsTesting];
		}

		int noOfimageRows, noOfimageCols;
		noOfimageCols = imageTesting.cols;
		noOfimageRows = imageTesting.rows;

		// Arrange image in Nx1 column vectors
		for (int j = 0; j < imageTesting.rows; j++)
		{
			for (int k = 0; k < imageTesting.cols; k++)
			{
				testingSetMat[j * imageTesting.cols + k][0] = imageTesting.at<uchar>(j,k);
			}
		}

		imshow("image", imageTesting);
		waitKey(200);

		// Multiply trainingEigenVecsTrns and testingSetMat
		float ** testingProjSet1 = createFloatMatrix(noOfCols, 1);
		matrixMultiplication(trainingEigenVecsTrns, testingSetMat, testingProjSet1, noOfCols, noOfRows, 1);

		float ** testingProjSet = createFloatMatrix(noOfReducedCols, 1);
		matrixMultiplication(wOptTrns, testingProjSet1, testingProjSet, noOfReducedCols, noOfCols, 1);

		int clusterClosestToTestingImage = -1;
		float minVariance = 0;
		for (int j = 0; j < noOfTrainingData; j++)
		{
			// Find Diff matrix
			float ** diffMat = new float * [noOfReducedCols];
			float ** diffTransMat = new float * [1];
			for (int l = 0; l < noOfReducedCols; l++)
			{
				diffMat[l] = new float[1];
			}
			for (int l = 0; l < 1; l++)
			{
				diffTransMat[l] = new float[noOfReducedCols];
			}
			for (int l = 0; l < noOfReducedCols; l++)
			{
				diffMat[l][0] = trainingProjectMean[l][j] - testingProjSet[l][0];
				diffTransMat[0][l] = diffMat[l][0];
			}
		
			// Find Diagonal matrix / covariance matrix between the testing image and current training image and check if the minimum value along the diagonal is less than THRESHOLD_SIMILARITY
			float ** covarianceMat = new float * [1];
			covarianceMat[0] = new float[1];
			matrixMultiplication(diffTransMat, diffMat, covarianceMat, 1, noOfReducedCols, 1);

			// Sqr root of diagonal elements
			covarianceMat[0][0] = sqrt(covarianceMat[0][0]);

			// Compare
			//cout << "Covariance: " << covarianceMat[0][0] << "\n";
			if (minVariance < covarianceMat[0][0] && (covarianceMat[0][0] < TVLDA))
			{
				minVariance = covarianceMat[0][0];
				clusterClosestToTestingImage = j;
			}

			delete[] covarianceMat[0];
			delete[] covarianceMat;

			for (int l = 0; l < noOfReducedCols; l++)
			{
				delete[] diffMat[l];
			}
			delete[] diffMat;

			delete[] diffTransMat[0];
			delete[] diffTransMat;
		}

		for (int j = 0; j < noOfTrainingData; j++)
		{
			// Find Diff matrix
			float ** diffMat = new float * [noOfReducedCols];
			float ** diffTransMat = new float * [1];
			for (int l = 0; l < noOfReducedCols; l++)
			{
				diffMat[l] = new float[1];
			}
			for (int l = 0; l < 1; l++)
			{
				diffTransMat[l] = new float[noOfReducedCols];
			}
			for (int l = 0; l < noOfReducedCols; l++)
			{
				diffMat[l][0] = trainingProjectMean[l][j] - testingProjSet[l][0];
				diffTransMat[0][l] = diffMat[l][0];
			}
		
			// Find Diagonal matrix / covariance matrix between the testing image and current training image and check if the minimum value along the diagonal is less than THRESHOLD_SIMILARITY
			float ** covarianceMat = new float * [1];
			covarianceMat[0] = new float[1];
			matrixMultiplication(diffTransMat, diffMat, covarianceMat, 1, noOfReducedCols, 1);

			// Sqr root of diagonal elements
			covarianceMat[0][0] = sqrt(covarianceMat[0][0]);

			// Comparecout
			//cout << "Covariance: " << covarianceMat[0][0] << "\n";
			if (minVariance < covarianceMat[0][0] && (covarianceMat[0][0] > 2700))
			{
				minVariance = covarianceMat[0][0];
				clusterClosestToTestingImage = j;
			}

			delete[] covarianceMat[0];
			delete[] covarianceMat;

			for (int l = 0; l < noOfReducedCols; l++)
			{
				delete[] diffMat[l];
			}
			delete[] diffMat;

			delete[] diffTransMat[0];
			delete[] diffTransMat;
		}

		// Display cluster images
		if (clusterClosestToTestingImage != -1 )
		{
			vector<int> * lTemp = &trainingClustersImageNo[clusterClosestToTestingImage];
			for (int l = 0; l < lTemp->size(); l++)
			{
				string filePath = trainingFiles[lTemp->at(l)];
				image = imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
				imshow("Found", image);
				waitKey(0);
			}
		}
		else
		{
			cout << "No matching image\n";
		}
	
		char s;
		cout << "Press any key to continue: ";
		cin >> s;
	}

	char s;
	cout << "Press any key to terminate: ";
	cin >> s;
}