#include "bow.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat TrainVocabulary(const vector<string>& filesList,
					const vector<bool>& is_voc,
					const Ptr<FeatureDetector>& keypointsDetector,
					const Ptr<DescriptorExtractor>& descriptorsExtractor,
					int vocSize)
{
	BOWKMeansTrainer voc(vocSize);
	//Mat img;
	//Mat descriptors;
	Mat training_descriptors(1,
		descriptorsExtractor->descriptorSize(), descriptorsExtractor->descriptorType());
	//vector <KeyPoint> keypoints;

	int toTrain = 0;
	for (int i = 0; i < filesList.size(); i++)
		if (is_voc[i]) toTrain++;

	int c = 0;
	#pragma omp parallel for
	for (int i = 0; i < filesList.size(); i++) {
		if (!is_voc[i])
			continue;
		Mat img = imread(filesList[i]);
		Mat descriptors;
		vector <KeyPoint> keypoints;
		keypointsDetector->detect(img, keypoints);
		descriptorsExtractor->compute(img, keypoints, descriptors);
		#pragma omp critical
		{
			voc.add(descriptors);
			printf("Detecting keypoints: %.0f%%\r", (c + 0.0) / toTrain*100.0);
			c++;
		}
	}
	printf("Detecting keypoints: Done!\n");
	printf("Clustering...");
	return voc.cluster();
}

Mat ExtractFeaturesFromImage(	Ptr<FeatureDetector> keypointsDetector,
								Ptr<BOWImgDescriptorExtractor> bowExtractor,
								const string& fileName)
{
	Mat img = imread(fileName), result;
	vector<KeyPoint> keypoints;
	keypointsDetector->detect(img, keypoints);;
	bowExtractor->compute(img, keypoints, result);
	return result;
}

void ExtractTrainData(	const vector<string>& filesList,
						const vector<bool>& isTrain, 
						const Mat& responses,
						const Ptr<FeatureDetector>& keypointsDetector,
						const Ptr<BOWImgDescriptorExtractor>& bowExtractor,
						Mat& trainData,
						Mat& trainResponses)
{
	int trainCount = 0;
	for (int i = 0; i < isTrain.size(); i++)
		if (isTrain[i]) trainCount++;

	trainData = Mat(trainCount, bowExtractor->descriptorSize(), CV_32F);
	trainResponses = Mat(trainCount, 1, CV_32S);

	//Mat result;
	int c = 0;
	#pragma omp parallel for
	for (int i = 0; i < isTrain.size(); i++)
		if (isTrain[i])
		{
			Mat result = ExtractFeaturesFromImage(keypointsDetector, bowExtractor, filesList[i]);
			//trainData.at<Mat>(i) = result;
			#pragma omp critical
			{
				trainData.push_back(result);
				//trainResponses.at<Mat>(i) = responses.row(i);
				trainResponses.push_back(responses.row(i));
			}
			printf("Preparing train set: %.0f%%\r", (++c + 0.0) / trainCount*100.0);
		}
	printf("Preparing train set: Done!\n");
}

Ptr<ml::RTrees> TrainClassifier(const Mat& trainData, const Mat& trainResponses)
{
	Ptr<ml::RTrees> rTrees = ml::RTrees::create();
	rTrees->setMaxDepth(200);
	TermCriteria termCreteria;
	termCreteria.type = TermCriteria::COUNT;
	//rTrees->setTermCriteria(termCreteria);

	Mat types(1, trainData.cols + 1, CV_8U);
	for (int i = 0; i < trainData.cols; i++)
		types.at<uchar>(i) = ml::VAR_ORDERED;
	types.at<uchar>(trainData.cols) = ml::VAR_CATEGORICAL;

	Ptr<ml::TrainData> td = ml::TrainData::create(trainData, ml::ROW_SAMPLE, trainResponses,
		noArray(), noArray(), noArray(), types);

	//Ptr<ml::TrainData> td = ml::TrainData::create(trainData, ml::ROW_SAMPLE, trainResponses);

	//rTrees->train(trainData, ml::ROW_SAMPLE, trainResponses);
	rTrees->train(td);

	return rTrees;
}

Ptr<ml::SVM> TrainSVM(const Mat& trainData, const Mat& trainResponses) {
	Ptr<ml::SVM> svm = ml::SVM::create();
	TermCriteria termCreteria;
	termCreteria.type = TermCriteria::COUNT;
	//svm->setTermCriteria(termCreteria);

	svm->setKernel(ml::SVM::RBF);

	Mat types(1, trainData.cols + 1, CV_8U);
	for (int i = 0; i < trainData.cols; i++)
		types.at<uchar>(i) = ml::VAR_ORDERED;
	types.at<uchar>(trainData.cols) = ml::VAR_CATEGORICAL;

	//Ptr<ml::TrainData> td = ml::TrainData::create(trainData, ml::ROW_SAMPLE, trainResponses);
	Ptr<ml::TrainData> td = ml::TrainData::create(trainData, ml::ROW_SAMPLE, trainResponses,
		noArray(), noArray(), noArray(), types);
	svm->train(td);
	return svm;
}

int Predict(const Ptr<FeatureDetector> keypointsDetector,
			const Ptr<BOWImgDescriptorExtractor> bowExtractor,
			const Ptr<ml::StatModel> classifier,
			const string& fileName)
{
	return classifier->predict(ExtractFeaturesFromImage(keypointsDetector, bowExtractor, fileName));
}

Mat PredictOnTestData(	const vector<string>& filesList, 
						const vector<bool>& isTrain,
						const Ptr<FeatureDetector> keypointsDetector,
						const Ptr<BOWImgDescriptorExtractor> bowExtractor,
						const Ptr<ml::StatModel> classifier,
						vector<int>& posMap)
{
	int testCount = 0;
	for (int i = 0; i < isTrain.size(); i++)
		if (!isTrain[i]) testCount++;

	Mat predictions(testCount, 1, CV_32S);

	int prediction = 0;
	for (int i = 0; i < isTrain.size(); i++)
		if (!isTrain[i]) {
			predictions.at<int>(prediction++) = Predict(keypointsDetector, bowExtractor, classifier, filesList[i]);
			posMap.push_back(prediction - 1);
			printf("Prediction: %.0f%%\r", (prediction + 0.0) / testCount*100.0);
		}
	printf("Prediction: Done!\n");
	return predictions;
}

Mat GetTestResponses(const Mat& responses, const vector<bool>& isTrain)
{
	int testCount = 0;
	for (int i = 0; i < isTrain.size(); i++)
		if (!isTrain[i]) testCount++;

	Mat testResponses(testCount, 1, CV_32S);

	int response = 0;
	for (int i = 0; i < isTrain.size(); i++)
		if (!isTrain[i])
			testResponses.at<int>(response++) = responses.at<int>(i);

	return testResponses;
}

float CalculateMisclassificationError(Mat& responses, Mat& predictions, const std::vector<string>& files, const std::vector<string>& classes, const std::vector<int>& posMap)
{
	float error = 0;
	printf("Misclassified files:\n");
	for (int i = 0; i < responses.rows; i++)
		if (responses.at<int>(i) != predictions.at<int>(i)) {
			error++;
			printf("%s classified as %s\n",
				files[posMap[i]].c_str(), classes[predictions.at<int>(i)].c_str());
		}

	return error * 100 / responses.rows;
}