#include "bow.h"
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

Mat TrainVocabulary(const vector<string>& filesList,
					const vector<bool>& is_voc,
					const Ptr<FeatureDetector>& keypointsDetector,
					const Ptr<DescriptorExtractor>& descriptorsExtractor,
					int vocSize)
{
	
	BOWKMeansTrainer voc(vocSize);
	Mat img, descriptors;
	Mat training_descriptors(1,
		descriptorsExtractor->descriptorSize(), descriptorsExtractor->descriptorType());
	vector <KeyPoint> keypoints;
	for (int i = 0; i < filesList.size(); i++) {
		if (!is_voc[i])
			continue;
		img = imread(filesList[i]);
		keypointsDetector->detect(img, keypoints);
		descriptorsExtractor->compute(img, keypoints, descriptors);
		voc.add(descriptors);
		//training_descriptors.push_back(descriptors);
	}
	//voc.add(training_descriptors);
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

	Mat result;
	for (int i = 0; i < isTrain.size(); i++)
		if (isTrain[i])
		{
			result = ExtractFeaturesFromImage(keypointsDetector, bowExtractor, filesList[i]);
			trainData.push_back(result);
			trainResponses.push_back(responses.row(i));
		}
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
		}
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

float CalculateMisclassificationError(Mat& responses, Mat& predictions, const std::vector<string>& files, const std::vector<int>& posMap)
{
	float error = 0;
	printf("Misclassified files:\n");
	for (int i = 0; i < responses.rows; i++)
		if (responses.at<int>(i) != predictions.at<int>(i)) {
			error++;
			printf("%s\n", files[posMap[i]].c_str());
		}

	return error * 100 / responses.rows;
}