#include "auxiliary.h"
#include "bow.h"
//#include <opencv2\nonfree\nonfree.hpp>
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

const int nDetectors = 10;
const char* detectors[] = { "FAST", "STAR", "SIFT", "SURF",
	"ORB", "MSER", "GFTT", "BRISK", "AKAZE", "SimpleBlob" };
enum { USE_FAST=0, USE_STAR, USE_SIFT, USE_SURF, USE_ORB,
	USE_MSER, USE_GFTT, USE_BRISK, USE_AKAZE, USE_SimpleBlob };
const int nDescriptors = 4;
const char* descriptors[] = { "SIFT", "SURF", "BRIEF", "ORB"};
enum { SIFT_DESC=0, SURF_DESC, BRIEF_DESC, ORB_DESC};

Ptr<FeatureDetector> detector;
Ptr<DescriptorExtractor> descriptor;

Ptr<DescriptorExtractor> getDescriptorExtractor() {
	int choice = -1;
	while (choice < 0 || choice >= nDescriptors) {
		printf("Available descriptors:\n");
		for (int i = 0; i < nDescriptors; i++) {
			printf("%4d - %s\n", i, descriptors[i]);
		}
		printf("Choose descriptor: ");

		scanf("%d", &choice);
	}
	switch (choice)
	{
	case SIFT_DESC: return SIFT::create();
	case SURF_DESC: return SURF::create();
	case BRIEF_DESC: return BriefDescriptorExtractor::create();
	case ORB_DESC: return ORB::create();
	}
}

Ptr<FeatureDetector> getFeatureDetector() {
	int choice = -1;
	while (choice < 0 || choice >= nDetectors) {
		printf("Available detectors:\n");
		for (int i = 0; i < nDetectors; i++) {
			printf("%4d - %s\n", i, detectors[i]);
		}
		printf("Choose detector: ");

		scanf("%d", &choice);
	}
	switch (choice)
	{
	case USE_FAST: return FastFeatureDetector::create();
	case USE_STAR: return StarDetector::create();
	case USE_SIFT: return SIFT::create();
	case USE_SURF: return SURF::create();
	case USE_ORB: return ORB::create();
	case USE_MSER: return MSER::create();
	case USE_GFTT: return GFTTDetector::create();
	case USE_BRISK: return BRISK::create();
	case USE_AKAZE: return AKAZE::create();
	case USE_SimpleBlob: return SimpleBlobDetector::create();
	}
}

int main(int argc, char* argv[])
{
	detector = getFeatureDetector();
	descriptor = getDescriptorExtractor();
	int vocSize = 25;

	Ptr<DescriptorMatcher> descriptorsMatcher =
		DescriptorMatcher::create("BruteForce");

	Ptr<BOWImgDescriptorExtractor> bowExtractor = new
		BOWImgDescriptorExtractor(
			descriptor,
			descriptorsMatcher);

	vector<string> allFiles, crocodile;
	GetFilesInFolder("bikes", allFiles, "jpg");
	int nLeopards = allFiles.size();
	GetFilesInFolder("bycicles", crocodile, "jpg");
	allFiles.insert(allFiles.end(), crocodile.begin(), crocodile.end());
	vector<bool> isTrain(allFiles.size());

	Mat category(allFiles.size(), 1, CV_32S);
	for (int i = 0; i < allFiles.size(); i++)
		category.at<int>(i) = (i < nLeopards ? 1 : -1);

	InitRandomBoolVector(isTrain, 0.7);

	int nTrain = 0;
	for (int i = 0; i < isTrain.size(); i++)
		if (isTrain[i])
			nTrain++;

	//bowExtractor->setVocabulary(TrainVocabulary(allFiles, isTrain, detector, descriptor, vocSize));
	bowExtractor->setVocabulary(TrainVocabulary(allFiles, isTrain, detector, descriptor, nTrain));

	Mat trainData, trainResp;
	ExtractTrainData(allFiles, isTrain, category, detector, bowExtractor, trainData, trainResp);

	Ptr<ml::RTrees> model = TrainClassifier(trainData, trainResp);

	//Ptr<ml::SVM> model = TrainSVM(trainData, trainResp);
	
	//Mat predictions = PredictOnTestData(allFiles, isTrain, detector, bowExtractor, rTree);
	vector<int> posMap;
	Mat predictions = PredictOnTestData(allFiles, isTrain, detector, bowExtractor, model, posMap);

	Mat testResponses = GetTestResponses(category, isTrain);

	float error = CalculateMisclassificationError(testResponses, predictions, allFiles, posMap);

	printf("Total images: %d\n Training set size: %d\n Classification error: %f\n",
		allFiles.size(), nTrain, error);
	return 0;
}