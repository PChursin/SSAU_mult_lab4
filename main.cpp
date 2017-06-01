#include "auxiliary.h"
#include "bow.h"
//#include <opencv2\nonfree\nonfree.hpp>
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

const int nMenu = 3;
const char* menuOptions[] = {"Train model", "Classify images", "Exit"};
const int nDetectors = 9;
const char* detectors[] = { "FAST", "SIFT", "SURF",
	"ORB", "MSER", "GFTT", "BRISK", "AKAZE", "KAZE" };
enum { USE_FAST=0, USE_SIFT, USE_SURF, USE_ORB,
	USE_MSER, USE_GFTT, USE_BRISK, USE_AKAZE, USE_KAZE};
const int nDescriptors = 3;
const char* descriptors[] = { "SIFT", "SURF", "KAZE"};
enum { SIFT_DESC=0, SURF_DESC, KAZE_DESC};

Ptr<FeatureDetector> detector;
Ptr<DescriptorExtractor> descriptor;
Ptr<ml::StatModel> model;
Ptr<BOWImgDescriptorExtractor> bowExtractor;
Mat trainData, trainResp;

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
	case KAZE_DESC: return KAZE::create();
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
	case USE_SIFT: return SIFT::create();
	case USE_SURF: return SURF::create();
	case USE_ORB: return ORB::create();
	case USE_MSER: return MSER::create();
	case USE_GFTT: return GFTTDetector::create();
	case USE_BRISK: return BRISK::create();
	case USE_AKAZE: return AKAZE::create();
	case USE_KAZE: return KAZE::create();
	}
}

void trainModel()
{
	detector = getFeatureDetector();
	descriptor = getDescriptorExtractor();

	bool useForest;
	int choice = -1;
	while (choice < 0 || choice >= 2) {
		printf("Choose model:\n");
		printf("%4d - %s\n", 0, "Random forest");
		printf("%4d - %s\n", 1, "SVM");
		printf("Your choice: ");
		scanf("%d", &choice);
	}
	useForest = !choice;

	int vocSize = 25;

	Ptr<DescriptorMatcher> descriptorsMatcher =
		DescriptorMatcher::create("BruteForce");

	bowExtractor = new
		BOWImgDescriptorExtractor(
			descriptor,
			descriptorsMatcher);

	vector<string> allFiles, crocodile, folders;
	vector<int> responsesV;
	GetAllFolders("images", folders);

	for (string s : folders) {
		GetFilesInFolder(s, allFiles, "jpg");
		responsesV.push_back(allFiles.size());
	}
	/*
	GetFilesInFolder("bikes", allFiles, "jpg");
	int nLeopards = allFiles.size();
	GetFilesInFolder("bycicles", crocodile, "jpg");
	allFiles.insert(allFiles.end(), crocodile.begin(), crocodile.end());*/
	vector<bool> isTrain(allFiles.size());

	Mat category(allFiles.size(), 1, CV_32S);
	int cat = 0;
	for (int i = 0; i < allFiles.size(); i++) {
		if (i == responsesV[cat])
			cat++;
		category.at<int>(i) = cat;
	}
		//category.at<int>(i) = (i < nLeopards ? 1 : -1);

	InitRandomBoolVector(isTrain, 0.6);

	int nTrain = 0;
	for (int i = 0; i < isTrain.size(); i++)
		if (isTrain[i])
			nTrain++;

	//bowExtractor->setVocabulary(TrainVocabulary(allFiles, isTrain, detector, descriptor, vocSize));
	clock_t begin = clock();
	bowExtractor->setVocabulary(TrainVocabulary(allFiles, isTrain, detector, descriptor, nTrain));
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("Done! (took %.2fsecs)\n", elapsed_secs);

	ExtractTrainData(allFiles, isTrain, category, detector, bowExtractor, trainData, trainResp);

	printf("Training classifier...");
	if (useForest) {
		model = TrainClassifier(trainData, trainResp).dynamicCast<ml::StatModel>();
	}
	else {
		model = TrainSVM(trainData, trainResp).dynamicCast<ml::StatModel>();
	}
	 //= (useForest ? TrainClassifier(trainData, trainResp) :
		//TrainSVM(trainData, trainResp));
	//Ptr<ml::RTrees> model = TrainClassifier(trainData, trainResp);
	printf("Done!\n");
	//Ptr<ml::SVM> model = TrainSVM(trainData, trainResp);
	
	//Mat predictions = PredictOnTestData(allFiles, isTrain, detector, bowExtractor, rTree);
	vector<int> posMap;
	Mat predictions = PredictOnTestData(allFiles, isTrain, detector, bowExtractor, model, posMap);

	Mat testResponses = GetTestResponses(category, isTrain);

	float error = CalculateMisclassificationError(testResponses, predictions, allFiles, folders, posMap);

	printf("Total images: %zd\n Training set size: %d\n Classification error: %f\n",
		allFiles.size(), nTrain, error);
}

void testImages() {
	if (model.get() == NULL) {
		printf("Train the model first!\n");
		return;
	}
	vector<string> allFiles, folders;
	GetFilesInFolder("testImages", allFiles, "jpg");
	GetAllFolders("images", folders);
	for (string s : allFiles) {
		int p = Predict(detector, bowExtractor, model, s);
		printf("%s classified as %s\n", s.c_str(), folders[p].c_str());
	}
}

void showMainMenu() {
	int choice = -1;
	while (choice < 0 || choice >= nMenu) {
		printf("Choose action:\n");
		for (int i = 0; i < nMenu; i++) {
			printf("%4d - %s\n", i, menuOptions[i]);
		}
		printf("Your choice --> ");
		scanf("%d", &choice);
	}
	switch (choice) {
	case 0:
		trainModel();
		break;
	case 1:
		testImages();
		break;
	case 2:
		exit(0);
	}
}

int main(int argc, char* argv[])
{
	//trainModel();
	while (true)
		showMainMenu();
	return 0;
}