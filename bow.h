#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include <string>
#include <ctime>
#include <omp.h>

cv::Mat TrainVocabulary(const std::vector<std::string>& filesList, const std::vector<bool>& is_voc, 
	const cv::Ptr<cv::FeatureDetector>& keypointsDetector, const cv::Ptr<cv::DescriptorExtractor>& descriptorsExtractor, int vocSize);
cv::Mat ExtractFeaturesFromImage(cv::Ptr<cv::FeatureDetector> keypointsDetector, cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor, const std::string& fileName);
void ExtractTrainData(const std::vector<std::string>& filesList, const std::vector<bool>& isTrain, const cv::Mat& responses, 
	const cv::Ptr<cv::FeatureDetector>& keypointsDetector, const cv::Ptr<cv::BOWImgDescriptorExtractor>& bowExtractor, cv::Mat& trainData, cv::Mat& trainResponses);
cv::Ptr<cv::ml::RTrees> TrainClassifier(const cv::Mat& trainData, const cv::Mat& trainResponses);
cv::Ptr<cv::ml::SVM> TrainSVM(const cv::Mat& trainData, const cv::Mat& trainResponses);
int Predict(const cv::Ptr<cv::FeatureDetector> keypointsDetector, const cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor, 
	const cv::Ptr<cv::ml::StatModel> classifier, const std::string& fileName);
cv::Mat PredictOnTestData(const std::vector<std::string>& filesList, const std::vector<bool>& isTrain, 
	const cv::Ptr<cv::FeatureDetector> keypointsDetector, const cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor, const cv::Ptr<cv::ml::StatModel> classifier, std::vector<int>& posMap);
cv::Mat GetTestResponses(const cv::Mat& responses, const std::vector<bool>& isTrain);
float CalculateMisclassificationError(cv::Mat& responses, cv::Mat& predictions, const std::vector<std::string>& files, const std::vector<std::string>& classes, const std::vector<int>& posMap);