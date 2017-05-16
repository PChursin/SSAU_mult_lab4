#include <string>
#include <vector>
#include <opencv2/xfeatures2d.hpp>

void GetFilesInFolder(const std::string& dirPath, std::vector<std::string> &filesList, const char* ext);
void InitRandomBoolVector(std::vector<bool>& mask, double prob);
