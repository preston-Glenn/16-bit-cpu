#ifndef YOLO_CPP_H
#define YOLO_CPP_H

#include <opencv2/opencv.hpp>

void letterboxImage(cv::Mat& image, const std::pair<int, int>& size);
cv::Mat preProcess(const cv::Mat& image, const std::pair<int, int>& modelImageSize);
std::vector<std::string> getClass(const std::string& classesPath);
std::vector<float> getAnchors(const std::string& anchorsPath);
std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> getFeats(const cv::Mat& feats, const std::vector<float>& anchors, int numClasses, const std::pair<int, int>& inputShape);
cv::Mat correctBoxes(const cv::Mat& boxXY, const cv::Mat& boxWH, const std::pair<int, int>& inputShape, const std::pair<int, int>& imageShape);
std::tuple<cv::Mat, cv::Mat> boxesAndScores(const cv::Mat& feats, const std::vector<float>& anchors, int classesNum, const std::pair<int, int>& inputShape, const std::pair<int, int>& imageShape);
cv::Mat drawBbox(const cv::Mat& image, const cv::Mat& bboxes, const std::vector<std::string>& classes);
std::vector<int> nmsBoxes(const cv::Mat& boxes, const cv::Mat& scores);
std::tuple<cv::Mat, cv::Mat, cv::Mat> eval(const std::vector<cv::Mat>& yoloOutputs, const std::pair<int, int>& imageShape, int maxBoxes = 80);
std::tuple<std::vector<std::vector<float>>, std::tuple<float, float, float, float>> inferImage(const std::string& imagePath, void* task, const std::vector<std::string>& classNames, const std::vector<float>& anchors, const std::pair<int, int>& inputShape);

#endif // YOLO_CPP_H