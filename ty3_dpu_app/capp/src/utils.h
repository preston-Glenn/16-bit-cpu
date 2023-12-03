#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

cv::Mat letterbox_image(const cv::Mat& image, const cv::Size& size);
cv::Mat pre_process(const cv::Mat& image, const cv::Size& model_image_size);
std::vector<std::string> get_class(const std::string& classes_path);
std::vector<float> get_anchors(const std::string& anchors_path);
std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> get_feats(const cv::Mat& feats, const std::vector<float>& anchors, int num_classes, const cv::Size& input_shape);
cv::Mat correct_boxes(const cv::Mat& box_xy, const cv::Mat& box_wh, const cv::Size& input_shape, const cv::Size& image_shape);
std::tuple<cv::Mat, cv::Mat> boxes_and_scores(const cv::Mat& feats, const std::vector<float>& anchors, int classes_num, const cv::Size& input_shape, const cv::Size& image_shape);
cv::Mat draw_bbox(const cv::Mat& image, const cv::Mat& bboxes, const std::vector<std::string>& classes);
std::vector<int> nms_boxes(const cv::Mat& boxes, const cv::Mat& scores);

#endif // UTILS_H