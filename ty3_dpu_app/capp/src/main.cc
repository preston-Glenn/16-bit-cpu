#include "yolo_cpp.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <dnndk/dnndk.h>


void letterboxImage(cv::Mat& image, const std::pair<int, int>& size) {
    int ih = image.rows;
    int iw = image.cols;
    int w = size.first;
    int h = size.second;

    float scale = std::min(static_cast<float>(w) / iw, static_cast<float>(h) / ih);
    int nw = static_cast<int>(iw * scale);
    int nh = static_cast<int>(ih * scale);

    cv::resize(image, image, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    cv::Mat newImage = cv::Mat::ones(h, w, image.type()) * 128;
    
    int hStart = (h - nh) / 2;
    int wStart = (w - nw) / 2;
    
    cv::Rect roi(wStart, hStart, nw, nh);
    image.copyTo(newImage(roi));
    image = newImage;
}

cv::Mat preProcess(const cv::Mat& image, const std::pair<int, int>& modelImageSize) {
    cv::Mat imageCopy = image.clone();
    cv::cvtColor(imageCopy, imageCopy, cv::COLOR_BGR2RGB);
    int imageH = imageCopy.rows;
    int imageW = imageCopy.cols;

    int modelH = modelImageSize.first;
    int modelW = modelImageSize.second;

    if (modelImageSize != std::make_pair(-1, -1)) {
        assert(modelH % 32 == 0 && modelW % 32 == 0 && "Multiples of 32 required");
        letterboxImage(imageCopy, std::make_pair(modelW, modelH));
    } else {
        int newImageW = imageW - (imageW % 32);
        int newImageH = imageH - (imageH % 32);
        letterboxImage(imageCopy, std::make_pair(newImageW, newImageH));
    }

    cv::Mat imageData;
    imageCopy.convertTo(imageData, CV_32F, 1.0 / 255.0);
    cv::cvtColor(imageData, imageData, CV_BGR2RGB);
    cv::transpose(imageData, imageData);
    return imageData;
}

std::vector<std::string> getClass(const std::string& classesPath) {
    std::ifstream file(classesPath);
    std::vector<std::string> classNames;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            classNames.push_back(line);
        }
        file.close();
    }
    return classNames;
}

std::vector<float> getAnchors(const std::string& anchorsPath) {
    std::ifstream file(anchorsPath);
    std::vector<float> anchors;
    if (file.is_open()) {
        std::string line;
        std::getline(file, line);
        std::istringstream iss(line);
        float val;
        while (iss >> val) {
            anchors.push_back(val);
        }
        file.close();
    }
    return anchors;
}

std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> getFeats(const cv::Mat& feats, const std::vector<float>& anchors, int numClasses, const std::pair<int, int>& inputShape) {
    int numAnchors = anchors.size() / 2;
    cv::Mat anchorsTensor = cv::Mat(1, 1, CV_32FC2, anchors.data()).reshape(1, 1, 1, numAnchors, 2);
    cv::Size gridSize = feats.size[1];
    int nu = numClasses + 5;

    cv::Mat predictions = feats.reshape(1, gridSize.height, gridSize.width, numAnchors * nu);

    cv::Mat gridX = cv::Mat(gridSize.height, gridSize.width, CV_32F);
    cv::Mat gridY = cv::Mat(gridSize.height, gridSize.width, CV_32F);

    for (int i = 0; i < gridSize.height; ++i) {
        gridX.row(i).setTo(i);
    }

    for (int j = 0; j < gridSize.width; ++j) {
        gridY.col(j).setTo(j);
    }

    cv::Mat grid;
    cv::merge(std::vector<cv::Mat>{gridX, gridY}, grid);

    grid.convertTo(grid, CV_32F);

    cv::Mat boxXY = (1 / (1 + cv::exp(-predictions.colRange(0, 2)))) + grid;
    cv::Mat boxWH = cv::exp(predictions.colRange(2, 4)) * anchorsTensor / cv::Size2f(inputShape.second, inputShape.first);
    cv::Mat boxConfidence = 1 / (1 + cv::exp(-predictions.colRange(4, 5)));
    cv::Mat boxClassProbs = 1 / (1 + cv::exp(-predictions.colRange(5, nu)));

    return std::make_tuple(boxXY, boxWH, boxConfidence, boxClassProbs);
}

cv::Mat correctBoxes(const cv::Mat& boxXY, const cv::Mat& boxWH, const std::pair<int, int>& inputShape, const std::pair<int, int>& imageShape) {
    cv::Mat boxYX;
    cv::Mat boxHW;
    cv::flip(boxXY, boxYX, 2);
    cv::flip(boxWH, boxHW, 2);

    cv::Mat inputShapeMat = (cv::Mat_<float>(1, 2) << inputShape.second, inputShape.first);
    cv::Mat imageShapeMat = (cv::Mat_<float>(1, 2) << imageShape.second, imageShape.first);

    cv::Mat newShapeMat = cv::round(imageShapeMat * std::min(inputShapeMat / imageShapeMat));
    cv::Mat offset = (inputShapeMat - newShapeMat) / 2.0 / inputShapeMat;
    cv::Mat scale = inputShapeMat / newShapeMat;

    boxYX = (boxYX - offset) * scale;
    boxHW = boxHW * scale;

    cv::Mat boxMins = boxYX - (boxHW / 2.0);
    cv::Mat boxMaxes = boxYX + (boxHW / 2.0);

    std::vector<cv::Mat> boxMinsMaxes{boxMins.col(0), boxMins.col(1), boxMaxes.col(0), boxMaxes.col(1)};
    cv::Mat boxes;
    cv::hconcat(boxMinsMaxes, boxes);

    boxes = boxes.mul(cv::Mat(cv::Size(1, 4), CV_32FC1, cv::Scalar(imageShape.second, imageShape.first, imageShape.second, imageShape.first)));
    return boxes;
}

std::tuple<cv::Mat, cv::Mat> boxesAndScores(const cv::Mat& feats, const std::vector<float>& anchors, int classesNum, const std::pair<int, int>& inputShape, const std::pair<int, int>& imageShape) {
    auto [boxXY, boxWH, boxConfidence, boxClassProbs] = getFeats(feats, anchors, classesNum, inputShape);
    cv::Mat boxes = correctBoxes(boxXY, boxWH, inputShape, imageShape);
    boxes = boxes.reshape(1, 1);
    cv::Mat boxScores = boxConfidence.mul(boxClassProbs);
    boxScores = boxScores.reshape(1, 1);
    return std::make_tuple(boxes, boxScores);
}

cv::Mat drawBbox(const cv::Mat& image, const cv::Mat& bboxes, const std::vector<std::string>& classes) {
    int numClasses = classes.size();
    cv::Mat resultImage = image.clone();
    int imageH = image.rows;
    int imageW = image.cols;

    std::vector<cv::Scalar> colors;
    for (int i = 0; i < numClasses; ++i) {
        cv::Scalar color = cv::Scalar(std::rand() % 256, std::rand() % 256, std::rand() % 256);
        colors.push_back(color);
    }

    for (int i = 0; i < bboxes.rows; ++i) {
        auto coor = bboxes.row(i).colRange(0, 4);
        int classInd = static_cast<int>(bboxes.at<float>(i, 5));
        auto bboxColor = colors[classInd];

        cv::Rect bboxRect(coor.at<float>(0), coor.at<float>(1), coor.at<float>(2) - coor.at<float>(0), coor.at<float>(3) - coor.at<float>(1));
        cv::rectangle(resultImage, bboxRect, bboxColor, 2);

        // Add text with predicted class
        std::ostringstream labelStream;
        labelStream << classes[classInd] << " " << std::fixed << std::setprecision(2) << bboxes.at<float>(i, 4);
        std::string label = labelStream.str();

        int fontScale = 0.5;
        int bboxThick = static_cast<int>(0.6 * (imageH + imageW) / 600);

        cv::Point c1(coor.at<float>(0), coor.at<float>(1));
        cv::Point c2(coor.at<float>(2), coor.at<float>(3));

        cv::rectangle(resultImage, c1, c2, bboxColor, bboxThick);

        // Add text with predicted class
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, 2, nullptr);
        c2.x = c1.x + textSize.width;
        c2.y = c1.y - textSize.height - 5;

        cv::rectangle(resultImage, c1, c2, bboxColor, -1); // filled rectangle for text background
        cv::putText(resultImage, label, c1, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    }

    return resultImage;
}

std::vector<int> nmsBoxes(const cv::Mat& boxes, const cv::Mat& scores) {
    int numBoxes = boxes.rows;

    std::vector<float> x1, y1, x2, y2, areas, order;
    x1.reserve(numBoxes);
    y1.reserve(numBoxes);
    x2.reserve(numBoxes);
    y2.reserve(numBoxes);
    areas.reserve(numBoxes);
    order.reserve(numBoxes);

    for (int i = 0; i < numBoxes; ++i) {
        x1.push_back(boxes.at<float>(i, 0));
        y1.push_back(boxes.at<float>(i, 1));
        x2.push_back(boxes.at<float>(i, 2));
        y2.push_back(boxes.at<float>(i, 3));
        areas.push_back((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1));
        order.push_back(i);
    }

    std::sort(order.begin(), order.end(), [&](int i, int j) { return scores.at<float>(i) > scores.at<float>(j); });

    std::vector<int> keep;
    while (!order.empty()) {
        int i = order[0];
        keep.push_back(i);

        std::vector<int> inds;
        for (size_t j = 1; j < order.size(); ++j) {
            int k = order[j];
            float xx1 = std::max(x1[i], x1[k]);
            float yy1 = std::max(y1[i], y1[k]);
            float xx2 = std::min(x2[i], x2[k]);
            float yy2 = std::min(y2[i], y2[k]);

            float w = std::max(0.0f, xx2 - xx1 + 1);
            float h = std::max(0.0f, yy2 - yy1 + 1);

            float inter = w * h;
            float ovr = inter / (areas[i] + areas[k] - inter);

            if (ovr <= 0.55f) {
                inds.push_back(j);
            }
        }

        std::vector<int> newOrder;
        for (auto ind : inds) {
            newOrder.push_back(order[ind]);
        }

        order = newOrder;
    }

    return keep;
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> eval(const std::vector<cv::Mat>& yoloOutputs, const std::pair<int, int>& imageShape, int maxBoxes) {
    const float scoreThreshold = 0.2f;
    const float nmsThreshold = 0.45f;

    std::vector<std::string> classNames = getClass("./image/coco_classes.txt");
    std::vector<float> anchors = getAnchors("./model_data/yolo_anchors.txt");

    std::vector<cv::Mat> boxesVec, scoresVec, classesVec;

    for (size_t i = 0; i < yoloOutputs.size(); ++i) {
        auto [boxXY, boxWH, boxConfidence, boxClassProbs] = getFeats(yoloOutputs[i], anchors, classNames.size(), imageShape);
        auto [boxes, boxScores] = boxesAndScores(yoloOutputs[i], anchors, classNames.size(), imageShape, imageShape);

        boxesVec.push_back(boxes);
        scoresVec.push_back(boxScores);
        classesVec.push_back(boxClassProbs);
    }

    cv::Mat allBoxes = cv::vconcat(boxesVec);
    cv::Mat allScores = cv::vconcat(scoresVec);
    cv::Mat allClasses = cv::vconcat(classesVec);

    std::vector<int> indices = nmsBoxes(allBoxes, allScores, nmsThreshold);
    int numBoxes = std::min(static_cast<int>(indices.size()), maxBoxes);

    cv::Mat selectedBoxes(numBoxes, 4, CV_32FC1);
    cv::Mat selectedScores(numBoxes, classNames.size(), CV_32FC1);
    cv::Mat selectedClasses(numBoxes, classNames.size(), CV_32FC1);

    for (int i = 0; i < numBoxes; ++i) {
        int index = indices[i];

        cv::Mat box = allBoxes.row(index);
        cv::Mat score = allScores.row(index);
        cv::Mat classProb = allClasses.row(index);

        box.copyTo(selectedBoxes.row(i));
        score.copyTo(selectedScores.row(i));
        classProb.copyTo(selectedClasses.row(i));
    }

    return std::make_tuple(selectedBoxes, selectedScores, selectedClasses);
}
std::tuple<std::vector<std::vector<float>>, std::tuple<float, float, float, float>> inferImage(const std::string& imagePath, void* task, const std::vector<std::string>& classNames, const std::vector<float>& anchors, const std::pair<int, int>& inputShape) {
    cv::Mat image = cv::imread(imagePath);
    int imageHeight = image.rows;
    int imageWidth = image.cols;

    cv::Mat processedImage = preProcess(image, inputShape);

    // Set input tensor
    size_t inputLen = n2cube::dpuGetInputTensorSize(task, CONV_INPUT_NODE);
    n2cube::dpuSetInputTensorInHWCFP32(task, CONV_INPUT_NODE, processedImage.data, inputLen);

    // Run the DPU task
    n2cube::dpuRunTask(task);

    // Get the output tensors
    cv::Mat convOut1 = n2cube::dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE1, n2cube::dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE1));
    cv::Mat convOut2 = n2cube::dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE2, n2cube::dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE2));

    std::vector<cv::Mat> yoloOutputs = {convOut1, convOut2};

    auto [selectedBoxes, selectedScores, selectedClasses] = eval(yoloOutputs, {imageHeight, imageWidth}, 80);

    std::vector<std::vector<float>> resultItems;
    for (int i = 0; i < selectedBoxes.rows; ++i) {
        std::vector<float> item = {
            selectedClasses.at<float>(i),
            selectedScores.at<float>(i),
            selectedBoxes.at<float>(i, 0),
            selectedBoxes.at<float>(i, 1),
            selectedBoxes.at<float>(i, 2),
            selectedBoxes.at<float>(i, 3)
        };
        resultItems.push_back(item);
    }

    return std::make_tuple(resultItems, std::make_tuple(0.0, 0.0, 0.0, 0.0)); // Replace with actual time information
}
