#ifndef ARGMAX_H
#define ARGMAX_H

#include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv/cv.hpp>
// #include <opencv2/imgcodecs/imgcodecs.hpp>
// using namespace cv;
// extern cv::Mat channelArgMax(cv::Mat src);

class Matrix {
    public:
        cv::Mat channelArgMax(cv::Mat src);
    private:
        int INPUT_H = 360;
        int INPUT_W = 640;

};

#endif