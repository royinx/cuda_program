#include "argmax.h"
#include <opencv2/core/core.hpp>
#include <iostream>
using namespace std;


template<class ForwardIterator>
inline int argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

cv::Mat Matrix::channelArgMax(cv::Mat src) {
    cv::Mat out = cv::Mat::zeros(Matrix::INPUT_H, Matrix::INPUT_W, CV_8U);
    for (int h = 0; h < INPUT_H; ++h) {
        for (int w = 0; w < INPUT_W; ++w) {
            uchar *p = src.ptr(h, w); // prob of a point
            out.at<uchar>(h, w) = (uchar) argmax(p, p + 3);
        }
    }
    return out;
}

// int slow_calc(int x, int a, int b) {
//     return a * x + b;
// }

int main(){
    return 0;
}

// clear && gcc -shared -fPIC argmax.cpp -L/usr/local/lib/ `pkg-config --cflags --libs opencv`  -o argmax.so -lstdc++ 


// clear && g++  -Wall argmax.cpp -L/usr/local/lib/ `pkg-config --cflags --libs opencv` -o argmax && time ./argmax
// clear && g++ -shared -fPIC argmax.cpp -L/usr/local/lib/ `pkg-config --cflags --libs opencv` -o argmax.so

// clear && gcc -c argmax.cpp -L/usr/local/lib/ `pkg-config --cflags --libs opencv`  -o argmax.o -lstdc++ && time gcc -shared -o argmax.so argmax.o

// clear && gcc -shared -fPIC -c argmax.cpp -L/usr/local/lib/ `pkg-config --cflags --libs opencv`  -o argmax.so -lstdc++ 

// clear && 
// gcc -g -fPIC -c argmax.cpp -L/usr/local/lib/ -o argmax.o -lstdc++ `pkg-config --cflags --libs opencv` 
// gcc -shared -o argmax.so argmax.o


// g++ -c ../argmax.cpp -L/usr/local/lib/ `pkg-config --cflags --libs opencv` -o ../argmax.o

// g++ -fPIC -shared -o ../argmax.so ../argmax.o