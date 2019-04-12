// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef USE_OPENCV
#include "opencv_wraper.h"
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <samples/slog.hpp>

using namespace std;
using namespace FormatReader;

OCVReader::OCVReader(const string &filename) {
    img = cv::imread(filename);
    _size = 0;

    if (img.empty()) {
        return;
    }

    _size   = img.size().width * img.size().height * img.channels();
    _width  = img.size().width;
    _height = img.size().height;
}

std::shared_ptr<unsigned char> OCVReader::getData(size_t width = 0, size_t height = 0) {
    cv::Mat resized(img);
    if (width != 0 && height != 0) {
        size_t iw = img.size().width;
        size_t ih = img.size().height;
        if (width != iw || height != ih) {
            slog::warn << "Image is resized from (" << iw << ", " << ih << ") to (" << width << ", " << height << ")" << slog::endl;
        }
        cv::resize(img, resized, cv::Size(width, height));
    }

    size_t size = resized.size().width * resized.size().height * resized.channels();
    _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
    for (size_t id = 0; id < size; ++id) {
        _data.get()[id] = resized.data[id];
    }
    return _data;
}
#endif
