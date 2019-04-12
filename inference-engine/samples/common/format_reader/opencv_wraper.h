// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief Image reader
 * \file opencv_wraper.h
 */
#pragma once

#ifdef USE_OPENCV
#include <memory>
#include <string>
#include <format_reader.h>

#include <opencv2/opencv.hpp>

#include "register.h"

namespace FormatReader {
/**
 * \class OCVMAT
 * \brief OpenCV Wraper
 */
class OCVReader : public Reader {
private:
    cv::Mat img;
    size_t _size;
    static Register<OCVReader> reg;

public:
    /**
    * \brief Constructor of BMP reader
    * @param filename - path to input data
    * @return BitMap reader object
    */
    explicit OCVReader(const std::string &filename);
    virtual ~OCVReader() {
    }

    /**
    * \brief Get size
    * @return size
    */
    size_t size() const override {
        return _size;
    }

    void Release() noexcept override {
        delete this;
    }

    std::shared_ptr<unsigned char> getData(size_t width, size_t height) override;
};
}  // namespace FormatReader
#endif