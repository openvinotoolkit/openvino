// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief Format reader abstract class implementation
 * \file format_reader.h
 */
#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace reference {
namespace img {
/**
 * \class FormatReader
 * \brief This is an abstract class for reading input data
 */
class images {
protected:
    /// \brief height
    size_t _height = 0;
    /// \brief width
    size_t _width = 0;
    /// \brief data
    // std::shared_ptr<unsigned char> _data;
    /// \brief shape - data shape
    std::vector<size_t> _shape;

    const char* _data;
    size_t _length;
    size_t _offset;

public:

    images() {}

    virtual ~images() = default;

    virtual bool isSupported(const char* content, size_t img_length) = 0;

    virtual void closeFile() = 0;
    /**
     * \brief Get width
     * @return width
     */
    size_t width() const {
        return _width;
    }

    /**
     * \brief Get height
     * @return height
     */
    size_t height() const {
        return _height;
    }

    /**
     * \brief Get full shape vector
     * @return vector of size_t values determining data shape
     */
    std::vector<size_t> shape() const {
        return _shape;
    }

    /**
     * \brief Get input data ptr
     * @return shared pointer with input data
     * @In case of using OpenCV, parameters width and height will be used for image resizing
     */
    virtual int getData(Tensor& output) = 0;

    /**
     * \brief Get size
     * @return size
     */
    virtual size_t size() const = 0;
};

static std::vector<std::shared_ptr<images>> image_formats;

std::shared_ptr<images> ParserImages(const char* content, size_t img_length) ;
}  // namespace img
}  // namespace reference
}  // namespace ov
