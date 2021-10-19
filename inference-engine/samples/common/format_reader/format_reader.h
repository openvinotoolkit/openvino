// Copyright (C) 2018-2021 Intel Corporation
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

#include "openvino/core/visibility.hpp"

#ifdef format_reader_EXPORTS
#    define FORMAT_READER_API(type) OPENVINO_CORE_EXPORTS type
#else
#    define FORMAT_READER_API(type) OPENVINO_CORE_IMPORTS type
#endif

namespace FormatReader {
/**
 * \class FormatReader
 * \brief This is an abstract class for reading input data
 */
class Reader {
protected:
    /// \brief height
    size_t _height = 0;
    /// \brief width
    size_t _width = 0;
    /// \brief data
    std::shared_ptr<unsigned char> _data;

public:
    virtual ~Reader() = default;

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
     * \brief Get input data ptr
     * @return shared pointer with input data
     * @In case of using OpenCV, parameters width and height will be used for image resizing
     */
    virtual std::shared_ptr<unsigned char> getData(size_t width = 0, size_t height = 0) = 0;

    /**
     * \brief Get size
     * @return size
     */
    virtual size_t size() const = 0;
};
}  // namespace FormatReader

/**
 * \brief Function for create reader
 * @return FormatReader pointer
 */
FORMAT_READER_API(FormatReader::Reader*) CreateFormatReader(const char* filename);