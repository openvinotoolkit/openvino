// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief Format reader abstract class implementation
 * \file format_reader.h
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include<iostream>

#ifdef _WIN32
    #define FORMAT_READER_API(TYPE) extern "C"   __declspec(dllexport)  TYPE __cdecl
#else  // Linux and Mac
    #define FORMAT_READER_API(TYPE) extern "C" TYPE
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
    /**
     * \brief Get width
     * @return width
     */
    size_t width() const { return _width; }

    /**
     * \brief Get height
     * @return height
     */
    size_t height() const { return _height; }

    /**
     * \brief Get input data ptr
     * @return shared pointer with input data
     * @In case of using OpenCV, parameters width and height will be used for image resizing
     */
    virtual std::shared_ptr<unsigned char> getData(int width = 0, int height = 0) = 0;

    /**
     * \brief Get size
     * @return size
     */
    virtual size_t size() const = 0;

    virtual void Release() noexcept = 0;
};
}  // namespace FormatReader

/**
 * \brief Function for create reader
 * @return FormatReader pointer
 */
FORMAT_READER_API(FormatReader::Reader*)CreateFormatReader(const char *filename);