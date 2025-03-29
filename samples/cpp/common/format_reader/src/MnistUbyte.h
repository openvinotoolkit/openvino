// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief Mnist reader
 * \file MnistUbyte.h
 */
#pragma once

#include <memory>
#include <string>

// clang-format off
#include "format_reader.h"
#include "register.h"
// clang-format on

namespace FormatReader {
/**
 * \class MnistUbyte
 * \brief Reader for mnist db files
 */
class MnistUbyte : public Reader {
private:
    int reverseInt(int i);

    static Register<MnistUbyte> reg;

public:
    /**
     * \brief Constructor of Mnist reader
     * @param filename - path to input data
     * @return MnistUbyte reader object
     */
    explicit MnistUbyte(const std::string& filename);
    virtual ~MnistUbyte() {}

    /**
     * \brief Get size
     * @return size
     */
    size_t size() const override {
        return _width * _height * 1;
    }

    std::shared_ptr<unsigned char> getData(size_t width, size_t height) override {
        if ((width * height != 0) && (_width * _height != width * height)) {
            std::cout << "[ WARNING ] Image won't be resized! Please use OpenCV.\n";
            return nullptr;
        }
        return _data;
    }
};
}  // namespace FormatReader
