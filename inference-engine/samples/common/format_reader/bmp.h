// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief BMP reader
 * \file bmp.h
 */
#pragma once

#include <memory>
#include <string>
#include <format_reader.h>

#include "register.h"

namespace FormatReader {
/**
 * \class BitMap
 * \brief Reader for bmp files
 */
class BitMap : public Reader {
private:
    static Register<BitMap> reg;

    typedef struct {
        unsigned short type   = 0u;              /* Magic identifier            */
        unsigned int size     = 0u;              /* File size in bytes          */
        unsigned int reserved = 0u;
        unsigned int offset   = 0u;              /* Offset to image data, bytes */
    } BmpHeader;

    typedef struct {
        unsigned int size = 0u;                  /* Header size in bytes      */
        int width = 0, height = 0;               /* Width and height of image */
        unsigned short planes = 0u;              /* Number of colour planes   */
        unsigned short bits = 0u;                /* Bits per pixel            */
        unsigned int compression = 0u;           /* Compression type          */
        unsigned int imagesize = 0u;             /* Image size in bytes       */
        int xresolution = 0, yresolution = 0;    /* Pixels per meter          */
        unsigned int ncolours = 0u;              /* Number of colours         */
        unsigned int importantcolours = 0u;      /* Important colours         */
    } BmpInfoHeader;

public:
    /**
     * \brief Constructor of BMP reader
     * @param filename - path to input data
     * @return BitMap reader object
     */
    explicit BitMap(const std::string &filename);
    virtual ~BitMap() {
    }

    /**
     * \brief Get size
     * @return size
     */
    size_t size() const override {
        return _width * _height * 3;
    }

    void Release() noexcept override {
        delete this;
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
