// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <string>

#include <ie_blob.h>
#include <inference_engine.hpp>
#include "base_matcher.hpp"

/**
 * @class Color
 * @brief A Color class stores channels of a given color
 */
class Color {
private:
    unsigned char _r;
    unsigned char _g;
    unsigned char _b;

public:
    /**
     * A default constructor.
     * @param r - value for red channel
     * @param g - value for green channel
     * @param b - value for blue channel
     */
    Color(unsigned char r,
          unsigned char g,
          unsigned char b) : _r(r), _g(g), _b(b) {}

    inline unsigned char red() {
        return _r;
    }

    inline unsigned char blue() {
        return _b;
    }

    inline unsigned char green() {
        return _g;
    }
};

namespace Regression { namespace Matchers {

class SegmentationMatcher : public BaseMatcher {
 private:
    InferenceEngine::TBlob<float>::Ptr output;
    std::vector<std::vector<size_t>> outArray;
    size_t C = -1;

 public:
    SegmentationMatcher (const RegressionConfig & config)
        : BaseMatcher(config) {
    }

    virtual void match();

    static float compareOutputBmp(std::vector<std::vector<size_t>> data, size_t classesNum, const std::string& inFileName);

    void checkResult(std::string imageFileName);

    SegmentationMatcher& to(std::string imageFileName) {
        match();
        checkResult(imageFileName);
        return *this;
    }
};

} }  //  namespace matchers
