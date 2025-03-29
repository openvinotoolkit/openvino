// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class interpAttributes {
public:
    ov::AxisSet axes;
    std::string mode;
    bool align_corners;
    bool antialias;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;

    bool shouldBeTransformed;

    interpAttributes() = default;

    interpAttributes(const ov::AxisSet& axes,
                     const std::string& mode,
                     const bool& align_corners,
                     const bool& antialias,
                     const std::vector<size_t>& pads_begin,
                     const std::vector<size_t>& pads_end,
                     const bool& shouldBeTransformed = true) :
            axes(axes), mode(mode), align_corners(align_corners),
            antialias(antialias), pads_begin(pads_begin), pads_end(pads_end) {}
};

typedef std::tuple<
    ov::element::Type,
    std::pair<ov::PartialShape, ov::Shape>,
    std::string,
    interpAttributes> InterpolateTransformationParams;

class InterpolateTransformation :
    public testing::WithParamInterface<InterpolateTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
