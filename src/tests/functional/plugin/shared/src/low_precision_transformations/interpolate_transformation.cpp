// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/interpolate_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/interpolate.hpp"

namespace LayerTestsDefinitions {

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& values) {
    os << "{";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ",";
        }
    }
    os << "}";
    return os;
}

std::string InterpolateTransformation::getTestCaseName(const testing::TestParamInfo<InterpolateTransformationParams>& obj) {
    ov::element::Type precision;
    std::pair<ov::PartialShape, ov::Shape> shapes;
    std::string targetDevice;
    interpAttributes attributes;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    std::tie(precision, shapes, targetDevice, attributes) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(precision, shapes.first, targetDevice, params) << "_" <<
           shapes.second << "_" <<
        attributes.align_corners << "_" <<
        attributes.antialias << "_" <<
        attributes.axes << "_" <<
        attributes.mode << "_" <<
        attributes.pads_begin << "_" <<
        attributes.pads_end;
    return result.str();
}

void InterpolateTransformation::SetUp() {
    ov::element::Type precision;
    std::pair<ov::PartialShape, ov::Shape> shapes;
    interpAttributes attributes;
    std::tie(precision, shapes, targetDevice, attributes) = this->GetParam();

    init_input_shapes(shapes.first);

    ov::op::v0::Interpolate::Attributes interpAttrs;
    interpAttrs.axes = attributes.axes;
    interpAttrs.mode = attributes.mode;
    interpAttrs.align_corners = attributes.align_corners;
    interpAttrs.antialias = attributes.antialias;
    interpAttrs.pads_begin = attributes.pads_begin;
    interpAttrs.pads_end = attributes.pads_end;

    function = ov::builder::subgraph::InterpolateFunction::getOriginal(precision, shapes.first, shapes.second, interpAttrs);
}

TEST_P(InterpolateTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
