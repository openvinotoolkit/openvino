// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <vector>
#include <string>
#include <queue>

#include "transformations/init_node_info.hpp"
#include "low_precision_transformations/squeeze_transformation.hpp"
#include "ov_lpt_models/squeeze.hpp"

namespace LayerTestsDefinitions {

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}


std::string SqueezeTransformation::getTestCaseName(const testing::TestParamInfo<SqueezeTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::string targetDevice;
    SqueezeTransformationParam squeezeParam;
    std::tie(netPrecision, targetDevice, params, squeezeParam) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, squeezeParam.shape, targetDevice, params) << "_" <<
           squeezeParam.fakeQuantize << "_" <<
        squeezeParam.squeezeAxes << "_" <<
        params.updatePrecisions << "_" <<
        squeezeParam.shape;

    return result.str();
}
void SqueezeTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    SqueezeTransformationParam squeezeParam;

    std::tie(netPrecision, targetDevice, params, squeezeParam) = this->GetParam();

    init_input_shapes(squeezeParam.shape);

    function = ov::builder::subgraph::SqueezeFunction::getOriginal(
        netPrecision,
        squeezeParam.shape,
        squeezeParam.fakeQuantize,
        squeezeParam.squeezeAxes);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(SqueezeTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
