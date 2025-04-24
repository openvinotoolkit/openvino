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
#include "low_precision_transformations/unsqueeze_transformation.hpp"
#include "ov_lpt_models/unsqueeze.hpp"

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


std::string UnsqueezeTransformation::getTestCaseName(const testing::TestParamInfo<UnsqueezeTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::string targetDevice;
    UnsqueezeTransformationParam unsqueezeParam;
    std::tie(netPrecision, targetDevice, params, unsqueezeParam) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, unsqueezeParam.shape, targetDevice, params) << "_" <<
           unsqueezeParam.fakeQuantize << "_" <<
        unsqueezeParam.unsqueezeAxes << "_" <<
        params.updatePrecisions << "_" <<
        unsqueezeParam.shape;

    return result.str();
}
void UnsqueezeTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    UnsqueezeTransformationParam unsqueezeParam;

    std::tie(netPrecision, targetDevice, params, unsqueezeParam) = this->GetParam();

    init_input_shapes(unsqueezeParam.shape);

    function = ov::builder::subgraph::UnsqueezeFunction::getOriginal(
        netPrecision,
        unsqueezeParam.shape,
        unsqueezeParam.fakeQuantize,
        unsqueezeParam.unsqueezeAxes);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(UnsqueezeTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
