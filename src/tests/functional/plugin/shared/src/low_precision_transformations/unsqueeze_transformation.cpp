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
    auto [netPrecision, device, unsqueezeParam] = obj.param;
    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, unsqueezeParam.shape, device) << "_" <<
           unsqueezeParam.fakeQuantize << "_" <<
           unsqueezeParam.unsqueezeAxes << "_" <<
           unsqueezeParam.shape;

    return result.str();
}
void UnsqueezeTransformation::SetUp() {
    auto [netPrecision, device, unsqueezeParam] = this->GetParam();
    targetDevice = device;

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
