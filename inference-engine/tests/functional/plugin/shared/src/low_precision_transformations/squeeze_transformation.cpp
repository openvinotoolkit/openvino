// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <queue>
#include <ie_core.hpp>

#include "ngraph/op/op.hpp"
#include <transformations/init_node_info.hpp>
#include "low_precision_transformations/squeeze_transformation.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/squeeze_function.hpp"

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

InferenceEngine::Blob::Ptr SqueezeTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    SqueezeTransformationParam squeezeParam;
    std::string targetDevice;

    std::tie(netPrecision, targetDevice, params, squeezeParam) = this->GetParam();

    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData = squeezeParam.fakeQuantize;

    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]),
        static_cast<int32_t>(fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0]),
        1ul);
}

std::string SqueezeTransformation::getTestCaseName(testing::TestParamInfo<SqueezeTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::string targetDevice;
    SqueezeTransformationParam squeezeParam;
    std::tie(netPrecision, targetDevice, params, squeezeParam) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, squeezeParam.shape, targetDevice, params) << "_" <<
        squeezeParam.fakeQuantize << "_" <<
        squeezeParam.squeezeAxes << "_" <<
        params.updatePrecisions << "_" <<
        squeezeParam.shape;

    return result.str();
}
void SqueezeTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    SqueezeTransformationParam squeezeParam;

    std::tie(netPrecision, targetDevice, params, squeezeParam) = this->GetParam();

    function = ngraph::builder::subgraph::SqueezeFunction::getOriginal(
        netPrecision,
        squeezeParam.shape,
        squeezeParam.fakeQuantize,
        squeezeParam.squeezeAxes);

    ngraph::pass::InitNodeInfo().run_on_function(function);
}

TEST_P(SqueezeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
