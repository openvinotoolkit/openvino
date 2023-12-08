// Copyright (C) 2018-2023 Intel Corporation
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
#include "low_precision_transformations/unsqueeze_transformation.hpp"
#include "ov_models/subgraph_builders.hpp"
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

InferenceEngine::Blob::Ptr UnsqueezeTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    UnsqueezeTransformationParam squeezeParam;
    std::string targetDevice;

    std::tie(netPrecision, targetDevice, params, squeezeParam) = this->GetParam();

    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData = squeezeParam.fakeQuantize;

    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]),
        static_cast<int32_t>(fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0]),
        1ul);
}

std::string UnsqueezeTransformation::getTestCaseName(const testing::TestParamInfo<UnsqueezeTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::string targetDevice;
    UnsqueezeTransformationParam unsqueezeParam;
    std::tie(netPrecision, targetDevice, params, unsqueezeParam) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, unsqueezeParam.shape, targetDevice, params) << "_" <<
        unsqueezeParam.fakeQuantize << "_" <<
        unsqueezeParam.unsqueezeAxes << "_" <<
        params.updatePrecisions << "_" <<
        unsqueezeParam.shape;

    return result.str();
}
void UnsqueezeTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    UnsqueezeTransformationParam unsqueezeParam;

    std::tie(netPrecision, targetDevice, params, unsqueezeParam) = this->GetParam();

    function = ngraph::builder::subgraph::UnsqueezeFunction::getOriginal(
        netPrecision,
        unsqueezeParam.shape,
        unsqueezeParam.fakeQuantize,
        unsqueezeParam.unsqueezeAxes);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(UnsqueezeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
