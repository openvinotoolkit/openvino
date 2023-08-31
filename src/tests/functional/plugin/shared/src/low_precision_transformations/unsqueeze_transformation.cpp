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
#include "common_test_utils/ov_tensor_utils.hpp"
#include "low_precision_transformations/unsqueeze_transformation.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/unsqueeze_function.hpp"

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

ov::test::utils::InputsMap UnsqueezeTransformation::get_input_map() {
    auto generate_default = [this](const std::shared_ptr<ngraph::Node>&node,
                               size_t port,
                               const ov::element::Type & elemType,
                               const ov::Shape & targetShape) -> ov::runtime::Tensor {
        ngraph::element::Type netPrecision;
        ngraph::pass::low_precision::LayerTransformation::Params params;
        UnsqueezeTransformationParam squeezeParam;
        std::string targetDevice;

        std::tie(netPrecision, targetDevice, params, squeezeParam) = this->GetParam();

        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData = squeezeParam.fakeQuantize;

        const float range = static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]);
        const double start_from = fqOnData.empty() ? -12.5 : fqOnData.outputLowValues[0];
        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, range, start_from);
    };

    static ov::test::utils::InputsMap inputs_map{
        { ov::op::Op::get_type_info_static(), generate_default }
    };
    return inputs_map;
}

std::string UnsqueezeTransformation::getTestCaseName(const testing::TestParamInfo<UnsqueezeTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
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
    ngraph::pass::low_precision::LayerTransformation::Params params;
    UnsqueezeTransformationParam unsqueezeParam;

    std::tie(netPrecision, targetDevice, params, unsqueezeParam) = this->GetParam();
    init_input_shapes(unsqueezeParam.shape);

    function = ngraph::builder::subgraph::UnsqueezeFunction::getOriginal(
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
