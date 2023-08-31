// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_intermediate_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace LayerTestsDefinitions {

std::string ConcatWithIntermediateTransformation::getTestCaseName(const testing::TestParamInfo<ConcatWithIntermediateTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(netPrecision, inputShapes, targetDevice, params, transparentIntermediate, multichannel) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) <<
        (transparentIntermediate ? "" : "_notTransparentIntermediate") <<
        (multichannel ? "_multichannel" : "");

    return result.str();
}

ov::test::utils::InputsMap ConcatWithIntermediateTransformation::get_input_map() {
    auto generate_default = [](const std::shared_ptr<ngraph::Node>& node,
                               size_t port,
                               const ov::element::Type& elemType,
                               const ov::Shape& targetShape) -> ov::runtime::Tensor {
        const auto name = node->get_friendly_name();
        if ((name != "fakeQuantize1") && (name != "fakeQuantize2")) {
            OPENVINO_THROW("unknown name: " + name);
        }
        const double k = (name == "fakeQuantize1") ? 1.0 : 2.0;
        const auto interval = LayerTestsUtils::LayerTransformation::getQuantizationInterval(ngraph::element::u8);
        const double low = interval.first / k;
        const double high = interval.second / k;

        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, static_cast<uint32_t>(high - low), low);
    };

    static ov::test::utils::InputsMap inputs_map{{ov::op::Op::get_type_info_static(), generate_default}};
    return inputs_map;
}

/*
* FQ       FQ
*  \       /
*   \  Intermediate (MaxPooling or Convolution)
*    \  /    \
*   Concat   Convolution
*/

void ConcatWithIntermediateTransformation::SetUp() {
    ngraph::element::Type ngPrecision;
    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params trasformationParams;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(ngPrecision, inputShape, targetDevice, trasformationParams, transparentIntermediate, multichannel) = this->GetParam();

    ngraph::PartialShape inputShape1 = inputShape;

    if (inputShape1[2].is_static() && transparentIntermediate) {
        inputShape1[2] = inputShape1[2].get_length() - 2;
    }

    if (inputShape1[3].is_static() && transparentIntermediate) {
        inputShape1[3] = inputShape1[3].get_length() - 2;
    }

    init_input_shapes({ inputShape1, inputShape });

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithIntermediate(
        ngPrecision,
        inputShape,
        transparentIntermediate,
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} });
}

TEST_P(ConcatWithIntermediateTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
