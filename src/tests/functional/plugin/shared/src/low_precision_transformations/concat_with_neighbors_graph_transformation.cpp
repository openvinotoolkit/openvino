// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_neighbors_graph_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"

namespace LayerTestsDefinitions {

std::string ConcatWithNeighborsGraphTransformation::getTestCaseName(const testing::TestParamInfo<ConcatNeighboringGraphTransformationParams>& obj) {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(precision, inputShapes, targetDevice, params) = obj.param;

    return getTestCaseNameByParams(precision, inputShapes, targetDevice, params);
}

ov::test::utils::InputsMap ConcatWithNeighborsGraphTransformation::get_input_map() {
    auto generate_default = [](const std::shared_ptr<ngraph::Node>&node,
                               size_t port,
                               const ov::element::Type & elemType,
                               const ov::Shape & targetShape) -> ov::runtime::Tensor {
        const auto name = node->get_friendly_name();
        if ((name != "fakeQuantize1") && (name != "fakeQuantize2") && (name != "fakeQuantize3")) {
            OPENVINO_THROW("unknown name: " + name);
        }
        const double k = (name == "fakeQuantize1") ? 1.0 : (name == "fakeQuantize2" ? 2.0 : 3.0);
        const auto interval = LayerTestsUtils::LayerTransformation::getQuantizationInterval(ngraph::element::u8);
        const double low = interval.first / k;
        const double high = interval.second / k;

        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, static_cast<uint32_t>(high - low), low);
    };

    static ov::test::utils::InputsMap inputs_map{
        { ov::op::Op::get_type_info_static(), generate_default }
    };
    return inputs_map;
}

void ConcatWithNeighborsGraphTransformation::SetUp() {
    rel_threshold = 2.e-2;
    ngraph::element::Type ngPrecision;
    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(ngPrecision, inputShape, targetDevice, params) = this->GetParam();

    init_input_shapes({ inputShape, inputShape, inputShape });

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithNeighbors(
        ngPrecision,
        inputShape,
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 3.f} },
        "concat",
        "");
}

TEST_P(ConcatWithNeighborsGraphTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
