// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/single_layer/eye.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/pass/constant_folding.hpp>

#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

std::string EyeLayerTest::getTestCaseName(testing::TestParamInfo<EyeLayerTestParams> obj) {
    EyeLayerTestParams params = obj.param;

    std::string td;
    std::vector<ov::Shape> input_shapes;
    ElementType net_precision;
    std::vector<LocalElementType> out_batch_shape;
    std::vector<int> eye_par;
    std::tie(input_shapes, out_batch_shape, eye_par, net_precision, td) = params;
    std::ostringstream result;
    result << "EyeTest_";
    result << "IS=(";
    for (const auto& shape : input_shapes) {
        result << ov::test::utils::partialShape2str({shape}) << "_";
    }
    result << ")";
    result << "rowNum=" << eye_par[0] << "_";
    result << "colNum=" << eye_par[1] << "_";
    result << "diagShift=" << eye_par[2] << "_";
    result << "batchShape=" << ov::test::utils::vec2str(out_batch_shape) << "_";
    result << net_precision << "_";
    result << std::to_string(obj.index);
    return result.str();
}

void EyeLayerTest::SetUp() {
    std::vector<ov::Shape> input_shapes;
    LocalElementType row_num, col_num;
    LocalElementType shift;
    std::vector<LocalElementType> out_batch_shape;
    ElementType net_precision;
    EyeLayerTestParams basicParamsSet = this->GetParam();

    std::vector<int> eye_par;
    std::tie(input_shapes, out_batch_shape, eye_par, net_precision, targetDevice) = basicParamsSet;
    row_num = eye_par[0];
    col_num = eye_par[1];
    shift = eye_par[2];

    std::shared_ptr<ngraph::op::v9::Eye> eye_operation;

    auto rows_const = std::make_shared<ov::op::v0::Constant>(ngraph::element::i32, input_shapes[0], &row_num);
    rows_const->set_friendly_name("rows");
    auto cols_const = std::make_shared<ov::op::v0::Constant>(ngraph::element::i32, input_shapes[1], &col_num);
    cols_const->set_friendly_name("cols");
    auto diag_const = std::make_shared<ov::op::v0::Constant>(ngraph::element::i32, input_shapes[2], &shift);
    diag_const->set_friendly_name("diagInd");

    if (!out_batch_shape.empty() && out_batch_shape[0] != 0) {
        auto batch_shape_par = std::make_shared<ov::op::v0::Constant>(ngraph::element::i32,
                                                                      ov::Shape{out_batch_shape.size()},
                                                                      out_batch_shape.data());
        batch_shape_par->set_friendly_name("batchShape");
        eye_operation =
            std::make_shared<ngraph::op::v9::Eye>(rows_const, cols_const, diag_const, batch_shape_par, net_precision);
    } else {
        eye_operation = std::make_shared<ngraph::op::v9::Eye>(rows_const, cols_const, diag_const, net_precision);
    }
    // Without this call the eye operation will be calculated by CPU and substituted by Constant operator
    ov::pass::disable_constant_folding(eye_operation);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eye_operation)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{}, "eye");
}
}  // namespace LayerTestsDefinitions
