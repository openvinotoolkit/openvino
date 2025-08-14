// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/single_op/eye.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/eye.hpp"

namespace ov {
namespace test {
std::string EyeLayerTest::getTestCaseName(testing::TestParamInfo<EyeLayerTestParams> obj) {
    const auto& [input_shapes, out_batch_shape, eye_par, model_type, td] = obj.param;
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
    result << model_type << "_";
    result << std::to_string(obj.index);
    return result.str();
}

void EyeLayerTest::SetUp() {
    int row_num, col_num;
    int shift;

    const auto& [input_shapes, out_batch_shape, eye_par, model_type, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;
    row_num = eye_par[0];
    col_num = eye_par[1];
    shift = eye_par[2];

    std::shared_ptr<ov::op::v9::Eye> eye_operation;

    auto rows_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, input_shapes[0], &row_num);
    rows_const->set_friendly_name("rows");
    auto cols_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, input_shapes[1], &col_num);
    cols_const->set_friendly_name("cols");
    auto diag_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, input_shapes[2], &shift);
    diag_const->set_friendly_name("diagInd");

    if (!out_batch_shape.empty() && out_batch_shape[0] != 0) {
        auto batch_shape_par = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                      ov::Shape{out_batch_shape.size()},
                                                                      out_batch_shape.data());
        batch_shape_par->set_friendly_name("batchShape");
        eye_operation = std::make_shared<ov::op::v9::Eye>(rows_const, cols_const, diag_const, batch_shape_par, model_type);
    } else {
        eye_operation = std::make_shared<ov::op::v9::Eye>(rows_const, cols_const, diag_const, model_type);
    }

    // Without this call the eye operation will be calculated by CPU and substituted by Constant operator
    ov::pass::disable_constant_folding(eye_operation);
    auto result = std::make_shared<ov::op::v0::Result>(eye_operation);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector(), "eye");
}
}  // namespace test
}  // namespace ov
