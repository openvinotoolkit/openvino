// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fold_rms_transposes.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"

namespace ov::test::intel_gpu {

TEST_F(TransformationTestsF, FoldRMSTransposesFeatureAxis) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 288, 1, -1, -1});
    auto input_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 2, 3, 4, 1});
    auto input_transpose = std::make_shared<ov::op::v1::Transpose>(input, input_order);
    auto gamma = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 1, 1, 288}, {1.0f});
    auto rms = std::make_shared<ov::op::internal::RMS>(input_transpose, gamma, 1e-6);
    auto output_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 4, 1, 2, 3});
    auto output_transpose = std::make_shared<ov::op::v1::Transpose>(rms, output_order);
    output_transpose->set_friendly_name("channel_rms");
    model = std::make_shared<ov::Model>(ov::OutputVector{output_transpose}, ov::ParameterVector{input});
    manager.register_pass<ov::intel_gpu::FoldRMSTransposes>();

    auto ref_rms = std::make_shared<ov::op::internal::RMS>(input, gamma, 1e-6, ov::element::f16, 1);
    ref_rms->set_friendly_name("channel_rms");
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{ref_rms}, ov::ParameterVector{input});
}

TEST_F(TransformationTestsF, FoldRMSTransposesRejectsNonInverseOrder) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 288, 1, -1, -1});
    auto input_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 2, 3, 4, 1});
    auto input_transpose = std::make_shared<ov::op::v1::Transpose>(input, input_order);
    auto gamma = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 1, 1, 288}, {1.0f});
    auto rms = std::make_shared<ov::op::internal::RMS>(input_transpose, gamma, 1e-6);
    auto output_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 4, 1, 3, 2});
    auto output_transpose = std::make_shared<ov::op::v1::Transpose>(rms, output_order);
    model = std::make_shared<ov::Model>(ov::OutputVector{output_transpose}, ov::ParameterVector{input});
    model_ref = model->clone();
    manager.register_pass<ov::intel_gpu::FoldRMSTransposes>();
}

}  // namespace ov::test::intel_gpu
