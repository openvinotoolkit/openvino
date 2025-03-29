// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/transpose_after_mat_mul.hpp"
#include "low_precision/network_helper.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {
    std::shared_ptr<ov::Model> TransposeAfterMatMulFunction::getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape) {
        const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
        input1->set_friendly_name("input1");

        const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
        input2->set_friendly_name("input2");

        const float k = 50.f;
        const auto fakeQuantize1 = ov::test::utils::make_fake_quantize(input1, precision, 256ul, { 1ul }, { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
        input2->set_friendly_name("fakeQuantize1");
        const auto fakeQuantize2 = ov::test::utils::make_fake_quantize(input2, precision, 256ul, { 1ul }, { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
        input2->set_friendly_name("fakeQuantize2");
        const auto matMul = std::make_shared<ov::opset1::MatMul>(fakeQuantize1, fakeQuantize2, false, false);
        input2->set_friendly_name("matMul");
        const auto transpose = std::make_shared<ov::opset1::Transpose>(
            matMul,
            ov::opset1::Constant::create(ov::element::i64, ov::Shape{ 4ul }, { 0, 2, 1, 3 }));
        transpose->set_friendly_name("transpose");

        ov::ResultVector results{ std::make_shared<ov::opset1::Result>(transpose) };
        std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
            results,
            ov::ParameterVector{ input1, input2 },
            "TransposeAfterMatMulFunction");

        return function;
    }
}  // namespace subgraph
}  // namespace builder
}  // namespace ov
