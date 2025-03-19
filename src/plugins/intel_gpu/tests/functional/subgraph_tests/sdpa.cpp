// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"

#include "openvino/opsets/opset13.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "openvino/pass/manager.hpp"

namespace {
// validate the batch axis padding for sdpa_micro kernel.
class SDPA : virtual public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        auto inType = ov::element::f16;
        ov::Shape inputShape{3, 4, 8, 16};
        auto constant1 = ov::op::v0::Constant::create(ov::element::i32, {4}, {1, 4, 8, 16});
        auto constant2 = ov::op::v0::Constant::create(ov::element::i32, {4}, {1, 4, 8, 16});
        auto constant3 = ov::op::v0::Constant::create(ov::element::i32, {4}, {1, 4, 8, 16});
        auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputShape);
        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i32, ov::Shape{}, std::vector<int64_t>{0});
        auto split = std::make_shared<ov::op::v1::Split>(input, split_axis_op, 3);

        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(split->output(0), constant1, false);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(split->output(1), constant2, false);
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(split->output(2), constant3, false);
        auto sdpa = std::make_shared<ov::opset13::ScaledDotProductAttention>(reshape1, reshape2, reshape3, false);
        sdpa->set_friendly_name("sdpa");

        auto output = std::make_shared<ov::op::v0::Result>(sdpa->output(0));
        function = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input}, "sdpa_model");

        functionRefs = function->clone();
        ov::pass::Manager manager;

        // Decompose ScaledDotProductAttention
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);

        bool has_long_seq = inputShape[2] >= 384 || inputShape[3] >= 128;
        if (inType == ov::element::f16) {
            if (has_long_seq) {
                abs_threshold = 0.025;
                rel_threshold = 0.025;
            } else {
                abs_threshold = 0.005;
                rel_threshold = 0.005;
            }
        }
    }
};

TEST_F(SDPA, Inference) {
    run();
}
}  // namespace
