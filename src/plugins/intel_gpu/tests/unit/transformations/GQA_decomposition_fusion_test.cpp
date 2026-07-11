// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/variadic_split.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "plugin/transformations/GQA_decomposition_fusion.hpp"
#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, GQADecompositionFusion1) {
    std::vector<int64_t> in0_order = {0, 1, 2, 3};
    std::vector<int64_t> in1_order = {0, 1, 2, 3};
    std::vector<int64_t> in2_order = {0, 1, 2, 3};
    std::vector<int64_t> out_order = {0, 1, 2, 3};
    const bool is_causal = false;
    {
        auto input_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 8, 16});
        auto input_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 8, 16});
        auto input_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 8, 16});

        auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
        auto k_updates = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 8, 16}, {0.0f});
        auto v_updates = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 8, 16}, {0.0f});

        auto scatter_k = std::make_shared<ov::op::v3::ScatterUpdate>(input_k, indices, k_updates, axis);
        auto scatter_v = std::make_shared<ov::op::v3::ScatterUpdate>(input_v, indices, v_updates, axis);
        auto split_k = std::make_shared<ov::op::v1::VariadicSplit>(scatter_k, axis, split_lengths);
        auto split_v = std::make_shared<ov::op::v1::VariadicSplit>(scatter_v, axis, split_lengths);
        auto split_k_out = split_k->output(0);
        auto split_v_out = split_v->output(0);

        auto reshape_k_5d_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {1, 1, 1, 8, 16});
        auto reshape_v_5d_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {1, 1, 1, 8, 16});

        auto reshape_k_5d = std::make_shared<ov::op::v1::Reshape>(split_k_out, reshape_k_5d_pattern, false);
        auto reshape_v_5d = std::make_shared<ov::op::v1::Reshape>(split_v_out, reshape_v_5d_pattern, false);

        auto concat_k = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reshape_k_5d, reshape_k_5d, reshape_k_5d, reshape_k_5d},2);
        auto concat_v = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reshape_v_5d, reshape_v_5d, reshape_v_5d, reshape_v_5d},2);

        auto reshape_k_4d_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 4 }, { 1, 4, 8, 16 });
        auto reshape_v_4d_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 4 }, { 1, 4, 8, 16 });

        auto reshape_k_4d = std::make_shared<ov::op::v1::Reshape>(concat_k, reshape_k_4d_pattern, false);
        auto reshape_v_4d = std::make_shared<ov::op::v1::Reshape>(concat_v, reshape_v_4d_pattern, false);

        auto inputs = ov::OutputVector{input_q, reshape_k_4d, reshape_v_4d};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs, is_causal, in0_order, in1_order, in2_order, out_order);

        model = std::make_shared<ov::Model>(ov::OutputVector{ sdpa }, ov::ParameterVector{input_q, input_k, input_v});
        manager.register_pass<GQADecompositionfusion>();
    }
    {
        auto input_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 8, 16});
        auto input_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 8, 16});
        auto input_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 8, 16});

        auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
        auto k_updates = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 8, 16}, {0.0f});
        auto v_updates = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 8, 16}, {0.0f});

        auto scatter_k = std::make_shared<ov::op::v3::ScatterUpdate>(input_k, indices, k_updates, axis);
        auto scatter_v = std::make_shared<ov::op::v3::ScatterUpdate>(input_v, indices, v_updates, axis);
        auto split_k = std::make_shared<ov::op::v1::VariadicSplit>(scatter_k, axis, split_lengths);
        auto split_v = std::make_shared<ov::op::v1::VariadicSplit>(scatter_v, axis, split_lengths);

        auto inputs = ov::OutputVector{ input_q, split_k->output(1), split_v->output(1) };
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs, is_causal, in0_order, in1_order, in2_order, out_order);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{ sdpa }, ov::ParameterVector{ input_q, input_k, input_v });
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov