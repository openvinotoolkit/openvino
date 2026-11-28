// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <numeric>
#include <random>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

using ov::test::InputShape;

constexpr size_t hidden_size = 128;
constexpr size_t num_experts = 8;

using MoEScatterReduceParams = std::tuple<std::vector<InputShape>,  // hidden states, the router ids and weights are derived from its number of tokens
                                          size_t,                   // top_k
                                          size_t>;                  // output size of the experts

// A grouped experts forward of a MoE block as exported from transformers (`grouped_mm_experts_forward`):
// the (token, slot) rows are sorted by expert, GroupedMatMul computes the rows of each expert, the rows are gathered back
// with the argsort of the permutation, weighted by the router weights and summed over the top-k slots of each token.
class MoEScatterReduceTest : public testing::WithParamInterface<MoEScatterReduceParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoEScatterReduceParams>& info) {
        const auto& [input_shapes, top_k, out_size] = info.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({input_shapes[0].first}) << "_TS=";
        for (const auto& shape : input_shapes[0].second) {
            result << ov::test::utils::vec2str(shape) << ",";
        }
        result << "top_k=" << top_k << "_out=" << out_size;
        return result.str();
    }

protected:
    static ov::Output<ov::Node> gather0(const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& indices) {
        return std::make_shared<ov::op::v8::Gather>(data, indices, ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
    }

    static std::shared_ptr<ov::Node> argsort(const ov::Output<ov::Node>& data) {
        auto len = std::make_shared<ov::op::v0::Squeeze>(std::make_shared<ov::op::v3::ShapeOf>(data));
        return std::make_shared<ov::op::v11::TopK>(data,
                                                   len,
                                                   -1,
                                                   ov::op::v11::TopK::Mode::MIN,
                                                   ov::op::v11::TopK::SortType::SORT_VALUES,
                                                   ov::element::i64);
    }

    std::shared_ptr<ov::Model> build_model(const ov::PartialShape& hidden_shape, int64_t top_k, size_t out_size) {
        auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, hidden_shape);
        auto topk_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{hidden_shape[0], top_k});
        auto topk_weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{hidden_shape[0], top_k});

        auto flat_ids = std::make_shared<ov::op::v1::Reshape>(topk_ids, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
        auto sort = argsort(flat_ids);
        auto perm = sort->output(1);

        // token of each sorted row
        auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(topk_ids);
        auto num_tokens = gather0(ids_shape, ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        auto range = std::make_shared<ov::op::v4::Range>(ov::op::v0::Constant::create(ov::element::i64, {}, {0}),
                                                         num_tokens,
                                                         ov::op::v0::Constant::create(ov::element::i64, {}, {1}),
                                                         ov::element::i64);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(range, ov::op::v0::Constant::create(ov::element::i64, {}, {1}));
        auto expand = std::make_shared<ov::op::v3::Broadcast>(unsqueeze, ids_shape);
        auto token_src = std::make_shared<ov::op::v1::Reshape>(expand, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
        auto sorted_hidden = gather0(hidden, gather0(token_src, perm));

        // rows of each expert
        auto sorted_ids = gather0(flat_ids, perm);
        auto ones = std::make_shared<ov::op::v3::Broadcast>(ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                                                            std::make_shared<ov::op::v3::ShapeOf>(perm));
        auto counts = std::make_shared<ov::op::v12::ScatterElementsUpdate>(ov::op::v0::Constant::create(ov::element::i32, {num_experts}, {0}),
                                                                           sorted_ids,
                                                                           ones,
                                                                           ov::op::v0::Constant::create(ov::element::i64, {}, {0}),
                                                                           ov::op::v12::ScatterElementsUpdate::Reduction::SUM);
        auto offsets = std::make_shared<ov::op::v0::CumSum>(counts, ov::op::v0::Constant::create(ov::element::i64, {}, {0}));

        ov::test::utils::InputGenerateData weights_data(-1, 2, 128);
        auto weights = std::make_shared<ov::op::v0::Constant>(
            ov::test::utils::create_and_fill_tensor(ov::element::f16, ov::Shape{num_experts, out_size, hidden_size}, weights_data));
        auto experts_out = std::make_shared<ov::op::v17::GroupedMatMul>(sorted_hidden, weights, offsets);

        // combine: back to the (token, slot) order, weighted by the router weights and summed over the slots
        auto rows = gather0(experts_out, argsort(perm)->output(1));
        auto flat_weights = std::make_shared<ov::op::v1::Reshape>(topk_weights, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
        auto col_weights = std::make_shared<ov::op::v0::Unsqueeze>(flat_weights, ov::op::v0::Constant::create(ov::element::i64, {}, {-1}));
        auto weighted = std::make_shared<ov::op::v1::Multiply>(rows, col_weights);
        auto split = std::make_shared<ov::op::v1::Reshape>(
            weighted,
            ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{-1, top_k, static_cast<int64_t>(out_size)}),
            false);
        auto out = std::make_shared<ov::op::v1::ReduceSum>(split, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), false);
        return std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{hidden, topk_ids, topk_weights}, "MoEScatterReduce");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        // The MoE pipeline is gated on supports_immad.
        const auto caps = core->get_property(targetDevice, ov::device::capabilities);
        if (std::find(caps.begin(), caps.end(), ov::intel_gpu::capability::HW_MATMUL) == caps.end()) {
            GTEST_SKIP() << "MoE pipeline requires a systolic GPU (HW_MATMUL capability)";
        }
        const auto& [input_shapes, top_k, out_size] = GetParam();
        this->m_top_k = top_k;
        inference_precision = ov::element::f16;
        abs_threshold = 0.05f;
        rel_threshold = 0.01f;

        std::vector<ov::Shape> router_shapes;
        for (const auto& shape : input_shapes[0].second) {
            router_shapes.push_back({shape[0], top_k});
        }
        const InputShape router_shape{ov::PartialShape{input_shapes[0].first[0], static_cast<int64_t>(top_k)}, router_shapes};
        init_input_shapes({input_shapes[0], router_shape, router_shape});
        function = build_model(inputDynamicShapes[0], static_cast<int64_t>(top_k), out_size);
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& params = function->get_parameters();
        ov::test::utils::InputGenerateData hidden_data(-1, 2, 128);
        inputs.insert({params[0], ov::test::utils::create_and_fill_tensor(ov::element::f16, target_input_static_shapes[0], hidden_data)});

        // distinct experts per token, as produced by a router TopK
        const size_t tokens = target_input_static_shapes[0][0];
        std::mt19937 gen(static_cast<uint32_t>(tokens));
        std::vector<int64_t> experts(num_experts);
        std::iota(experts.begin(), experts.end(), 0);
        ov::Tensor ids_tensor(params[1]->get_element_type(), target_input_static_shapes[1]);
        for (size_t t = 0; t < tokens; t++) {
            std::shuffle(experts.begin(), experts.end(), gen);
            std::copy_n(experts.begin(), m_top_k, ids_tensor.data<int64_t>() + t * m_top_k);
        }
        inputs.insert({params[1], ids_tensor});

        ov::test::utils::InputGenerateData router_weights_data(0, 1, 256);
        inputs.insert({params[2], ov::test::utils::create_and_fill_tensor(ov::element::f16, target_input_static_shapes[2], router_weights_data)});
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        // The combine tail is fused into one moe_scatter_reduction on the inverse permutation
        ov::test::CheckNumberOfNodesWithType(compiledModel, "moe_scatter_reduction", 1);
        ov::test::CheckNumberOfNodesWithTypes(compiledModel, {"reduce", "eltwise"}, 0);
    }

    size_t m_top_k = 0;
};

TEST_P(MoEScatterReduceTest, Inference) {
    run();
}

// Token counts change between the inferences of one compiled model
const std::vector<std::vector<InputShape>> input_shapes = {
    {{{-1, hidden_size}, {{1, hidden_size}, {30, hidden_size}, {7, hidden_size}}}},
};

// 66 is not a multiple of the kernel vector size
INSTANTIATE_TEST_SUITE_P(smoke_MoEScatterReduce,
                         MoEScatterReduceTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(size_t{2}, size_t{4}),
                                            ::testing::Values(size_t{64}, size_t{66})),
                         MoEScatterReduceTest::getTestCaseName);

}  // namespace
