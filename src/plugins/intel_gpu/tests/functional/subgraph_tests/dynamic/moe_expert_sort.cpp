// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <numeric>
#include <random>

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

using ov::test::InputShape;

using MoEExpertSortTestParams = std::tuple<std::vector<InputShape>,  // hidden states and router ids
                                           ov::element::Type,        // router ids type
                                           size_t>;                  // number of experts

// A grouped experts forward of a MoE block as exported from transformers (`grouped_mm_experts_forward`):
// the (token, slot) rows are sorted by expert with an argsort, GroupedMatMul computes the rows of each expert
// and the rows are gathered back with the argsort of the permutation.
class MoEExpertSortTest : public testing::WithParamInterface<MoEExpertSortTestParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoEExpertSortTestParams>& info) {
        const auto& [input_shapes, ids_type, num_experts] = info.param;
        using ov::test::operator<<;
        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : input_shapes) {
            result << shape << "_";
        }
        result << "ids=" << ids_type << "_E=" << num_experts;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [input_shapes, ids_type, num_experts] = GetParam();
        inference_precision = ov::element::f16;
        abs_threshold = 0.05f;
        rel_threshold = 0.01f;
        init_input_shapes(input_shapes);

        auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        auto topk_ids = std::make_shared<ov::op::v0::Parameter>(ids_type, inputDynamicShapes[1]);
        const auto [perm, offsets, token, inv_perm] = ov::test::build_moe_expert_sort_subgraph(topk_ids, num_experts);
        auto axis0 = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
        auto sorted_hidden = std::make_shared<ov::op::v8::Gather>(hidden, token, axis0);

        const auto hidden_size = static_cast<size_t>(inputDynamicShapes[0][1].get_length());
        ov::test::utils::InputGenerateData weights_data(-1, 2, 128);
        auto weights = std::make_shared<ov::op::v0::Constant>(
            ov::test::utils::create_and_fill_tensor(ov::element::f16, ov::Shape{num_experts, hidden_size, hidden_size}, weights_data));
        auto experts_out = std::make_shared<ov::op::v17::GroupedMatMul>(sorted_hidden, weights, offsets);
        auto out = std::make_shared<ov::op::v8::Gather>(experts_out, inv_perm, axis0);
        function = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{hidden, topk_ids}, "MoEExpertSort");
    }

    // The router ids must be distinct experts of [0, E) for each token, as the router TopK gives:
    // random ids out of [0, E) would make the reference ScatterElementsUpdate and GroupedMatMul read out of bounds
    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& params = function->get_parameters();
        ov::test::utils::InputGenerateData hidden_data(-1, 2, 128);
        inputs.insert({params[0], ov::test::utils::create_and_fill_tensor(ov::element::f16, target_input_static_shapes[0], hidden_data)});

        const size_t num_experts = std::get<2>(GetParam());
        const size_t tokens = target_input_static_shapes[1][0];
        const size_t top_k = target_input_static_shapes[1][1];
        std::mt19937 gen(static_cast<uint32_t>(tokens));
        std::vector<int32_t> experts(num_experts);
        std::vector<int32_t> ids(tokens * top_k);
        std::iota(experts.begin(), experts.end(), 0);
        for (size_t t = 0; t < tokens; t++) {
            std::shuffle(experts.begin(), experts.end(), gen);
            std::copy_n(experts.begin(), top_k, ids.begin() + t * top_k);
        }
        ov::Tensor ids_tensor(params[1]->get_element_type(), target_input_static_shapes[1]);
        for (size_t i = 0; i < ids.size(); i++) {
            if (ids_tensor.get_element_type() == ov::element::i64) {
                ids_tensor.data<int64_t>()[i] = ids[i];
            } else {
                ids_tensor.data<int32_t>()[i] = ids[i];
            }
        }
        inputs.insert({params[1], ids_tensor});
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        // The sort chain is fused into one moe_mask_gen
        ov::test::CheckNumberOfNodesWithType(compiledModel, "moe_mask_gen", 1);
        ov::test::CheckNumberOfNodesWithTypes(compiledModel, {"cum_sum", "scatter_elements_update", "arg_max_min"}, 0);
    }
};

TEST_P(MoEExpertSortTest, Inference) {
    run();
}

// Token counts change between the inferences of one compiled model
const std::vector<std::vector<InputShape>> input_shapes = {
    {{{-1, 128}, {{1, 128}, {30, 128}, {7, 128}}}, {{-1, 4}, {{1, 4}, {30, 4}, {7, 4}}}},
    {{{-1, 64}, {{5, 64}, {1, 64}, {64, 64}}}, {{-1, 8}, {{5, 8}, {1, 8}, {64, 8}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MoEExpertSort,
                         MoEExpertSortTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes), ::testing::Values(ov::element::i32, ov::element::i64), ::testing::Values(8, 64)),
                         MoEExpertSortTest::getTestCaseName);

}  // namespace
