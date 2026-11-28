// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/dynamic_pa_tokens.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_set>

#include "model_builder.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

namespace {

std::shared_ptr<ov::Model> make_llm() {
    ov::test::npuw::LLMConfig cfg;
    cfg.num_layers = 1;
    cfg.hidden_size = 64;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.num_kv_heads = 4;
    cfg.vocab_size = 256;
    return ov::test::npuw::ModelBuilder().build_llm(cfg);
}

// The PA block of a variant as NPUW partitions it: the PA op alone, its
// queries, keys and values coming in as static [tokens, heads * head_size].
std::shared_ptr<ov::Model> make_pa_block(std::size_t tokens) {
    auto model = make_llm();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::SDPAToPagedAttention>();
    manager.run_passes(model);

    std::shared_ptr<ov::Node> pa;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::PagedAttentionExtension>(op)) {
            pa = op;
            break;
        }
    }
    ov::ParameterVector params;
    for (std::size_t i = 0; i < pa->get_input_size(); ++i) {
        const auto source = pa->input_value(i);
        if (ov::op::util::is_constant(source.get_node())) {
            continue;
        }
        auto shape = source.get_partial_shape();
        if (i < 3) {
            shape[0] = static_cast<int64_t>(tokens);
        }
        auto param = std::make_shared<ov::op::v0::Parameter>(source.get_element_type(), shape);
        pa->input(i).replace_source_output(param);
        params.push_back(param);
    }
    pa->set_friendly_name("pa");
    pa->output(0).get_tensor().set_names({"attention"});
    auto block = std::make_shared<ov::Model>(ov::OutputVector{pa->output(0)}, params);
    block->validate_nodes_and_infer_types();
    return block;
}

TEST(DynamicPATokensTest, RelaxesTheTokenCountAndKeepsTheOutput) {
    const auto block = make_pa_block(16);
    const auto out_shape = block->output(0).get_partial_shape();
    ASSERT_TRUE(out_shape.is_static());

    EXPECT_TRUE(ov::npuw::DynamicPATokens().run_on_model(block));
    for (std::size_t i = 0; i < 3; ++i) {
        const auto shape = block->get_parameters()[i]->get_partial_shape();
        EXPECT_TRUE(shape[0].is_dynamic()) << "input " << i;
        EXPECT_TRUE(shape[1].is_static()) << "input " << i;
    }
    // The output keeps its shape and its names behind a Reshape.
    const auto source = block->get_results()[0]->get_input_node_ptr(0);
    EXPECT_TRUE(ov::is_type<ov::op::v1::Reshape>(source));
    EXPECT_EQ(block->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(block->output(0).get_names(), (std::unordered_set<std::string>{"attention"}));
    EXPECT_EQ(source->get_friendly_name(), "pa");

    // A second run finds nothing left to relax.
    EXPECT_FALSE(ov::npuw::DynamicPATokens().run_on_model(block));
}

TEST(DynamicPATokensTest, LeavesAModelWithoutPAAlone) {
    const auto model = make_llm();
    const auto ops = model->get_ops().size();
    EXPECT_FALSE(ov::npuw::DynamicPATokens().run_on_model(model));
    EXPECT_EQ(model->get_ops().size(), ops);
}

}  // namespace
