// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_qsa_indexer.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <sstream>

#include "openvino/op/parameter.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"
#include "visitors/visitors.hpp"

using ov::PartialShape;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

namespace {
ov::OutputVector make_paged_indexer_args() {
    const auto f32 = ov::element::f32;
    const auto i32 = ov::element::i32;
    // Six tokens, H_iq = 2, H_ik = 1, D_idx = 4, two rotary dims, eight physical pages of eight slots (r = 4).
    return {std::make_shared<Parameter>(f32, PartialShape{6, 8}),
            std::make_shared<Parameter>(f32, PartialShape{6, 4}),
            std::make_shared<Parameter>(f32, PartialShape{6, 2}),
            std::make_shared<Parameter>(f32, PartialShape{6, 2}),
            std::make_shared<Parameter>(f32, PartialShape{4}),
            std::make_shared<Parameter>(f32, PartialShape{4}),
            std::make_shared<Parameter>(i32, PartialShape{2}),
            std::make_shared<Parameter>(i32, PartialShape{3}),
            std::make_shared<Parameter>(i32, PartialShape{4}),
            std::make_shared<Parameter>(i32, PartialShape{3}),
            std::make_shared<Parameter>(f32, PartialShape{8, 1, 8, 4}),
            std::make_shared<Parameter>(f32, PartialShape{8, 2, 4}),
            std::make_shared<Parameter>(f32, PartialShape{8, 1, 2, 4})};
}
}  // namespace

TEST(attributes, paged_qsa_indexer) {
    using ov::op::internal::PagedQSAIndexer;
    NodeBuilder::opset().insert<PagedQSAIndexer>();
    const auto args = make_paged_indexer_args();

    PagedQSAIndexer::Config config;
    config.key_block_size = 4;
    config.block_topk = 3;
    config.num_selection_heads = 1;
    config.indexer_rotary_dim = 2;
    config.indexer_head_dim = 4;
    config.epsilon = 1e-5f;
    config.scale = 0.5f;
    config.query_norm = false;
    config.query_rope = false;
    const auto op = std::make_shared<PagedQSAIndexer>(args, config);

    NodeBuilder builder(op, args);
    const auto restored = ov::as_type_ptr<PagedQSAIndexer>(builder.create());
    ASSERT_NE(restored, nullptr);
    const auto& restored_config = restored->get_config();
    EXPECT_EQ(restored_config.key_block_size, config.key_block_size);
    EXPECT_EQ(restored_config.block_topk, config.block_topk);
    EXPECT_EQ(restored_config.num_selection_heads, config.num_selection_heads);
    EXPECT_EQ(restored_config.indexer_rotary_dim, config.indexer_rotary_dim);
    EXPECT_EQ(restored_config.indexer_head_dim, config.indexer_head_dim);
    EXPECT_EQ(restored_config.epsilon, config.epsilon);
    EXPECT_EQ(restored_config.scale, config.scale);
    EXPECT_EQ(restored_config.query_norm, config.query_norm);
    EXPECT_EQ(restored_config.query_rope, config.query_rope);
    EXPECT_EQ(restored->get_output_size(), 2);
    const auto clone = ov::as_type_ptr<PagedQSAIndexer>(restored->clone_with_new_inputs(restored->input_values()));
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->get_config().query_norm, config.query_norm);
    EXPECT_EQ(clone->get_config().query_rope, config.query_rope);
}

TEST(serialize, paged_qsa_indexer_legacy_query_flags) {
    using ov::op::internal::PagedQSAIndexer;
    const auto args = make_paged_indexer_args();
    PagedQSAIndexer::Config config;
    config.block_topk = 3;
    config.indexer_rotary_dim = 2;
    config.indexer_head_dim = 4;
    config.query_norm = false;
    config.query_rope = false;
    const auto indexer = std::make_shared<PagedQSAIndexer>(args, config);
    ov::ParameterVector parameters;
    for (const auto& arg : args) {
        parameters.push_back(ov::as_type_ptr<Parameter>(arg.get_node_shared_ptr()));
    }
    const auto model =
        std::make_shared<ov::Model>(ov::OutputVector{indexer->output(0), indexer->output(1)}, parameters);
    std::ostringstream xml, bin;
    ov::pass::Serialize(xml, bin).run_on_model(model);
    const auto weights_data = bin.str();
    ov::Tensor weights(ov::element::u8, ov::Shape{weights_data.size()});
    if (!weights_data.empty()) {
        std::memcpy(weights.data(), weights_data.data(), weights_data.size());
    }

    ov::Core core;
    core.add_extension<PagedQSAIndexer>();
    const auto find_indexer = [](const std::shared_ptr<ov::Model>& loaded) {
        for (const auto& node : loaded->get_ops()) {
            if (const auto found = ov::as_type_ptr<PagedQSAIndexer>(node)) {
                return found;
            }
        }
        return std::shared_ptr<PagedQSAIndexer>{};
    };
    const auto current = find_indexer(core.read_model(xml.str(), weights));
    ASSERT_NE(current, nullptr);
    EXPECT_FALSE(current->get_config().query_norm);
    EXPECT_FALSE(current->get_config().query_rope);

    auto legacy_xml = xml.str();
    for (const std::string attribute : {"query_norm", "query_rope"}) {
        const auto start = legacy_xml.find(" " + attribute + "=\"");
        ASSERT_NE(start, std::string::npos);
        const auto end = legacy_xml.find('"', start + attribute.size() + 3);
        ASSERT_NE(end, std::string::npos);
        legacy_xml.erase(start, end - start + 1);
    }
    const auto legacy = find_indexer(core.read_model(legacy_xml, weights));
    ASSERT_NE(legacy, nullptr);
    EXPECT_TRUE(legacy->get_config().query_norm);
    EXPECT_TRUE(legacy->get_config().query_rope);
}
