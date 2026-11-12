// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_indexer.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <sstream>

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"

namespace {
using ov::PartialShape;
using ov::op::internal::QSAIndexer;
using ov::op::v0::Parameter;

ov::OutputVector make_inputs() {
    const auto f32 = ov::element::f32;
    return {std::make_shared<Parameter>(f32, PartialShape{2, 3, 2, 16}),
            std::make_shared<Parameter>(f32, PartialShape{2, 3, 1, 16}),
            std::make_shared<Parameter>(f32, PartialShape{2, 3, 8}),
            std::make_shared<Parameter>(f32, PartialShape{2, 3, 8}),
            std::make_shared<Parameter>(ov::element::i32, PartialShape{2}),
            std::make_shared<Parameter>(f32, PartialShape{16}),
            std::make_shared<Parameter>(f32, PartialShape{16}),
            std::make_shared<Parameter>(f32, PartialShape{2, 1, 4, 16}),
            std::make_shared<Parameter>(f32, PartialShape{2, 1, 3, 16}),
            std::make_shared<Parameter>(f32, PartialShape{2, 16})};
}

QSAIndexer::Config make_config() {
    QSAIndexer::Config config;
    config.block_topk = 2;
    config.indexer_head_dim = 16;
    config.indexer_rotary_dim = 8;
    config.epsilon = 1e-5f;
    config.scale = 0.5f;
    return config;
}

void expect_config_eq(const QSAIndexer::Config& actual, const QSAIndexer::Config& expected) {
    EXPECT_EQ(actual.key_block_size, expected.key_block_size);
    EXPECT_EQ(actual.block_topk, expected.block_topk);
    EXPECT_EQ(actual.num_selection_heads, expected.num_selection_heads);
    EXPECT_EQ(actual.indexer_rotary_dim, expected.indexer_rotary_dim);
    EXPECT_EQ(actual.indexer_head_dim, expected.indexer_head_dim);
    EXPECT_EQ(actual.epsilon, expected.epsilon);
    EXPECT_EQ(actual.scale, expected.scale);
}

TEST(QSAIndexerTest, PreservesSharedTypeAndConfigThroughCloneAndIR) {
    const auto config = make_config();
    const auto inputs = make_inputs();
    const auto indexer = std::make_shared<QSAIndexer>(inputs, config);
    EXPECT_EQ(indexer->get_output_size(), QSAIndexer::OutputIdx::COUNT);
    EXPECT_EQ(indexer->get_output_partial_shape(0), (PartialShape{2, 1, 3, 2}));
    EXPECT_EQ(indexer->get_output_partial_shape(1), (PartialShape{2, 1, 3}));
    EXPECT_EQ(indexer->get_index_query_heads(), 2u);
    EXPECT_EQ(indexer->get_index_key_heads(), 1u);

    const auto clone = ov::as_type_ptr<QSAIndexer>(indexer->clone_with_new_inputs(inputs));
    ASSERT_NE(clone, nullptr);
    expect_config_eq(clone->get_config(), config);
    for (size_t i = 0; i < QSAIndexer::OutputIdx::COUNT; ++i) {
        EXPECT_EQ(clone->get_output_partial_shape(i), indexer->get_output_partial_shape(i));
        EXPECT_EQ(clone->get_output_element_type(i), indexer->get_output_element_type(i));
    }

    ov::ParameterVector parameters;
    for (const auto& input : inputs) {
        parameters.push_back(ov::as_type_ptr<Parameter>(input.get_node_shared_ptr()));
    }
    const auto model = std::make_shared<ov::Model>(indexer->outputs(), parameters);
    std::ostringstream xml, bin;
    ov::pass::Serialize(xml, bin).run_on_model(model);
    ov::Core core;
    core.add_extension<QSAIndexer>();
    const auto restored = core.read_model(xml.str(), ov::Tensor{ov::element::u8, ov::Shape{0}});
    const auto loaded = ov::as_type_ptr<QSAIndexer>(restored->get_results()[0]->input_value(0).get_node_shared_ptr());
    ASSERT_NE(loaded, nullptr);
    expect_config_eq(loaded->get_config(), config);
}

TEST(QSAIndexerTest, RejectsDimensionsThatCannotBeRepresented) {
    auto config = make_config();
    config.block_topk = static_cast<size_t>(std::numeric_limits<int64_t>::max()) + 1;
    EXPECT_THROW(std::make_shared<QSAIndexer>(make_inputs(), config), ov::NodeValidationFailure);
    config = make_config();
    config.num_selection_heads = static_cast<size_t>(std::numeric_limits<int64_t>::max()) + 1;
    EXPECT_THROW(std::make_shared<QSAIndexer>(make_inputs(), config), ov::NodeValidationFailure);
}

TEST(QSAIndexerTest, SupportsDynamicShapes) {
    auto inputs = make_inputs();
    inputs[0] = std::make_shared<Parameter>(ov::element::f32, PartialShape::dynamic(4));
    inputs[1] = std::make_shared<Parameter>(ov::element::f32, PartialShape::dynamic(4));
    const auto indexer = std::make_shared<QSAIndexer>(inputs, make_config());
    EXPECT_EQ(indexer->get_output_partial_shape(0), (PartialShape{-1, 1, -1, 2}));
    EXPECT_EQ(indexer->get_output_partial_shape(1), (PartialShape{-1, 1, -1}));
}
}  // namespace
