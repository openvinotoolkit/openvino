// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_embedding_bag_offsets15_downgrade.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset15.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

namespace {

std::shared_ptr<ov::Model> create_v15_model(const op::v15::EmbeddingBagOffsets::Reduction reduction_type,
                                            const size_t num_inputs) {
    const auto emb_table = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{5, 2});
    const auto indices = std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::Shape{4});
    const auto offsets = std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::Shape{3});
    const auto default_index = std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::Shape{});
    const auto per_sample_weights = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{4});
    std::shared_ptr<ov::op::v15::EmbeddingBagOffsets> emb;
    ov::ParameterVector params;
    switch (num_inputs) {
    case 0:
        emb = std::make_shared<ov::opset15::EmbeddingBagOffsets>();
        params = {};
        break;
    case 3:
        emb = std::make_shared<ov::opset15::EmbeddingBagOffsets>(emb_table, indices, offsets, reduction_type);
        params = {emb_table, indices, offsets};
        break;
    case 4:
        emb = std::make_shared<ov::opset15::EmbeddingBagOffsets>(emb_table,
                                                                 indices,
                                                                 offsets,
                                                                 default_index,
                                                                 reduction_type);
        params = {emb_table, indices, offsets, default_index};
        break;
    case 5:
        emb = std::make_shared<ov::opset15::EmbeddingBagOffsets>(emb_table,
                                                                 indices,
                                                                 offsets,
                                                                 default_index,
                                                                 per_sample_weights,
                                                                 reduction_type);
        params = {emb_table, indices, offsets, default_index, per_sample_weights};
        break;
    }

    emb->set_friendly_name("emb15");

    return std::make_shared<ov::Model>(emb->outputs(), params);
}

std::shared_ptr<ov::Model> create_v3_model(const size_t num_inputs) {
    const auto emb_table = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{5, 2});
    const auto indices = std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::Shape{4});
    const auto offsets = std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::Shape{3});
    const auto default_index = std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::Shape{});
    const auto per_sample_weights = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{4});
    std::shared_ptr<ov::op::v3::EmbeddingBagOffsetsSum> emb;
    ov::ParameterVector params;
    switch (num_inputs) {
    case 0:
        emb = std::make_shared<ov::opset3::EmbeddingBagOffsetsSum>();
        params = {};
        break;
    case 3:
        emb = std::make_shared<ov::opset3::EmbeddingBagOffsetsSum>(emb_table, indices, offsets);
        params = {emb_table, indices, offsets};
        break;
    case 4:
        emb = std::make_shared<ov::opset3::EmbeddingBagOffsetsSum>(emb_table, indices, offsets, default_index);
        params = {emb_table, indices, offsets, default_index};
        break;
    case 5:
        emb = std::make_shared<ov::opset3::EmbeddingBagOffsetsSum>(emb_table,
                                                                   indices,
                                                                   offsets,
                                                                   default_index,
                                                                   per_sample_weights);
        params = {emb_table, indices, offsets, default_index, per_sample_weights};
        break;
    }

    emb->set_friendly_name("emb3");

    return std::make_shared<ov::Model>(emb->outputs(), params);
}

}  // namespace

TEST_F(TransformationTestsF, ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3_sum_0) {
    manager.register_pass<ov::pass::ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3>();
    model = create_v15_model(op::v15::EmbeddingBagOffsets::Reduction::SUM, 0);
    model_ref = create_v3_model(0);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3_sum_3) {
    manager.register_pass<ov::pass::ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3>();
    model = create_v15_model(op::v15::EmbeddingBagOffsets::Reduction::SUM, 3);
    model_ref = create_v3_model(3);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3_sum_4) {
    manager.register_pass<ov::pass::ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3>();
    model = create_v15_model(op::v15::EmbeddingBagOffsets::Reduction::SUM, 4);
    model_ref = create_v3_model(4);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3_sum_5) {
    manager.register_pass<ov::pass::ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3>();
    model = create_v15_model(op::v15::EmbeddingBagOffsets::Reduction::SUM, 5);
    model_ref = create_v3_model(5);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3_mean_0) {
    manager.register_pass<ov::pass::ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3>();
    model = create_v15_model(op::v15::EmbeddingBagOffsets::Reduction::MEAN, 0);
}

TEST_F(TransformationTestsF, ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3_mean_3) {
    manager.register_pass<ov::pass::ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3>();
    model = create_v15_model(op::v15::EmbeddingBagOffsets::Reduction::MEAN, 3);
}

TEST_F(TransformationTestsF, ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3_mean_4) {
    manager.register_pass<ov::pass::ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3>();
    model = create_v15_model(op::v15::EmbeddingBagOffsets::Reduction::MEAN, 4);
}
