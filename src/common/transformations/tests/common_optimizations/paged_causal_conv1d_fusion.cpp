// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_causal_conv1d_fusion.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/runtime/core.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;

std::shared_ptr<v0::Parameter> make_i32_param(const std::string& name, const Shape& shape) {
    auto p = std::make_shared<v0::Parameter>(element::i32, shape);
    p->set_friendly_name(name);
    p->get_output_tensor(0).set_names({name});
    return p;
}

std::shared_ptr<ov::Model> build_model(bool add_present_state_result) {
    const Shape input_shape{2, 3};
    const Shape state_shape{2, 3, 4};

    auto input_embeds = std::make_shared<v0::Parameter>(element::f32, input_shape);
    auto past_state = std::make_shared<v0::Parameter>(element::f32, state_shape);
    past_state->get_output_tensor(0).set_names({"cache_params.past.conv.0"});

    auto part_shape = v0::Constant::create(element::i64, Shape{3}, {2, 3, 1});
    auto part0 = std::make_shared<v1::Reshape>(input_embeds, part_shape, false);
    auto part1 = std::make_shared<v1::Reshape>(input_embeds, part_shape, false);
    auto part2 = std::make_shared<v1::Reshape>(input_embeds, part_shape, false);

    auto token_concat = std::make_shared<v0::Concat>(OutputVector{part0, part1, part2}, -1);
    auto transpose_order = v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
    auto token_transpose = std::make_shared<v1::Transpose>(token_concat, transpose_order);

    auto state_concat = std::make_shared<v0::Concat>(OutputVector{past_state, token_transpose}, -1);

    auto weights = v0::Constant::create(element::f32, Shape{3, 1, 1, 4}, std::vector<float>(12, 0.25f));
    auto group_conv = std::make_shared<v1::GroupConvolution>(state_concat,
                                                             weights,
                                                             Strides{1},
                                                             CoordinateDiff{0},
                                                             CoordinateDiff{0},
                                                             Strides{1});

    auto neg_one = v0::Constant::create(element::i64, Shape{1}, {-1});
    auto one = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_begin = std::make_shared<v1::Multiply>(neg_one, one);
    auto slice_end = v0::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto slice_step = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_axis = v0::Constant::create(element::i64, Shape{1}, {2});

    auto slice2 = std::make_shared<v8::Slice>(group_conv, slice_begin, slice_end, slice_step, slice_axis);
    auto swish = std::make_shared<v4::Swish>(slice2);

    auto state_slice = std::make_shared<v8::Slice>(state_concat, slice_begin, slice_end, slice_step, slice_axis);
    ResultVector results;
    results.push_back(std::make_shared<v0::Result>(swish));

    if (add_present_state_result) {
        auto present_res = std::make_shared<v0::Result>(state_slice);
        present_res->get_output_tensor(0).set_names({"cache_params.present.conv.0"});
        results.push_back(present_res);
    }

    auto subsequence_begins = make_i32_param("subsequence_begins", Shape{3});
    auto block_indices = make_i32_param("block_indices", Shape{5});
    auto block_indices_begins = make_i32_param("block_indices_begins", Shape{3});
    auto past_lens = make_i32_param("past_lens", Shape{2});
    auto cache_interval = make_i32_param("cache_interval", Shape{2});

    ParameterVector params{input_embeds,
                           past_state,
                           subsequence_begins,
                           block_indices,
                           block_indices_begins,
                           past_lens,
                           cache_interval};
    return std::make_shared<ov::Model>(results, params);
}

std::shared_ptr<ov::Model> build_model_with_multiply_post_op(bool add_present_state_result) {
    const Shape state_shape{2, 3, 4};
    const Shape token_shape{2, 3, 1};

    auto past_state = std::make_shared<v0::Parameter>(element::f32, state_shape);
    past_state->get_output_tensor(0).set_names({"cache_params.past.conv.0"});

    auto token = std::make_shared<v0::Parameter>(element::f32, token_shape);
    auto gate = std::make_shared<v0::Parameter>(element::f32, token_shape);
    auto token_mul = std::make_shared<v1::Multiply>(token, gate);

    auto state_concat = std::make_shared<v0::Concat>(OutputVector{past_state, token_mul}, -1);

    auto weights = v0::Constant::create(element::f32, Shape{3, 1, 1, 4}, std::vector<float>(12, 0.25f));
    auto group_conv = std::make_shared<v1::GroupConvolution>(state_concat,
                                                             weights,
                                                             Strides{1},
                                                             CoordinateDiff{0},
                                                             CoordinateDiff{0},
                                                             Strides{1});

    auto neg_one = v0::Constant::create(element::i64, Shape{1}, {-1});
    auto one = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_begin = std::make_shared<v1::Multiply>(neg_one, one);
    auto slice_end = v0::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto slice_step = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_axis = v0::Constant::create(element::i64, Shape{1}, {2});

    auto slice2 = std::make_shared<v8::Slice>(group_conv, slice_begin, slice_end, slice_step, slice_axis);
    auto post_op_mul_rhs = std::make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(3));
    auto post_op_mul = std::make_shared<v1::Multiply>(slice2, post_op_mul_rhs);

    auto state_slice = std::make_shared<v8::Slice>(state_concat, slice_begin, slice_end, slice_step, slice_axis);

    ResultVector results;
    results.push_back(std::make_shared<v0::Result>(post_op_mul));
    if (add_present_state_result) {
        auto present_res = std::make_shared<v0::Result>(state_slice);
        present_res->get_output_tensor(0).set_names({"cache_params.present.conv.0"});
        results.push_back(present_res);
    }

    auto subsequence_begins = make_i32_param("subsequence_begins", Shape{3});
    auto block_indices = make_i32_param("block_indices", Shape{5});
    auto block_indices_begins = make_i32_param("block_indices_begins", Shape{3});
    auto past_lens = make_i32_param("past_lens", Shape{2});
    auto cache_interval = make_i32_param("cache_interval", Shape{2});

    ParameterVector params{token,
                           gate,
                           post_op_mul_rhs,
                           past_state,
                           subsequence_begins,
                           block_indices,
                           block_indices_begins,
                           past_lens,
                           cache_interval};
    return std::make_shared<ov::Model>(results, params);
}

std::shared_ptr<ov::Model> build_model_with_converted_weights(bool add_present_state_result) {
    const Shape state_shape{2, 3, 4};
    const Shape token_shape{2, 3, 1};

    auto past_state = std::make_shared<v0::Parameter>(element::f32, state_shape);
    past_state->get_output_tensor(0).set_names({"cache_params.past.conv.0"});

    auto token = std::make_shared<v0::Parameter>(element::f32, token_shape);
    auto gate = std::make_shared<v0::Parameter>(element::f32, token_shape);
    auto token_mul = std::make_shared<v1::Multiply>(token, gate);

    auto state_concat = std::make_shared<v0::Concat>(OutputVector{past_state, token_mul}, -1);

    auto weights_u8 = v0::Constant::create(element::u8, Shape{3, 1, 1, 4}, std::vector<uint8_t>(12, 7));
    auto weights_f32 = std::make_shared<v0::Convert>(weights_u8, element::f32);
    auto group_conv = std::make_shared<v1::GroupConvolution>(state_concat,
                                                             weights_f32,
                                                             Strides{1},
                                                             CoordinateDiff{0},
                                                             CoordinateDiff{0},
                                                             Strides{1});
    group_conv->set_friendly_name("__module.model.model.layers.0.conv/aten::_convolution/GroupConvolution");

    auto neg_one = v0::Constant::create(element::i64, Shape{1}, {-1});
    auto one = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_begin = std::make_shared<v1::Multiply>(neg_one, one);
    auto slice_end = v0::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto slice_step = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_axis = v0::Constant::create(element::i64, Shape{1}, {2});

    auto slice2 = std::make_shared<v8::Slice>(group_conv, slice_begin, slice_end, slice_step, slice_axis);
    auto swish = std::make_shared<v4::Swish>(slice2);

    auto state_slice = std::make_shared<v8::Slice>(state_concat, slice_begin, slice_end, slice_step, slice_axis);
    ResultVector results;
    results.push_back(std::make_shared<v0::Result>(swish));
    if (add_present_state_result) {
        auto present_res = std::make_shared<v0::Result>(state_slice);
        present_res->get_output_tensor(0).set_names({"cache_params.present.conv.0"});
        results.push_back(present_res);
    }

    auto subsequence_begins = make_i32_param("subsequence_begins", Shape{3});
    auto block_indices = make_i32_param("paged_conv_block_indices", Shape{5});
    auto block_indices_begins = make_i32_param("paged_conv_block_indices_begins", Shape{3});
    auto past_lens = make_i32_param("paged_conv_past_lens", Shape{2});
    auto cache_interval = make_i32_param("paged_conv_cache_interval", Shape{2});

    ParameterVector params{token,
                           gate,
                           past_state,
                           subsequence_begins,
                           block_indices,
                           block_indices_begins,
                           past_lens,
                           cache_interval};
    return std::make_shared<ov::Model>(results, params);
}

std::shared_ptr<ov::Model> build_model_without_concat_lfm2_like() {
    const Shape state_shape{2, 3, 4};
    const Shape token_shape{2, 3, 1};

    auto conv_state_hint = std::make_shared<v0::Parameter>(element::f32, state_shape);
    conv_state_hint->set_friendly_name("cache_params_past_conv_hint");
    conv_state_hint->get_output_tensor(0).set_names({"cache_params.past.conv.0"});

    auto token = std::make_shared<v0::Parameter>(element::f32, token_shape);
    auto gate = std::make_shared<v0::Parameter>(element::f32, token_shape);
    auto conv_input = std::make_shared<v1::Multiply>(token, gate);

    auto weights = v0::Constant::create(element::f32, Shape{3, 1, 1, 4}, std::vector<float>(12, 0.25f));
    auto group_conv = std::make_shared<v1::GroupConvolution>(conv_input,
                                                             weights,
                                                             Strides{1},
                                                             CoordinateDiff{2},
                                                             CoordinateDiff{2},
                                                             Strides{1});
    group_conv->set_friendly_name("__module.model.model.layers.0.conv.conv/aten::_convolution/GroupConvolution");

    auto neg_one = v0::Constant::create(element::i64, Shape{1}, {-1});
    auto one = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_begin = std::make_shared<v1::Multiply>(neg_one, one);
    auto slice_end = v0::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto slice_step = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_axis = v0::Constant::create(element::i64, Shape{1}, {2});

    auto slice2 = std::make_shared<v8::Slice>(group_conv, slice_begin, slice_end, slice_step, slice_axis);
    auto swish = std::make_shared<v4::Swish>(slice2);

    auto subsequence_begins = make_i32_param("subsequence_begins", Shape{3});
    auto block_indices = make_i32_param("paged_conv_block_indices", Shape{5});
    auto block_indices_begins = make_i32_param("paged_conv_block_indices_begins", Shape{3});
    auto past_lens = make_i32_param("paged_conv_past_lens", Shape{2});
    auto cache_interval = make_i32_param("paged_conv_cache_interval", Shape{2});

    ParameterVector params{token,
                           gate,
                           conv_state_hint,
                           subsequence_begins,
                           block_indices,
                           block_indices_begins,
                           past_lens,
                           cache_interval};
    return std::make_shared<ov::Model>(ResultVector{std::make_shared<v0::Result>(swish)}, params);
}

std::shared_ptr<ov::Model> build_model_without_concat_lfm2_like_multiple() {
    const Shape state_shape{2, 3, 4};
    const Shape token_shape{2, 3, 1};

    auto conv_state_hint_0 = std::make_shared<v0::Parameter>(element::f32, state_shape);
    conv_state_hint_0->set_friendly_name("cache_params_past_conv_hint_0");
    conv_state_hint_0->get_output_tensor(0).set_names({"cache_params.past.conv.0"});

    auto conv_state_hint_1 = std::make_shared<v0::Parameter>(element::f32, state_shape);
    conv_state_hint_1->set_friendly_name("cache_params_past_conv_hint_1");
    conv_state_hint_1->get_output_tensor(0).set_names({"cache_params.past.conv.1"});

    auto token_0 = std::make_shared<v0::Parameter>(element::f32, token_shape);
    auto gate_0 = std::make_shared<v0::Parameter>(element::f32, token_shape);
    auto conv_input_0 = std::make_shared<v1::Multiply>(token_0, gate_0);

    auto token_1 = std::make_shared<v0::Parameter>(element::f32, token_shape);
    auto gate_1 = std::make_shared<v0::Parameter>(element::f32, token_shape);
    auto conv_input_1 = std::make_shared<v1::Multiply>(token_1, gate_1);

    auto weights_0 = v0::Constant::create(element::f32, Shape{3, 1, 1, 4}, std::vector<float>(12, 0.25f));
    auto group_conv_0 = std::make_shared<v1::GroupConvolution>(conv_input_0,
                                                               weights_0,
                                                               Strides{1},
                                                               CoordinateDiff{2},
                                                               CoordinateDiff{2},
                                                               Strides{1});
    group_conv_0->set_friendly_name("__module.model.model.layers.0.conv.conv/aten::_convolution/GroupConvolution");

    auto weights_1 = v0::Constant::create(element::f32, Shape{3, 1, 1, 4}, std::vector<float>(12, 0.5f));
    auto group_conv_1 = std::make_shared<v1::GroupConvolution>(conv_input_1,
                                                               weights_1,
                                                               Strides{1},
                                                               CoordinateDiff{2},
                                                               CoordinateDiff{2},
                                                               Strides{1});
    group_conv_1->set_friendly_name("__module.model.model.layers.1.conv.conv/aten::_convolution/GroupConvolution");

    auto neg_one = v0::Constant::create(element::i64, Shape{1}, {-1});
    auto one = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_begin = std::make_shared<v1::Multiply>(neg_one, one);
    auto slice_end = v0::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto slice_step = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_axis = v0::Constant::create(element::i64, Shape{1}, {2});

    auto slice_0 = std::make_shared<v8::Slice>(group_conv_0, slice_begin, slice_end, slice_step, slice_axis);
    auto slice_1 = std::make_shared<v8::Slice>(group_conv_1, slice_begin, slice_end, slice_step, slice_axis);

    auto swish_0 = std::make_shared<v4::Swish>(slice_0);
    auto swish_1 = std::make_shared<v4::Swish>(slice_1);

    auto subsequence_begins = make_i32_param("subsequence_begins", Shape{3});
    auto block_indices = make_i32_param("paged_conv_block_indices", Shape{5});
    auto block_indices_begins = make_i32_param("paged_conv_block_indices_begins", Shape{3});
    auto past_lens = make_i32_param("paged_conv_past_lens", Shape{2});
    auto cache_interval = make_i32_param("paged_conv_cache_interval", Shape{2});

    ParameterVector params{token_0,
                           gate_0,
                           token_1,
                           gate_1,
                           conv_state_hint_0,
                           conv_state_hint_1,
                           subsequence_begins,
                           block_indices,
                           block_indices_begins,
                           past_lens,
                           cache_interval};

    ResultVector results{std::make_shared<v0::Result>(swish_0), std::make_shared<v0::Result>(swish_1)};
    return std::make_shared<ov::Model>(results, params);
}

}  // namespace

class PagedCausalConv1DFusionTest : public ::TransformationTestsF {};

TEST_F(PagedCausalConv1DFusionTest, DoesNotFuseWithoutPresentStateResult) {
    model = build_model(false);
    model_ref = build_model(false);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PagedCausalConv1DFusion>();
    manager.run_passes(model);

    compare_functions(model, model_ref);
}

TEST_F(PagedCausalConv1DFusionTest, FusesWhenPresentStateIsResult) {
    model = build_model(true);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PagedCausalConv1DFusion>();
    manager.run_passes(model);

    size_t pcc_count = 0;
    size_t group_conv_count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (std::string(op->get_type_name()) == "PagedCausalConv1D") {
            ++pcc_count;
        }
        if (ov::is_type<ov::op::v1::GroupConvolution>(op)) {
            ++group_conv_count;
        }
    }
    EXPECT_EQ(pcc_count, 1u);
    EXPECT_EQ(group_conv_count, 0u);
}

TEST_F(PagedCausalConv1DFusionTest, FusesMultiplyPostOpAndGenericTokenInput) {
    model = build_model_with_multiply_post_op(true);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PagedCausalConv1DFusion>();
    manager.run_passes(model);

    size_t pcc_count = 0;
    size_t group_conv_count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (std::string(op->get_type_name()) == "PagedCausalConv1D") {
            ++pcc_count;
        }
        if (ov::is_type<ov::op::v1::GroupConvolution>(op)) {
            ++group_conv_count;
        }
    }
    EXPECT_EQ(pcc_count, 1u);
    EXPECT_EQ(group_conv_count, 0u);
}

TEST_F(PagedCausalConv1DFusionTest, FusesNonConcatPatternWithSymmetricPadding) {
    model = build_model_without_concat_lfm2_like();

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PagedCausalConv1DFusion>();
    manager.run_passes(model);

    size_t pcc_count = 0;
    size_t group_conv_count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (std::string(op->get_type_name()) == "PagedCausalConv1D") {
            ++pcc_count;
        }
        if (ov::is_type<ov::op::v1::GroupConvolution>(op)) {
            ++group_conv_count;
        }
    }
    EXPECT_EQ(pcc_count, 1u);
    EXPECT_EQ(group_conv_count, 0u);
}

TEST_F(PagedCausalConv1DFusionTest, FusesSeveralNonConcatPatternsWithSymmetricPadding) {
    model = build_model_without_concat_lfm2_like_multiple();

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PagedCausalConv1DFusion>();
    manager.run_passes(model);

    size_t pcc_count = 0;
    size_t group_conv_count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (std::string(op->get_type_name()) == "PagedCausalConv1D") {
            ++pcc_count;
        }
        if (ov::is_type<ov::op::v1::GroupConvolution>(op)) {
            ++group_conv_count;
        }
    }
    EXPECT_EQ(pcc_count, 2u);
    EXPECT_EQ(group_conv_count, 0u);
}

TEST_F(PagedCausalConv1DFusionTest, FusesWhenWeightsComeFromConvertPath) {
    model = build_model_with_converted_weights(true);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PagedCausalConv1DFusion>();
    manager.run_passes(model);

    size_t pcc_count = 0;
    size_t group_conv_count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (std::string(op->get_type_name()) == "PagedCausalConv1D") {
            ++pcc_count;
        }
        if (ov::is_type<ov::op::v1::GroupConvolution>(op)) {
            ++group_conv_count;
        }
    }
    EXPECT_EQ(pcc_count, 1u);
    EXPECT_EQ(group_conv_count, 0u);
}

TEST(PagedCausalConv1DRealModel, RealModelAfterPATransformation) {
    const char* model_path = std::getenv("OV_PCC_REAL_MODEL_PATH");
    if (!model_path || std::string(model_path).empty()) {
        GTEST_SKIP() << "OV_PCC_REAL_MODEL_PATH is not set";
    }

    ov::Core core;
    auto model = core.read_model(model_path);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::SDPAToPagedAttention>(false, false, false, false, false, false);
    manager.run_passes(model);

    size_t pcc_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (std::string(node->get_type_name()) == "PagedCausalConv1D") {
            ++pcc_count;
        }
    }

    EXPECT_GE(pcc_count, 1u);
}
