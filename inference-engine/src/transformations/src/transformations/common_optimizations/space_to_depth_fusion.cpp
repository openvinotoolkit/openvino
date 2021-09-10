// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/space_to_depth_fusion.hpp"

#include <limits>
#include <memory>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>
#include <vector>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::SpaceToDepthFusion, "SpaceToDepthFusion", 0);

using namespace ngraph;

const auto end_max = std::numeric_limits<int64_t>::max();

struct SliceSyntax {
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> stride;

    SliceSyntax() = default;

    operator bool() const {
        return begin.size() > 0 && end.size() > 0 && stride.size() > 0;
    }

    /*
    A -> StridedSlice1 -> B -> StridedSlice2 -> C
        <=>
    A -> StridedSlice3 -> C

    for 1 particular dimension

        StridedSlice1 (b1,e1,s1):   B[i]=A[i*s1+b1] for i*s1+b1<e1
        StridedSlice2 (b2,e2,s2):   C[i]=B[i*s2+b2] for i*s2+b2<e2

        C[i] = A[(i*s2+b2)*s1+b1]
            = A[i*s2*s1 + b2*s1 + b1] for  (i*s2*s1 + b2*s1+b1 < e1) && (i*s2+b2 < e2)
            = A[i*s3 + b3] for i*s3 + b3 < e3

            s3 = s1*s2
            b3 = b1 + b2*s1
            e3 = MIN(e1, e2*s1+b1)
    */
    void fuse_with(const SliceSyntax& s2) {
        auto rank = s2.begin.size();

        // expand rank to match s2
        if (begin.size() < rank) {
            begin.resize(rank, 0);
            end.resize(rank, end_max);
            stride.resize(rank, 1);
        }

        for (size_t i = 0; i < rank; i++) {
            auto new_stride = this->stride[i] * s2.stride[i];
            auto new_begin = this->begin[i] + s2.begin[i] * this->stride[i];
            auto new_end = s2.end[i] * this->stride[i] + this->begin[i];
            if (s2.end[i] >= end_max)  // overflow guard
                new_end = end_max;
            if (new_end > this->end[i])
                new_end = this->end[i];
            this->stride[i] = new_stride;
            this->begin[i] = new_begin;
            this->end[i] = new_end;
        }
    }
};

static SliceSyntax get_syntax(std::shared_ptr<ngraph::opset7::StridedSlice> ss) {
    SliceSyntax s;
    int rank;
    Shape in_shape_max;

    rank = ss->input_value(0).get_partial_shape().rank().get_length();

    if (ss->input_value(0).get_partial_shape().is_static()) {
        in_shape_max = ss->input_value(0).get_shape();
    } else {
        in_shape_max = Shape(rank, end_max);
    }

    const auto& new_axis_mask = ss->get_new_axis_mask();
    const auto& shrink_axis_mask = ss->get_shrink_axis_mask();
    const auto& ellipsis_mask = ss->get_ellipsis_mask();

    // no new, deleted or ellipsis axis is allowed
    for (auto& v : new_axis_mask) {
        if (v == 1)
            return s;
    }
    for (auto& v : shrink_axis_mask) {
        if (v == 1)
            return s;
    }
    for (auto& v : ellipsis_mask) {
        if (v == 1)
            return s;
    }

    auto get_masked_input = [&](int input_id, std::vector<int64_t> mask, int64_t masked_value) {
        std::vector<int64_t> ret;
        auto input =
            std::dynamic_pointer_cast<ngraph::opset7::Constant>(ss->input_value(input_id).get_node_shared_ptr());
        if (!input)
            return ret;

        ret = input->cast_vector<int64_t>();

        for (size_t k = 0; k < mask.size(); k++) {
            if (mask[k] == 1)
                ret[k] = masked_value;
        }
        return ret;
    };

    s.begin = get_masked_input(1, ss->get_begin_mask(), 0);
    s.end = get_masked_input(2, ss->get_end_mask(), end_max);
    for (size_t k = 0; k < in_shape_max.size(); k++) {
        if (s.end[k] >= static_cast<int64_t>(in_shape_max[k]))
            s.end[k] = end_max;
    }

    s.stride.resize(s.begin.size(), 1);
    if (ss->get_input_size() >= 4) {
        auto input = std::dynamic_pointer_cast<ngraph::opset7::Constant>(ss->input_value(3).get_node_shared_ptr());
        if (input)
            s.stride = input->cast_vector<int64_t>();
    }

    return s;
}

ngraph::pass::SpaceToDepthFusion::SpaceToDepthFusion() {
    MATCHER_SCOPE(SpaceToDepthFusion);

    const char* env_p = ::getenv("CROSS_CHECK_TOOL");
    const int cross_check_tool = env_p ? std::stol(env_p) : -1;

    if (cross_check_tool == 0) {
        printf("[%s]: cross_check_tool=%d, skipping.\n", __func__, cross_check_tool);
        return;
    } else {
        printf("[%s]: cross_check_tool=%d, enabled.\n", __func__, cross_check_tool);
    }

    auto concat_pattern = pattern::wrap_type<opset7::Concat>({}, [](const Output<Node>& value) {
        auto concat = std::dynamic_pointer_cast<opset7::Concat>(value.get_node_shared_ptr());
        if (!concat)
            return false;
        return concat->get_axis() == 1;
    });

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto concat = std::dynamic_pointer_cast<opset7::Concat>(pattern_map.at(concat_pattern).get_node_shared_ptr());
        if (!concat)
            return false;

        NodeVector nodes_to_delete{concat};

        const int slice_cnt = concat->get_input_size();
        std::vector<int> slice_order(slice_cnt, 0);
        std::vector<int> slice_from_order(slice_cnt, 0);
        int block_size = 0;
        int rank = 0;
        bool is_ordered = true;
        Output<Node> common_input;

        for (int i = 0; i < slice_cnt; i++) {
            SliceSyntax slice_syntax;
            auto input = concat->get_input_source_output(i);
            auto ss = std::dynamic_pointer_cast<opset7::StridedSlice>(input.get_node_shared_ptr());
            while (ss) {
                nodes_to_delete.push_back(ss);

                auto syntax = get_syntax(ss);
                if (!syntax)
                    return false;

                slice_syntax.fuse_with(syntax);
                input = ss->input_value(0);

                ss = std::dynamic_pointer_cast<opset7::StridedSlice>(input.get_node_shared_ptr());
            }

            // all path concated must originates from same input
            if (!common_input.get_node_shared_ptr())
                common_input = input;

            if (input != common_input)
                return false;

            if (rank == 0)
                rank = slice_syntax.stride.size();

            if (rank == 0)
                return false;

            if (static_cast<int>(slice_syntax.stride.size()) != rank)
                return false;

            // [N, C, D1, D2, ...]
            for (size_t k = 0; k < 2; k++) {
                if (slice_syntax.stride[k] != 1 || slice_syntax.begin[k] != 0 || slice_syntax.end[k] < end_max)
                    return false;
            }

            // check block size consistency
            for (int k = 2; k < rank; k++) {
                if (block_size == 0) {
                    block_size = slice_syntax.stride[k];
                    if (block_size < 2)
                        return false;

                    auto slice_expected = 1;
                    for (int j = 2; j < rank; j++)
                        slice_expected *= block_size;

                    if (slice_expected != slice_cnt)
                        return false;
                }
                if (slice_syntax.stride[k] != block_size)
                    return false;
                if (slice_syntax.end[k] < end_max)
                    return false;

                slice_order[i] = slice_order[i] * block_size + slice_syntax.begin[k];
            }

            if (slice_order[i] != i)
                is_ordered = false;

            if (slice_order[i] >= slice_cnt) {
                printf("ERROR slice_order[i]=%d\n", slice_order[i]);
                return false;
            }
            slice_from_order[slice_order[i]] = i;
        }

        if (is_ordered) {
            std::shared_ptr<Node> new_root =
                register_new_node<opset7::SpaceToDepth>(common_input,
                                                        opset7::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
                                                        block_size);

            new_root->set_friendly_name(concat->get_friendly_name());
            copy_runtime_info(nodes_to_delete, new_root);
            replace_node(m.get_match_root(), new_root);
        } else {
            // if output is connected to a Convolution node, channel re-order can be further fused
            // into weights
            bool b_further_opt = true;
            for (auto input_to : concat->get_default_output().get_target_inputs()) {
                auto conv = std::dynamic_pointer_cast<opset7::Convolution>(input_to.get_node()->shared_from_this());
                if (!conv) {
                    b_further_opt = false;
                    break;
                }
                auto filters = std::dynamic_pointer_cast<opset7::Constant>(conv->get_input_node_shared_ptr(1));
                if (!filters) {
                    b_further_opt = false;
                    break;
                }
            }

            if (!b_further_opt)
                return false;

            std::shared_ptr<Node> new_root =
                register_new_node<opset7::SpaceToDepth>(common_input,
                                                        opset7::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
                                                        block_size);

            new_root->set_friendly_name(concat->get_friendly_name());
            copy_runtime_info(nodes_to_delete, new_root);

            // add slplit & concat to Convolution's weights, const-folding will eliminate them later
            for (auto input_to : concat->get_default_output().get_target_inputs()) {
                auto conv = std::dynamic_pointer_cast<opset7::Convolution>(input_to.get_node()->shared_from_this());
                auto filters = std::dynamic_pointer_cast<opset7::Constant>(conv->get_input_node_shared_ptr(1));

                // filters are ordered by slice-order, now re-order them
                auto axis = register_new_node<opset7::Constant>(element::i32, Shape{}, std::vector<int32_t>{1});
                auto split = register_new_node<opset7::Split>(filters, axis, slice_cnt);
                OutputVector reorder;
                for (int i = 0; i < slice_cnt; i++)
                    reorder.push_back(split->output(slice_from_order[i]));
                auto new_filter = register_new_node<opset7::Concat>(reorder, 1);
                replace_node(filters, new_filter);
            }

            replace_node(m.get_match_root(), new_root);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_pattern, matcher_name);
    register_matcher(m, callback);
}