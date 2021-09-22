// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/depth_to_space_fusion.hpp"

#include <math.h>

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::DepthToSpaceFusion, "DepthToSpaceFusion", 0);
#define DEPTH_TO_SPACE_RANK_MIN 3

namespace {

enum class SHAPEORTRANS { NEITHER = -1, RESHAPE, TRANSPOSE };

class axis_data {
public:
    axis_data(int32_t idx = -1, size_t axis_len = 0, int32_t origin = -1)
        : m_order_idx(idx),
          m_axis_length(axis_len),
          m_origin_order(origin) {}
    axis_data(const axis_data&) = default;

    const int32_t m_order_idx;
    const size_t m_axis_length;
    const int32_t m_origin_order;
};

class axis {
public:
    axis(axis_data& data) : m_vec(std::vector<axis_data>{data}) {}
    axis(std::vector<axis_data>& vec) : m_vec(vec) {}

    const size_t get_axis_size() const {
        return m_vec.size();
    }
    const int32_t get_axis_order(int32_t idx = 0) const {
        return m_vec[idx].m_order_idx;
    }

    const int32_t get_axis_original_order(int32_t idx = 0) const {
        return m_vec[idx].m_origin_order;
    }

    const size_t get_axis_length(int32_t idx = 0) const {
        return m_vec[idx].m_axis_length;
    }

    const axis_data& get_axis_data(int32_t idx = 0) const {
        return m_vec[idx];
    }

private:
    std::vector<axis_data> m_vec;
};

class axis_state {
public:
    axis_state(const ngraph::Shape& shape, size_t blk_size) : m_block_size(blk_size), m_orig_shape(shape) {
        int32_t order_index = 0;
        for (auto indice_len : shape) {
            auto default_axis = axis_data{order_index, indice_len, -1};
            m_axis_vec.push_back(std::make_shared<axis>(default_axis));
            order_index++;
        }
    }

    bool transpose(const ov::AxisVector& axises);
    bool reshape(const ngraph::Shape& before, const ngraph::Shape& after);
    bool check_fusion(uint8_t& is_depth_first) const;

private:
    std::vector<std::shared_ptr<axis>> m_axis_vec;
    const size_t m_block_size;
    const ngraph::Shape m_orig_shape;

    bool axis_split(const std::vector<size_t>& before, const std::vector<size_t>& after);
    bool axis_combine(const std::vector<size_t>& before, const std::vector<size_t>& after);
};

bool axis_state::transpose(const ov::AxisVector& axises) {
    const std::vector<std::shared_ptr<axis>> axis_vec_bak{m_axis_vec};
    size_t idx = 0;
    if (axises.size() != m_axis_vec.size())
        return false;

    for (size_t axis : axises) {
        if (idx != axis)
            m_axis_vec[idx] = axis_vec_bak[axis];

        idx++;
    }
    return true;
}

bool axis_state::reshape(const ngraph::Shape& before, const ngraph::Shape& after) {
    if (ngraph::shape_size(before) != ngraph::shape_size(after) || before.size() == after.size())
        return false;

    // Only support split all the block dims in one reshape.
    if (after.size() == (before.size() + before.size() - 2))
        return axis_split(before, after);

    // Support combine one/multi block dims one reshape.
    else if (before.size() > after.size())
        return axis_combine(before, after);

    return false;
}

bool axis_state::check_fusion(uint8_t& is_depth_first) const {
    // Assume the dimension would be [N, C, D1, D2, ..., DK]
    auto dim_N = m_axis_vec[0];
    auto dim_C = m_axis_vec[1];
    auto rank = m_orig_shape.size();
    int32_t blk_idx_start = -1;
    int32_t blk_K = 0;
    // Axis of `N` should not divided.
    if (dim_N->get_axis_size() != 1 || dim_N->get_axis_original_order(0) != -1 || dim_N->get_axis_order(0) != 0)
        return false;
    // Current `C` axis should be divided from original `C` axis;
    if (dim_C->get_axis_size() != 1 || dim_C->get_axis_original_order(0) != 1)
        return false;

    auto depth_idx = dim_C->get_axis_order();
    if (depth_idx == 0) {
        // Current `C` axis is the first dim of the dimensions which are divided from original `C`. So depth first
        // maybe.
        is_depth_first = 1;
        blk_idx_start = 1;
    } else if (depth_idx == (int32_t)(rank - 2)) {
        // Current `C` axis is the last dim of the dimensions which are divided from original `C` So block first maybe.
        is_depth_first = 0;
        blk_idx_start = 0;
    } else {
        return false;
    }
    // Check the [D1 * block_size, D2 * block_size, D3 * block_size, ..., DK * block_size]
    for (size_t i = 2; i < m_axis_vec.size(); i++, blk_K++) {
        auto ptr_node = m_axis_vec[i];
        // Combination of 2 axises;
        if (ptr_node->get_axis_size() != 2)
            return false;
        // The first axis is the original `DK` axis
        if (ptr_node->get_axis_original_order(0) != -1 || ptr_node->get_axis_order(0) != (int32_t)i ||
            ptr_node->get_axis_length(0) != m_orig_shape[i])
            return false;
        // The second axis data is the divided block axis.
        if (ptr_node->get_axis_original_order(1) != 1 || ptr_node->get_axis_order(1) != (blk_idx_start + blk_K) ||
            ptr_node->get_axis_length(1) != m_block_size)
            return false;
    }
    return true;
}

bool axis_state::axis_split(const std::vector<size_t>& before, const std::vector<size_t>& after) {
    size_t index_before(0), offset(0);
    size_t block_dims_cnt = before.size() - 2;

    for (; index_before < before.size(); index_before++) {
        if (before[index_before] == after[index_before + offset]) {
            continue;
        } else if (offset == 0) {
            // split only once
            size_t split_out_idx = 0;
            auto res = before[index_before];
            std::vector<std::shared_ptr<axis>> split_vec;

            // The axis to be split can't be one dimension which has been split before.
            if (m_axis_vec[index_before]->get_axis_size() != 1)
                return false;

            // Split out the (block_dims_cnt+1) axises
            for (; split_out_idx <= block_dims_cnt; split_out_idx++) {
                if (res % after[index_before + split_out_idx])
                    return false;

                res = res / after[index_before + split_out_idx];
                auto split_axis = axis_data{(int32_t)split_out_idx,
                                            after[index_before + split_out_idx],
                                            m_axis_vec[index_before]->get_axis_order()};
                split_vec.push_back(std::make_shared<axis>(split_axis));
            }

            if (res != 1)
                return false;

            m_axis_vec.insert(m_axis_vec.begin() + index_before, split_vec.begin(), split_vec.end());
            m_axis_vec.erase(m_axis_vec.begin() + index_before + block_dims_cnt + 1);
            offset = block_dims_cnt;
        } else {
            return false;
        }
    }

    return true;
}

bool axis_state::axis_combine(const std::vector<size_t>& before, const std::vector<size_t>& after) {
    size_t idx_after(0), offset(0);
    std::vector<std::shared_ptr<axis>> vec_copy{};

    for (; idx_after < after.size(); idx_after++) {
        if (before[idx_after + offset] == after[idx_after]) {
            vec_copy.push_back(m_axis_vec[idx_after + offset]);
            continue;
        }

        std::vector<axis_data> combined_axis{};
        size_t combine_idx = offset + idx_after;
        size_t res = 1;
        auto cnt = 0;
        for (; res < after[idx_after]; combine_idx++, cnt++) {
            // only support 2 axises combined into one;
            if (cnt >= 2)
                return false;
            res = res * before[combine_idx];
            combined_axis.push_back(m_axis_vec[combine_idx]->get_axis_data(0));
        }
        // only res == after[idx_after];
        if (res != after[idx_after]) {
            return false;
        }
        vec_copy.push_back(std::make_shared<axis>(combined_axis));
        offset = combine_idx - 1 - idx_after;
    }
    // Update the axis_vec with new.
    m_axis_vec = vec_copy;

    return true;
}

inline bool depth_to_space_fusion_check_shape(const ngraph::Shape& start,
                                              const ngraph::Shape& end,
                                              size_t& block_size) {
    auto rank = start.size();

    if (rank < DEPTH_TO_SPACE_RANK_MIN || rank != end.size() || start[0] != end[0] || start[1] % end[1] != 0 ||
        end[2] % start[2] != 0)
        return false;
    size_t possible_block_size = end[2] / start[2];
    auto divisor = start[1] / end[1];
    if (pow(possible_block_size, rank - 2) != divisor)
        return false;
    for (size_t i = 2; i < rank; i++) {
        if (start[i] * possible_block_size != end[i])
            return false;
    }
    block_size = possible_block_size;
    return true;
}

class tranfrom_param {
public:
    tranfrom_param(SHAPEORTRANS type, const ngraph::Shape& before_reshape, const ngraph::Shape& after_reshape)
        : op_type(type),
          before_reshape(before_reshape),
          after_reshape(after_reshape) {}

    tranfrom_param(SHAPEORTRANS type, const ov::AxisVector& trans_param) : op_type(type), trans_param(trans_param) {}

    const SHAPEORTRANS op_type;
    const ngraph::Shape before_reshape;
    const ngraph::Shape after_reshape;
    const ov::AxisVector trans_param;
};

inline void push_param(SHAPEORTRANS type, std::shared_ptr<ov::Node> node, std::vector<tranfrom_param>& param_vec) {
    if (type == SHAPEORTRANS::RESHAPE) {
        const ngraph::Shape before_reshape = node->get_input_shape(0);
        const ngraph::Shape after_reshape = node->get_output_shape(0);
        param_vec.push_back(tranfrom_param{type, before_reshape, after_reshape});
    } else {
        auto const_input =
            std::dynamic_pointer_cast<ngraph::opset8::Constant>(node->input_values()[1].get_node_shared_ptr());
        const ov::AxisVector axis_vec = const_input->get_axis_vector_val();
        param_vec.push_back(tranfrom_param{type, axis_vec});
    }
}

std::function<bool(const ov::Output<ov::Node>&)> output_not_fed_to_reshapeortrans_and_not_fed_to_multi_node(void) {
    return [=](const ov::Output<ov::Node>& value) -> bool {
        auto transpose_or_reshape_node = value.get_node_shared_ptr();

        // No result? Will have in the actural graph?
        if (!transpose_or_reshape_node)
            return false;

        // The output feeds more than one nodes as input.
        if (transpose_or_reshape_node->get_output_target_inputs(0).size() > 1)
            return true;
        // check the next node type.
        auto next_node = transpose_or_reshape_node->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        if (std::dynamic_pointer_cast<ngraph::opset8::Reshape>(next_node) ||
            std::dynamic_pointer_cast<ngraph::opset8::Transpose>(next_node)) {
            return false;
        }
        return true;
    };
}

std::function<bool(const ov::Output<ov::Node>&)> check_static_shape() {
    return [=](const ov::Output<ov::Node>& output) -> bool {
        auto ret = output.get_partial_shape().is_static();
        return ret;
    };
}

}  // namespace

ngraph::pass::DepthToSpaceFusion::DepthToSpaceFusion() {
    MATCHER_SCOPE(DepthToSpaceFusion);

    auto data_pattern = std::make_shared<pattern::op::Label>(element::f32, ngraph::Shape{}, check_static_shape());

    auto reshape = pattern::wrap_type<opset8::Reshape>({data_pattern, pattern::wrap_type<opset8::Constant>()},
                                                       output_not_fed_to_reshapeortrans_and_not_fed_to_multi_node());
    auto trans = pattern::wrap_type<opset8::Transpose>({data_pattern, pattern::wrap_type<opset8::Constant>()},
                                                       output_not_fed_to_reshapeortrans_and_not_fed_to_multi_node());
    auto reshape_or_transpose_label = std::make_shared<pattern::op::Or>(OutputVector{reshape, trans});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto op_type = SHAPEORTRANS::NEITHER;
        auto expected_op_type = op_type;
        ngraph::NodeVector interleaved_node_vec;
        std::size_t possible_block_size(0);
        std::vector<tranfrom_param> interleaved_param;

        auto get_reshape_or_transpose = [&pattern_to_output](
                                            const std::shared_ptr<Node>& reshape_pattern,
                                            const std::shared_ptr<Node>& trans_pattern) -> std::shared_ptr<Node> {
            if (pattern_to_output.count(reshape_pattern))
                return pattern_to_output.at(reshape_pattern).get_node_shared_ptr();
            if (pattern_to_output.count(trans_pattern))
                return pattern_to_output.at(trans_pattern).get_node_shared_ptr();
            return nullptr;
        };

        auto interleaved_end = get_reshape_or_transpose(reshape, trans);
        if (!interleaved_end)
            return false;

        auto interleaved_begin = interleaved_end;
        auto node_iter = interleaved_begin;

        op_type = (std::dynamic_pointer_cast<ngraph::opset8::Reshape>(interleaved_end)) ? SHAPEORTRANS::RESHAPE
                                                                                        : SHAPEORTRANS::TRANSPOSE;

        expected_op_type = op_type;

        while (expected_op_type == op_type) {
            if (!node_iter->get_input_partial_shape(0).is_static()) {
                return false;
            }
            // update the interleaved beginning and save related parameter
            interleaved_begin = node_iter;
            push_param(op_type, interleaved_begin, interleaved_param);
            interleaved_node_vec.push_back(interleaved_begin);
            // interleaved is expected
            expected_op_type = (op_type == SHAPEORTRANS::TRANSPOSE ? SHAPEORTRANS::RESHAPE : SHAPEORTRANS::TRANSPOSE);

            node_iter = interleaved_begin->input_values()[0].get_node_shared_ptr();
            // When output is fed to several consumers, stop the interleaved checking.
            // Only try fusing the interleave series without branching straightforward.
            if (node_iter->get_output_target_inputs(0).size() > 1) {
                op_type = SHAPEORTRANS::NEITHER;
                break;
            }

            // update optype to be node_iter OP type
            op_type = (std::dynamic_pointer_cast<ngraph::opset8::Reshape>(node_iter) ||
                       std::dynamic_pointer_cast<ngraph::opset8::Transpose>(node_iter))
                          ? ((std::dynamic_pointer_cast<ngraph::opset8::Reshape>(node_iter)) ? SHAPEORTRANS::RESHAPE
                                                                                             : SHAPEORTRANS::TRANSPOSE)
                          : SHAPEORTRANS::NEITHER;
        }
        // not interleaved or series is below 3
        if (op_type != SHAPEORTRANS::NEITHER || interleaved_param.size() < 3)
            return false;

        auto input_shape = interleaved_begin->input_values()[0].get_shape();

        if (!depth_to_space_fusion_check_shape(input_shape, interleaved_end->get_output_shape(0), possible_block_size) &&
            !possible_block_size)
            return false;

        axis_state axis_tracker{input_shape, possible_block_size};

        for (size_t i = 0; i < interleaved_param.size(); i++) {
            auto reverse_idx = interleaved_param.size() - 1 - i;
            auto element = interleaved_param[reverse_idx];
            if (element.op_type == SHAPEORTRANS::RESHAPE) {
                if (!axis_tracker.reshape(element.before_reshape, element.after_reshape))
                    return false;
            } else {
                if (!axis_tracker.transpose(element.trans_param))
                    return false;
            }
        }

        uint8_t depth_first = 0;
        if (!axis_tracker.check_fusion(depth_first))
            return false;

        auto mode = depth_first ? ngraph::opset8::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST
                                : ngraph::opset8::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;

        auto depth_to_space = register_new_node<ngraph::opset8::DepthToSpace>(interleaved_begin->input_value(0),
                                                                             mode,
                                                                             possible_block_size);
        depth_to_space->set_friendly_name(interleaved_end->get_friendly_name());
        std::reverse(interleaved_node_vec.begin(), interleaved_node_vec.end());
        ngraph::copy_runtime_info(interleaved_node_vec, depth_to_space);
        ngraph::replace_node(interleaved_end, depth_to_space);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_or_transpose_label, matcher_name);
    register_matcher(m, callback);
}