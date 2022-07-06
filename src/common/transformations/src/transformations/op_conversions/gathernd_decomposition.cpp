// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//#include "transformations/common_optimizations/optimize_gather_nd.hpp"
#include "transformations/op_conversions/gathernd_decomposition.hpp"

#include <memory>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/op/util/gather_nd_base.hpp>

#include "itt.hpp"

ov::pass::GatherNDDecomposition::GatherNDDecomposition() {
    MATCHER_SCOPE(GatherNDDecomposition);
    auto indices = pattern::wrap_type<op::v0::Constant>();
    auto data = pattern::any_input(pattern::has_static_shape());
    auto gather_nd = pattern::wrap_type<ov::op::util::GatherNDBase>({data, indices});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto gather_nd_node = std::dynamic_pointer_cast<ov::op::util::GatherNDBase>(m.get_match_root());
        if (!gather_nd_node) {
            return false;
        }

        // transformation cannot be applied for non-default batch_dims value
        const auto batch_dims = gather_nd_node->get_batch_dims();
        if (batch_dims != 0) {
            return false;
        }

        // check if indices are given
        const auto& pattern_to_output = m.get_pattern_value_map();
        const auto indices_input =
            std::dynamic_pointer_cast<ngraph::opset8::Constant>(pattern_to_output.at(indices).get_node_shared_ptr());
        if (!indices_input) {
            return false;
        }
        const auto const_indices_values = indices_input->cast_vector<int64_t>();
        if (const_indices_values.size() == 0) {
            return false;
        }

        const auto data_input = pattern_to_output.at(data);
        const auto& data_partial_shape = data_input.get_partial_shape();
        const auto& indices_shape = indices_input->get_shape();
        const auto& data_shape = data_partial_shape.get_shape();
        const auto n_dims = indices_input->get_shape().back();
        std::vector<int64_t> meaningful_indices;

        // check if indices have just one meaningful dimension and all other dimensions of input have size 1
        for (int i = 0; i < n_dims; i++) {
            // get the indices values
            std::vector<int64_t> indices;
            int64_t dim_value_counter = i;
            while (dim_value_counter < const_indices_values.size()) {
                indices.push_back(const_indices_values[dim_value_counter]);
                dim_value_counter += n_dims;
            }
            // check if dimension is non-meaningful (all indices values are zeros)
            if (std::count(indices.cbegin(), indices.cend(), 0) == indices.size()) {
                // if is not meaningful, make sure input tensor's shape is 1
                if (data_shape[i] != 1) {
                    return false;
                }
            } else {
                // if it is meaningful, check if it is the first one found
                if (!meaningful_indices.empty()) {
                    return false;
                }
                std::copy(indices.begin(), indices.end(), std::back_inserter(meaningful_indices));
            }
        }

        // reshape the tensor for Gather node
        std::vector<int64_t> new_shape(data_shape.begin() + n_dims, data_shape.end());
        // fill the rest of the values into the first dimension
        new_shape.insert(new_shape.begin(), -1);
        const auto new_shape_node =
            op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{new_shape.size()}, new_shape);
        auto reshape_node =
            std::make_shared<ngraph::opset8::Reshape>(gather_nd_node->input_value(0), new_shape_node, true);

        // gather the final values
        const auto new_indices_shape = std::vector<size_t>(indices_shape.begin(), indices_shape.end() - 1);
        const auto new_indices_node =
            op::v0::Constant::create<int64_t>(element::Type_t::i64, new_indices_shape, meaningful_indices);
        const auto reshape_output_shape = reshape_node->get_shape();
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(
            reshape_node,
            new_indices_node,
            op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{}, {0}));

        gather_node->set_friendly_name(gather_nd_node->get_friendly_name());
        ngraph::copy_runtime_info(gather_nd_node, {reshape_node, gather_node});
        ngraph::replace_node(gather_nd_node, gather_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather_nd, matcher_name);
    register_matcher(m, callback);
}
