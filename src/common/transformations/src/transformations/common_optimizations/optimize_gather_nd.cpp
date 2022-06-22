// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/optimize_gather_nd.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <openvino/op/util/gather_nd_base.hpp>

#include "itt.hpp"

ov::pass::OptimizerGatherND::OptimizerGatherND() {
    MATCHER_SCOPE(OptimizerGatherND);
    auto gather_nd =
        pattern::wrap_type<ov::op::util::GatherNDBase>({pattern::any_input(), pattern::wrap_type<op::v0::Constant>()});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto gather_nd_node = std::dynamic_pointer_cast<ov::op::util::GatherNDBase>(m.get_match_root());
        if (!gather_nd_node)
            return false;

        const auto original_indices = gather_nd_node->get_input_source_output(1);
        const auto const_indices_node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(original_indices.get_node_shared_ptr());
        if (!const_indices_node) 
            return false;

        const auto& const_indices_values = const_indices_node->cast_vector<int64_t>();
        if (const_indices_values.size() == 0)
            return false;

        const auto data = gather_nd_node->get_input_source_output(0);
        const auto data_shape = data.get_shape();

        const auto original_indices_shape = original_indices.get_shape();
        const auto n_dims = original_indices_shape[original_indices_shape.size()-1];

        // check if indices have just one meaningful dimension and all other dimensions of input have size 1
        // for each dimension
        int meaningful_dims_count = 0;
        for (int i = 0; i < n_dims; i++) {
            std::vector<int64_t> dim;

            // get the column values
            int64_t column_element_counter = i;
            while (column_element_counter <= const_indices_values.size()) {
                dim.push_back(const_indices_values[i]);
                column_element_counter += n_dims;
            }

            // check if dimension is meaningful (has non-zeros)
            if (std::count(dim.cbegin(), dim.cend(), 0) != dim.size()) {

                // if is not meaningful, make sure its shape is 1
                if (data_shape[i] != 1)
                    return false;
            }
            else {
                ++meaningful_dims_count;
            }

            if (meaningful_dims_count > 1)
                return false;
        }
        

        //const auto gathernd_indices_node = gather_nd_node->input_value(0).get_node_shared_ptr();

        //const auto gathernd_indices_shape = gathernd_indices_node->get_shape();
        //const auto& gathernd_indices_tensor = gathernd_indices_node->get_output_tensor(0);

        // last shape element for Gather op
        //const auto n_dims = gathernd_indices_shape[gathernd_indices_shape.size()-1];

        
        
        //const auto gathernd_input_shape = gather_nd_node->input_value(1).get_shape();
        
        // can this shape be dynamic?
        //const auto new_shape = 

        //auto& gathernd_indices = gathernd_indices_node.get_tensor();

        auto new_shape_gather_node = std::make_shared<ngraph::opset8::Constant>(gather_nd_node->input_value(1).get_element_type(), Shape{n_dims});

        // pop the last shape element
        gathernd_indices_shape.pop_back();

        // shape for Reshape op
        auto new_shape_node = std::make_shared<ngraph::opset8::Constant>(gather_nd_node->input_value(1).get_element_type(), gathernd_indices_shape);

        // reshape the tensor with new_shape_node
        auto reshape = std::make_shared<ngraph::opset8::Reshape>(gather_nd_node->input_value(0), new_shape_node, true);

        // gather the values from the last dimension
        auto gather =
            std::make_shared<ngraph::opset8::Gather>(reshape,
                                        new_shape_gather_node,
                                        op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{}, {0}),
                                        gather_nd_node->get_batch_dims());

        gather->set_friendly_name(gather_nd_node->get_friendly_name());
        ngraph::copy_runtime_info(gather_nd_node, {reshape, gather});
        ngraph::replace_node(gather_nd_node, gather);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather_nd, matcher_name);
    register_matcher(m, callback);
}
