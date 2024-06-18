// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/tensor_list_ops_resolver.hpp"

#include "helper_ops/tensor_list_ops.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace ov::op;
using namespace ov::frontend;

namespace {
ov::Rank find_element_rank(const ov::Input<ov::Node>& target_input) {
    const auto& curr_node = target_input.get_node()->shared_from_this();
    if (const auto& tensor_list_set_item = ov::as_type_ptr<tensorflow::TensorListSetItem>(curr_node)) {
        // the third argument to TensorListSetItem is new element to be inserted
        const auto& item = tensor_list_set_item->input_value(2);
        return item.get_partial_shape().rank();
    } else if (const auto& multi_sub_graph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(curr_node)) {
        auto input_port_idx = static_cast<uint64_t>(target_input.get_index());
        int num_body_graphs = static_cast<int>(multi_sub_graph_op->get_functions().size());
        for (int body_idx = 0; body_idx < num_body_graphs; ++body_idx) {
            const auto& body_graph = multi_sub_graph_op->get_function(body_idx);
            const auto& input_desc = multi_sub_graph_op->get_input_descriptions(body_idx);

            // find parameter index that corresponds to external input target_input
            std::shared_ptr<v0::Parameter> param = nullptr;
            for (const auto& desc : input_desc) {
                if (desc && desc->m_input_index == input_port_idx) {
                    param = body_graph->get_parameters()[desc->m_body_parameter_index];
                    break;
                }
            }
            if (!param) {
                continue;
            }

            // walk through all consumer inputs and try to deduce element rank
            for (const auto& target_input : param->get_output_target_inputs(0)) {
                auto element_rank = find_element_rank(target_input);
                if (element_rank.is_static()) {
                    return element_rank;
                }
            }
        }
    }

    return ov::Rank::dynamic();
}
}  // namespace

ov::frontend::tensorflow::pass::TensorListReplacer::TensorListReplacer() {
    auto tensor_list_label = pattern::wrap_type<TensorList>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry rg;

        auto tensor_list = std::dynamic_pointer_cast<TensorList>(m.get_match_root());
        if (!tensor_list) {
            return false;
        }

        auto num_elements = tensor_list->get_num_elements();
        auto element_dtype = tensor_list->get_element_type();

        if (tensor_list->get_element_rank().is_dynamic()) {
            // try to deduce element rank using successive operations
            ov::Rank element_rank = ov::Rank::dynamic();
            for (const auto& target_input : tensor_list->get_output_target_inputs(0)) {
                element_rank = find_element_rank(target_input);
                if (element_rank.is_static()) {
                    break;
                }
            }
            tensor_list->set_element_rank(element_rank);
        }

        if (tensor_list->get_element_rank().is_static()) {
            size_t element_rank = static_cast<size_t>(tensor_list->get_element_rank().get_length());
            auto initial_element_shape = rg.make<v0::Constant>(element::i32, Shape{element_rank}, 1);

            auto initial_tensor_list_shape = rg.make<v0::Concat>(OutputVector{num_elements, initial_element_shape}, 0);
            auto one_element = rg.make<v0::Constant>(element_dtype, Shape{}, 0);

            // create initial container of tensors with zeros and a shape equal to [num_elements, 1, ..., 1]
            auto initial_tensor_list = rg.make<v1::Broadcast>(one_element, initial_tensor_list_shape);

            // preserve names of the node and the output tensor
            initial_tensor_list->set_friendly_name(tensor_list->get_friendly_name());
            copy_runtime_info(tensor_list, rg.get());

            ov::replace_node(tensor_list, ov::OutputVector{initial_tensor_list->output(0)});
            return true;
        }

        return true;
    };

    auto m =
        std::make_shared<pattern::Matcher>(tensor_list_label, "ov::frontend::tensorflow::pass::TensorListReplacer");
    register_matcher(m, callback);
}

ov::frontend::tensorflow::pass::TensorListSetItemReplacer::TensorListSetItemReplacer() {
    auto tensor_list_set_item_label = pattern::wrap_type<TensorListSetItem>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry rg;

        auto tensor_list_set_item = std::dynamic_pointer_cast<TensorListSetItem>(m.get_match_root());
        if (!tensor_list_set_item) {
            return false;
        }

        auto input_handle = tensor_list_set_item->input_value(0);
        auto index = tensor_list_set_item->input_value(1);
        auto item = tensor_list_set_item->input_value(2);

        // make index be of a shape [1]
        if (index.get_partial_shape() != PartialShape{1}) {
            auto new_index_shape = rg.make<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{1});
            index = rg.make<v1::Reshape>(index, new_index_shape, false);
        }

        // compute the current length of the list
        Output<Node> list_length = rg.make<v3::ShapeOf>(input_handle, element::i32);
        auto zero_const = rg.make<v0::Constant>(element::i32, Shape{1}, 0);
        auto one_const = rg.make<v0::Constant>(element::i32, Shape{1}, 1);
        list_length = rg.make<v8::Slice>(list_length, zero_const, one_const, one_const);

        // compute element shape of real elements to be inserted into the list
        auto item_shape = rg.make<v3::ShapeOf>(item, element::i32);

        // broadcast tensor list container to the real shape
        // since the initial state has shape [num_elements, 1, ..., 1]
        auto target_shape = rg.make<v0::Concat>(OutputVector{list_length, item_shape}, 0);
        input_handle = rg.make<v1::Broadcast>(input_handle, target_shape);

        // reshape item before insertion to source tensor
        item = rg.make<v0::Unsqueeze>(item, zero_const);

        // update the resulted tensor using ScatterUpdate
        auto scatter_update = rg.make<v3::ScatterUpdate>(input_handle, index, item, zero_const);

        // preserve names of the node and the output tensor
        scatter_update->set_friendly_name(tensor_list_set_item->get_friendly_name());
        copy_runtime_info(tensor_list_set_item, rg.get());

        ov::replace_node(tensor_list_set_item, ov::OutputVector{scatter_update->output(0)});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(tensor_list_set_item_label,
                                                "ov::frontend::tensorflow::pass::TensorListSetItemReplacer");
    register_matcher(m, callback);
}

ov::frontend::tensorflow::pass::TensorListGetItemReplacer::TensorListGetItemReplacer() {
    auto tensor_list_get_item_label = pattern::wrap_type<TensorListGetItem>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry rg;

        auto tensor_list_get_item = std::dynamic_pointer_cast<TensorListGetItem>(m.get_match_root());
        if (!tensor_list_get_item) {
            return false;
        }

        auto input_handle = tensor_list_get_item->input_value(0);
        auto index = tensor_list_get_item->input_value(1);
        auto element_dtype = tensor_list_get_item->get_element_type();
        // make index be a scalar
        if (index.get_partial_shape() != PartialShape{}) {
            auto new_index_shape = rg.make<v0::Constant>(element::i32, Shape{0}, vector<int32_t>{});
            index = rg.make<v1::Reshape>(index, new_index_shape, false);
        }

        // gather tensor element by the required position
        auto gather_axis = rg.make<v0::Constant>(element::i32, Shape{1}, 0);
        Output<Node> tensor_element = rg.make<v8::Gather>(input_handle, index, gather_axis);
        tensor_element = rg.make<v0::Convert>(tensor_element, element_dtype);

        // preserve names of the node and the output tensor
        tensor_element.get_node_shared_ptr()->set_friendly_name(tensor_list_get_item->get_friendly_name());
        copy_runtime_info(tensor_list_get_item, rg.get());

        ov::replace_node(tensor_list_get_item, ov::OutputVector{tensor_element});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(tensor_list_get_item_label,
                                                "ov::frontend::tensorflow::pass::TensorListGetItemReplacer");
    register_matcher(m, callback);
}
