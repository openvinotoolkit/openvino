// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/tensor_list_ops_resolver.hpp"

#include "helper_ops/tensor_list_ops.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace ov::op;
using namespace ov::frontend;
using namespace ov::op::util;

using InvariantD = ov::op::util::MultiSubGraphOp::InvariantInputDescription;
using SlicedD = ov::op::util::MultiSubGraphOp::SliceInputDescription;
using MergedD = ov::op::util::MultiSubGraphOp::MergedInputDescription;
using OutputD = ov::op::util::MultiSubGraphOp::BodyOutputDescription;
using ConcatD = ov::op::util::MultiSubGraphOp::ConcatOutputDescription;

namespace {
ov::Rank find_element_rank(const ov::Input<ov::Node>& target_input, ov::element::Type& element_type) {
    const auto& curr_node = target_input.get_node()->shared_from_this();
    if (const auto& tensor_list_set_item = ov::as_type_ptr<tensorflow::TensorListSetItem>(curr_node)) {
        // the third argument to TensorListSetItem is new element to be inserted
        const auto& item = tensor_list_set_item->input_value(2);
        if (element_type.is_dynamic()) {
            element_type = item.get_element_type();
        }
        return item.get_partial_shape().rank();
    } else if (const auto& tensor_list_push_back = ov::as_type_ptr<tensorflow::TensorListPushBack>(curr_node)) {
        // the second argument to TensorListPushBack is new element to be inserted
        const auto& item = tensor_list_push_back->input_value(1);
        if (element_type.is_dynamic()) {
            element_type = item.get_element_type();
        }
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
                auto element_rank = find_element_rank(target_input, element_type);
                if (element_rank.is_static()) {
                    return element_rank;
                }
            }
        }
    }

    return ov::Rank::dynamic();
}

bool find_input_description(const ov::op::util::InputDescriptionVector& input_descriptions,
                            uint64_t param_idx,
                            ov::op::util::SubGraphOp::InputDescription::Ptr& found_desc) {
    for (const auto& input_desc : input_descriptions) {
        if (input_desc->m_body_parameter_index == param_idx) {
            found_desc = input_desc;
            return true;
        }
    }

    return false;
}

void update_parameter_to_slice_input(const std::shared_ptr<ov::Node>& node,
                                     const std::shared_ptr<ov::Model>& body_graph,
                                     const ov::op::util::InputDescriptionVector& input_descriptions,
                                     std::vector<uint64_t>& update_param_ids) {
    // select only TensorListGetItem that accepts a tensor list from Parameter node
    // value of Parameter node is unchanged from one iteration to another one in Loop
    auto tensor_list_get_item = std::dynamic_pointer_cast<ov::frontend::tensorflow::TensorListGetItem>(node);
    if (!tensor_list_get_item) {
        return;
    }
    auto tensor_list = ov::as_type_ptr<v0::Parameter>(tensor_list_get_item->get_input_node_shared_ptr(0));
    if (!tensor_list) {
        return;
    }
    if (tensor_list->get_output_target_inputs(0).size() != 1) {
        return;
    }

    // tensor list must be invariant through iterations
    int64_t param_idx = body_graph->get_parameter_index(tensor_list);
    if (param_idx < 0) {
        return;
    }
    ov::op::util::SubGraphOp::InputDescription::Ptr input_desc = nullptr;
    if (!find_input_description(input_descriptions, static_cast<uint64_t>(param_idx), input_desc) || !input_desc) {
        return;
    }
    auto invariant_input_desc = ov::as_type_ptr<InvariantD>(input_desc);
    if (!invariant_input_desc) {
        return;
    }

    update_param_ids.push_back(static_cast<uint64_t>(param_idx));
}

void update_result_to_concat_output(const std::shared_ptr<ov::Node>& node,
                                    const std::shared_ptr<ov::Model>& body_graph,
                                    const ov::ResultVector& results,
                                    const ov::op::util::InputDescriptionVector& input_descriptions,
                                    std::vector<uint64_t>& update_result_ids,
                                    std::vector<uint64_t>& remove_param_ids) {
    // select only TensorListSetItem that accepts a tensor list from Parameter node
    // output of TensorListSetItem goes to Result that is connected with the tensor list by a back edge
    auto tensor_list_set_item = std::dynamic_pointer_cast<ov::frontend::tensorflow::TensorListSetItem>(node);
    if (!tensor_list_set_item) {
        return;
    }
    auto tensor_list = ov::as_type_ptr<v0::Parameter>(tensor_list_set_item->get_input_node_shared_ptr(0));
    if (!tensor_list) {
        return;
    }
    if (tensor_list->get_output_target_inputs(0).size() != 1) {
        return;
    }

    int64_t param_idx = body_graph->get_parameter_index(tensor_list);
    if (param_idx < 0) {
        return;
    }
    ov::op::util::SubGraphOp::InputDescription::Ptr input_desc = nullptr;
    if (!find_input_description(input_descriptions, static_cast<uint64_t>(param_idx), input_desc) || !input_desc) {
        return;
    }
    auto merged_input_desc = ov::as_type_ptr<MergedD>(input_desc);
    if (!merged_input_desc) {
        return;
    }

    uint64_t result_idx = merged_input_desc->m_body_value_index;
    if (results[result_idx]->get_input_node_shared_ptr(0) != tensor_list_set_item) {
        return;
    }

    update_result_ids.push_back(result_idx);
    remove_param_ids.push_back(static_cast<uint64_t>(param_idx));
}

uint64_t get_new_param_idx(const std::vector<uint64_t>& remove_parameter_idxs, uint64_t old_idx) {
    // compute a number of Parameters nodes standing before old_idx that will be removed
    uint64_t num_removed = 0;
    for (auto remove_idx : remove_parameter_idxs) {
        FRONT_END_GENERAL_CHECK(old_idx != remove_idx,
                                "[TensorFlow Frontend] internal error: incorrect old_idx for "
                                "TensorListInLoopOptimization transformation");
        if (remove_idx < old_idx) {
            ++num_removed;
        }
    }

    // compute shifted index
    FRONT_END_GENERAL_CHECK(num_removed <= old_idx,
                            "[TensorFlow Frontend] internal error: incorrect new parameter index computation "
                            "TensorListInLoopOptimization transformation");
    return old_idx - num_removed;
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

        if (tensor_list->get_element_rank().is_dynamic() || element_dtype.is_dynamic()) {
            // try to deduce element rank using successive operations
            ov::Rank element_rank = tensor_list->get_element_rank();
            for (const auto& target_input : tensor_list->get_output_target_inputs(0)) {
                element_rank = find_element_rank(target_input, element_dtype);
                if (element_rank.is_static() && element_dtype.is_static()) {
                    break;
                }
            }
            tensor_list->set_element_rank(element_rank);
            tensor_list->set_element_type(element_dtype);
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

ov::frontend::tensorflow::pass::TensorListPushBackReplacer::TensorListPushBackReplacer() {
    auto tensor_list_push_back_label = pattern::wrap_type<TensorListPushBack>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry rg;

        auto tensor_list_push_back = std::dynamic_pointer_cast<TensorListPushBack>(m.get_match_root());
        if (!tensor_list_push_back) {
            return false;
        }

        auto input_handle = tensor_list_push_back->input_value(0);
        auto tensor = tensor_list_push_back->input_value(1);

        // compute tensor shape to be inserted
        auto tensor_shape = rg.make<v3::ShapeOf>(tensor, element::i32);

        // broadcast the tensor list to the shape [num_elements, <tensor_shape>]
        Output<Node> num_elements = rg.make<v3::ShapeOf>(input_handle, element::i32);
        auto zero_const = rg.make<v0::Constant>(element::i32, Shape{1}, 0);
        auto one_const = rg.make<v0::Constant>(element::i32, Shape{1}, 1);
        num_elements = rg.make<v8::Slice>(num_elements, zero_const, one_const, one_const);
        auto new_input_handle_shape =
            rg.make<v0::Concat>(OutputVector{std::move(num_elements), std::move(tensor_shape)}, 0);
        input_handle = rg.make<v1::Broadcast>(input_handle, new_input_handle_shape);

        // unsqueeze tensor to be inserted into the list
        tensor = rg.make<v0::Unsqueeze>(tensor, zero_const);

        // insert the tensor into the end
        auto updated_list = rg.make<v0::Concat>(OutputVector{std::move(input_handle), std::move(tensor)}, 0);

        updated_list->set_friendly_name(tensor_list_push_back->get_friendly_name());
        copy_runtime_info(tensor_list_push_back, rg.get());

        ov::replace_node(tensor_list_push_back, ov::OutputVector{updated_list->output(0)});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(tensor_list_push_back_label,
                                                "ov::frontend::tensorflow::pass::TensorListPushBackReplacer");
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

ov::frontend::tensorflow::pass::TensorListInLoopOptimization::TensorListInLoopOptimization() {
    auto loop_label = pattern::wrap_type<v5::Loop>();

    // pattern for condition sub-graph in Loop operarion
    auto num_iterations_label = pattern::wrap_type<op::v0::Parameter>();
    auto counter_label = pattern::wrap_type<op::v0::Parameter>();
    auto counter_step_label = pattern::wrap_type<op::v0::Constant>();
    auto updated_counter_label = pattern::wrap_type<op::v1::Add>({counter_label, counter_step_label});
    auto less_label = pattern::wrap_type<op::v1::Less>({updated_counter_label, num_iterations_label});
    auto condition_label = pattern::wrap_type<op::v0::Result>({less_label});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry rg;

        auto loop = ov::as_type_ptr<v5::Loop>(m.get_match_root());
        if (!loop) {
            return false;
        }

        // check that condition sub-graph of the required form:
        // counter with zero initial value and increments each iteration
        // loop continues until counter is less than given number
        ov::pass::pattern::Matcher condition_matcher(condition_label);

        const auto& body = loop->get_function();
        const auto& body_params = body->get_parameters();
        const auto& body_results = body->get_results();
        const auto& special_body_ports = loop->get_special_body_ports();
        const auto& input_descriptions = loop->get_input_descriptions();
        const auto& output_descriptions = loop->get_output_descriptions();

        if (!condition_matcher.match(body_results[special_body_ports.body_condition_output_idx]->output(0))) {
            return false;
        }

        const auto& condition_map = condition_matcher.get_pattern_value_map();
        int64_t counter_step = -1;
        if (!get_constant_value(condition_map.at(counter_step_label).get_node_shared_ptr(), counter_step) ||
            counter_step != 1) {
            return false;
        }

        // get initial value of counter
        int64_t initial_counter = 0;
        auto counter_param = ov::as_type_ptr<v0::Parameter>(condition_map.at(counter_label).get_node_shared_ptr());
        if (!counter_param) {
            return false;
        }

        for (const auto& input_desc : loop->get_input_descriptions()) {
            auto body_param_idx = input_desc->m_body_parameter_index;
            auto input_idx = input_desc->m_input_index;
            if (body_params[body_param_idx] != counter_param) {
                continue;
            }

            // it must be merged input and incremented each iteration
            auto merged_desc = ov::as_type_ptr<MergedD>(input_desc);
            if (!merged_desc) {
                return false;
            }
            auto result_idx = merged_desc->m_body_value_index;
            auto update_counter = body_results[result_idx]->input_value(0).get_node_shared_ptr();
            if (update_counter != condition_map.at(updated_counter_label).get_node_shared_ptr()) {
                return false;
            }

            // get initial value of counter
            if (!get_constant_value(loop->get_input_node_shared_ptr(input_idx), initial_counter)) {
                return false;
            }

            // suitable counter-parameter is found and checked
            break;
        }

        if (initial_counter != 0) {
            return false;
        }

        // collect vector of updated Parameter indices (they will be converted to SlicedInput parameters)
        // and Result nodes (they will be converted to ConcatOutput results)
        // also, some parameters and results can be removed and they are directly connected
        // to updated Parameter/Result nodes
        std::vector<uint64_t> remove_parameter_ids;
        std::vector<uint64_t> update_parameter_ids, update_result_ids;
        for (const auto& target_input : counter_param->get_output_target_inputs(0)) {
            update_parameter_to_slice_input(target_input.get_node()->shared_from_this(),
                                            body,
                                            input_descriptions,
                                            update_parameter_ids);
            update_result_to_concat_output(target_input.get_node()->shared_from_this(),
                                           body,
                                           body_results,
                                           input_descriptions,
                                           update_result_ids,
                                           remove_parameter_ids);
        }

        // avoid TensorListSetItem that overrides tensor list with data computed for each iteration
        // by index zero that is equivalent to Loop returning data only from last iteration
        // TensorListSetItem accepts constant index equal to zero
        // TensorListSetItem is not connected with counter node
        std::vector<uint64_t> update_result_last_iter_ids;
        for (uint64_t result_idx = 0; result_idx < body_results.size(); ++result_idx) {
            const auto& result = body_results[result_idx];
            auto tensor_list_set_item =
                std::dynamic_pointer_cast<TensorListSetItem>(result->get_input_node_shared_ptr(0));
            if (!tensor_list_set_item) {
                continue;
            }
            int64_t index_value = -1;
            if (!get_constant_value(tensor_list_set_item->get_input_node_shared_ptr(1), index_value) ||
                (index_value != 0)) {
                continue;
            }

            auto tensor_list = ov::as_type_ptr<v0::Parameter>(tensor_list_set_item->get_input_node_shared_ptr(0));
            if (!tensor_list) {
                continue;
            }
            int64_t param_idx = body->get_parameter_index(tensor_list);
            if (param_idx < 0) {
                continue;
            }

            update_result_last_iter_ids.push_back(result_idx);
            remove_parameter_ids.push_back(static_cast<uint64_t>(param_idx));
        }

        // nothing to update
        if (update_parameter_ids.size() == 0 && update_result_ids.size() == 0 &&
            update_result_last_iter_ids.size() == 0) {
            return false;
        }

        // build a new body_graph
        auto new_body_results = body_results;
        std::vector<uint64_t> all_update_result_ids = update_result_ids;
        all_update_result_ids.insert(all_update_result_ids.end(),
                                     update_result_last_iter_ids.begin(),
                                     update_result_last_iter_ids.end());
        for (auto update_result_idx : all_update_result_ids) {
            const auto& body_result = body_results[update_result_idx];
            auto tensor_list_set_item =
                std::dynamic_pointer_cast<TensorListSetItem>(body_result->get_input_node_shared_ptr(0));
            FRONT_END_GENERAL_CHECK(tensor_list_set_item,
                                    "[TensorFlow Frontend] internal error: tensor_list_set_item is nullptr in "
                                    "TensorListInLoopOptimization");
            // unsqueeze newly generated data at this iteration
            // that will be concatenated
            auto new_data = tensor_list_set_item->input_value(2);
            auto axis = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, 0);
            auto unsqueeze_new_data = std::make_shared<v0::Unsqueeze>(new_data, axis);
            auto new_body_result = std::make_shared<v0::Result>(unsqueeze_new_data);
            new_body_results[update_result_idx] = new_body_result;
        }
        auto new_body_params = ParameterVector{};
        for (uint64_t param_idx = 0; param_idx < static_cast<uint64_t>(body_params.size()); ++param_idx) {
            // skip Parameter nodes from remove_parameter_ids list
            if (std::find(remove_parameter_ids.begin(), remove_parameter_ids.end(), param_idx) !=
                remove_parameter_ids.end()) {
                continue;
            }

            // use updated Parameter node if needed
            if (std::find(update_parameter_ids.begin(), update_parameter_ids.end(), param_idx) !=
                update_parameter_ids.end()) {
                const auto& body_param = body_params[param_idx];
                FRONT_END_GENERAL_CHECK(body_param->get_output_target_inputs(0).size() == 1,
                                        "[TensorFlow Frontend] internal error: tensor list must have only consumer "
                                        "TensorListGetItem operation in TensorListInLoopOptimization");
                auto target_input = *(body_param->get_output_target_inputs(0).begin());
                auto tensor_list_get_item =
                    std::dynamic_pointer_cast<TensorListGetItem>(target_input.get_node()->shared_from_this());
                FRONT_END_GENERAL_CHECK(tensor_list_get_item,
                                        "[TensorFlow Frontend] internal error: tensor list must have only consumer "
                                        "TensorListGetItem operation in TensorListInLoopOptimization");

                auto new_shape = body_param->get_output_partial_shape(0);
                if (new_shape.rank().is_static() && new_shape.rank().get_length() > 0) {
                    // set a static dimension equal to 1 since it is sliced by axis 0
                    new_shape[0] = 1;
                }
                auto new_param = std::make_shared<v0::Parameter>(body_param->get_output_element_type(0), new_shape);
                new_param->set_friendly_name(body_param->get_friendly_name());

                // adjust new_param since it comes after slicing and sliced input needs to be squeezed
                auto squeeze_axis = std::make_shared<v0::Constant>(element::i32, Shape{1}, 0);
                auto squeeze_param = std::make_shared<v0::Squeeze>(new_param, squeeze_axis);

                // replace data producer for all consumers of TensorListGetItem
                tensor_list_get_item->output(0).replace(squeeze_param->output(0));
                new_body_params.push_back(new_param);
                continue;
            }

            new_body_params.push_back(body_params[param_idx]);
        }
        auto new_body_graph = std::make_shared<ov::Model>(new_body_results, new_body_params);

        // eventually, only some Parameter nodes can be removed
        // so indices of Parameters can be changed
        // a number of Result nodes and their indices leave unchanged
        // create new Loop operation and set input and output descriptions
        const auto& trip_count = loop->input_value(0);
        const auto& exec_cond = loop->input_value(1);
        auto new_loop = rg.make<v5::Loop>(trip_count, exec_cond);
        new_loop->set_special_body_ports(special_body_ports);
        new_loop->set_function(new_body_graph);

        // update current_iteration_input_idx since some Parameters can be removed
        // Result nodes are not removed so body_condition_output_idx leaves unchanged
        auto current_iteration_input_idx = special_body_ports.current_iteration_input_idx;
        if (current_iteration_input_idx > 0) {
            auto new_idx = get_new_param_idx(remove_parameter_ids, static_cast<uint64_t>(current_iteration_input_idx));
            auto new_special_body_ports = special_body_ports;
            new_special_body_ports.current_iteration_input_idx = static_cast<int64_t>(new_idx);
            new_loop->set_special_body_ports(new_special_body_ports);
        }

        // set inputs for new Loop operation
        for (const auto& input_desc : input_descriptions) {
            // skip already removed body Parameters
            auto param_idx = input_desc->m_body_parameter_index;
            auto input_index = input_desc->m_input_index;
            if (std::find(remove_parameter_ids.begin(), remove_parameter_ids.end(), param_idx) !=
                remove_parameter_ids.end()) {
                continue;
            }

            auto new_param_idx = get_new_param_idx(remove_parameter_ids, param_idx);
            const auto& new_body_param = new_body_params[new_param_idx];
            const auto& init_value = loop->input_value(input_index);
            if (std::find(update_parameter_ids.begin(), update_parameter_ids.end(), param_idx) !=
                update_parameter_ids.end()) {
                // set this input as sliced input
                new_loop->set_sliced_input(new_body_param, init_value, initial_counter, counter_step, 1, -1, 0);
            } else if (const auto& invariant_input = ov::as_type_ptr<InvariantD>(input_desc)) {
                new_loop->set_invariant_input(new_body_param, init_value);
            } else if (const auto& merged_input = ov::as_type_ptr<MergedD>(input_desc)) {
                const auto& body_res = new_body_results[merged_input->m_body_value_index];
                new_loop->set_merged_input(new_body_param, init_value, body_res);
            } else if (const auto& sliced_input = ov::as_type_ptr<SlicedD>(input_desc)) {
                new_loop->set_sliced_input(new_body_param,
                                           init_value,
                                           sliced_input->m_start,
                                           sliced_input->m_stride,
                                           sliced_input->m_part_size,
                                           sliced_input->m_end,
                                           sliced_input->m_axis);
            } else {
                // unknown type of input
                // transformation is not applicable
                return false;
            }
        }

        // set outputs for new Loop operation
        std::unordered_map<uint64_t, Output<Node>> idx_to_new_output;
        for (const auto& output_desc : output_descriptions) {
            auto result_idx = output_desc->m_body_value_index;
            auto output_index = output_desc->m_output_index;
            auto new_body_result = new_body_results[result_idx];

            if (std::find(update_result_ids.begin(), update_result_ids.end(), result_idx) != update_result_ids.end()) {
                idx_to_new_output[output_index] =
                    new_loop->get_concatenated_slices(new_body_result, initial_counter, counter_step, 1, -1, 0);
            } else if (std::find(update_result_last_iter_ids.begin(), update_result_last_iter_ids.end(), result_idx) !=
                       update_result_last_iter_ids.end()) {
                idx_to_new_output[output_index] = new_loop->get_iter_value(new_body_result, -1);
            } else if (const auto& concat_output = ov::as_type_ptr<ConcatD>(output_desc)) {
                idx_to_new_output[output_index] = new_loop->get_concatenated_slices(new_body_result,
                                                                                    concat_output->m_start,
                                                                                    concat_output->m_stride,
                                                                                    concat_output->m_part_size,
                                                                                    concat_output->m_end,
                                                                                    concat_output->m_axis);
            } else if (const auto& iter_output = ov::as_type_ptr<OutputD>(output_desc)) {
                idx_to_new_output[output_index] = new_loop->get_iter_value(new_body_result, iter_output->m_iteration);
            } else {
                // unknown type of output
                return false;
            }
        }

        auto loop_outputs = loop->outputs();
        for (size_t i = 0; i < loop_outputs.size(); ++i) {
            loop_outputs[i].replace(idx_to_new_output[i]);
        }
        copy_runtime_info(loop, rg.get());
        new_loop->set_friendly_name(loop->get_friendly_name());

        return true;
    };

    auto m =
        std::make_shared<pattern::Matcher>(loop_label, "ov::frontend::tensorflow::pass::TensorListInLoopOptimization");
    register_matcher(m, callback);
}
