// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "prim_list_tuple_unpack_replacer.hpp"

#include <queue>

#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {


bool DecomposeListTupleParameters::run_on_model(const std::shared_ptr<Model>& model) {
    bool at_least_one_decomposed = false;
    std::queue<std::shared_ptr<ov::op::v0::Parameter>> parameters;
    for (auto par : model->get_parameters()) {
        parameters.push(par);
    }
    while (!parameters.empty()) {
        auto parameter = parameters.front();
        parameters.pop();
        auto consumers = parameter->get_output_target_inputs(0);
        size_t num_outputs = 0; // number of outputs in each unpack consumer should match
        bool all_unpacks = true;

        // collects all outputs per each consumer operation for this tuple/list Parameter
        std::vector<OutputVector> consumer_outputs;

        for(auto consumer: consumers) {
            auto node = consumer.get_node()->shared_from_this();
            auto tuple_unpack = cast_fw_node(node, "prim::TupleUnpack");
            auto list_unpack = cast_fw_node(node, "prim::ListUnpack");
            if (!tuple_unpack && !list_unpack) {
                all_unpacks = false;
                break;
            }
            if(num_outputs == 0) {
                num_outputs = node->get_output_size();
            } else if (num_outputs != node->get_output_size()) {
                std::cerr
                    << "[ PT FE WARNING ] Unpack node " << node
                    << " as one of the consumers of tuple/list object has number of outputs "
                    << node->get_output_size() << " not matching number of outputs " << num_outputs << " for other consumer.\n";
                all_unpacks = false;
                break;
            }
            consumer_outputs.push_back(node->outputs());
        }

        if (!all_unpacks || consumer_outputs.empty()) {
            // if at least one consumer is not an unpack-like op or there are not matching number of unpacked objects,
            // we cannot replace other unpacks even if they exist, leaving Unpack-op(s) in the graph for this Parameter
            continue;
        }

        for (size_t i = 0; i < num_outputs; ++i) {
            // Merged partial shape and element type among all the consumers of i-th result of unpack ops
            PartialShape ps = PartialShape::dynamic();
            element::Type et = element::dynamic;
            std::set<Input<Node>> inputs;

            for (auto outputs: consumer_outputs) {
                auto output = outputs[i];
                OPENVINO_ASSERT(
                    PartialShape::merge_into(ps, output.get_partial_shape()),
                    "Consumers for unpack op have incompatible shape");
                OPENVINO_ASSERT(
                    element::Type::merge(et, et, output.get_element_type()),
                    "Consumers for unpack op have incompatible types");
                auto target_inputs = output.get_target_inputs();
                inputs.insert(target_inputs.begin(), target_inputs.end());
            }

            auto new_parameter = std::make_shared<ov::op::v0::Parameter>(et, ps);

            for(auto input: inputs) {
                auto names = input.get_tensor().get_names();
                input.replace_source_output(new_parameter->output(0));
                new_parameter->output(0).add_names(names);
            }

            // TODO: Assign correct names
            model->add_parameters({new_parameter});
            parameters.push(new_parameter);
            model->remove_parameter(parameter);
            at_least_one_decomposed = true;
        }
    }

    return at_least_one_decomposed;
};
}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
