// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "prim_unpack_parameter_replacer.hpp"

#include <deque>
#include <sstream>

#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

bool DecomposeUnpackParameters::run_on_model(const std::shared_ptr<Model>& model) {
    bool at_least_one_decomposed = false;
    const auto& orig_parameters = model->get_parameters();
    std::deque<std::shared_ptr<ov::op::v0::Parameter>> parameters(orig_parameters.begin(), orig_parameters.end());
    ov::ParameterVector updated_parameters;  // will hold final fully unpacked parameters list

    while (!parameters.empty()) {
        auto parameter = parameters.front();
        parameters.pop_front();
        auto consumers = parameter->get_output_target_inputs(0);
        size_t num_outputs = 0;  // number of outputs in each unpack consumer should match
        bool all_unpacks = true;

        // collects all outputs per each consumer operation for this tuple/list Parameter
        std::vector<OutputVector> consumer_outputs;

        // The following vector track consumer nodes having prim::TupleUnpack type to form a detailed
        // error message in case when parameter replacement is required but not possible.
        std::vector<std::shared_ptr<Node>> consumer_unpacks;

        for (const auto& consumer : consumers) {
            auto node = consumer.get_node()->shared_from_this();
            std::shared_ptr<ov::op::util::FrameworkNode> unpack = cast_fw_node(node, "prim::TupleUnpack");
            if (!unpack) {
                unpack = cast_fw_node(node, "prim::ListUnpack");
            }
            if (!unpack) {
                all_unpacks = false;
                continue;  // need to look at all consumers to form good diagnostics
            }
            consumer_unpacks.push_back(node);
            if (num_outputs == 0) {
                num_outputs = node->get_output_size();
            } else if (num_outputs != node->get_output_size()) {
                std::stringstream message;
                message << "Unpack node " << node
                        << " as one of the consumers of a tuple/list, which is introduced by parameter "
                        << parameter->output(0) << ", has number of outputs " << node->get_output_size()
                        << " not matching number of outputs " << num_outputs << " for other consumer(s) found earlier.";
                add_exception_to_fw_node(node, message.str());
                all_unpacks = false;
                break;
            }
            consumer_outputs.push_back(node->outputs());
        }

        if (!all_unpacks || consumer_outputs.empty()) {
            // if at least one consumer is not an unpack-like op or there are not matching number of unpacked objects,
            // we cannot replace other unpacks even if they exist, leaving Unpack-op(s) in the graph for this Parameter

            updated_parameters.push_back(parameter);
            // In case if at least one Unpack exists there is an opportunity to attach diagnostics
            for (const auto& consumer : consumer_unpacks) {
                std::stringstream message;
                message << "Not unpack operations exist except this one: " << consumer
                        << " found as one of the consumers of a tuple/list, which is introduced by parameter "
                        << parameter->output(0) << ".";
                add_exception_to_fw_node(consumer, message.str());
            }
            continue;
        }

        // enumerating outputs in reverse order because of parameters.push_front below
        for (size_t i = num_outputs; i--;) {
            // Merged partial shape and element type among all the consumers of i-th result of unpack ops
            PartialShape ps = PartialShape::dynamic();
            element::Type et = element::dynamic;
            std::set<Input<Node>> inputs;

            for (const auto& outputs : consumer_outputs) {
                const auto& output = outputs[i];
                OPENVINO_ASSERT(PartialShape::merge_into(ps, output.get_partial_shape()),
                                "Consumers for unpack op have incompatible shape");
                OPENVINO_ASSERT(element::Type::merge(et, et, output.get_element_type()),
                                "Consumers for unpack op have incompatible types");
                auto target_inputs = output.get_target_inputs();
                inputs.insert(target_inputs.begin(), target_inputs.end());
            }

            auto new_parameter = std::make_shared<ov::op::v0::Parameter>(et, ps);

            for (auto& input : inputs) {
                const auto& names = input.get_tensor().get_names();
                input.replace_source_output(new_parameter->output(0));
                new_parameter->output(0).add_names(names);
            }

            // TODO: Assign correct names
            parameters.push_front(new_parameter);
            at_least_one_decomposed = true;
        }
    }

    if (at_least_one_decomposed) {
        // remove all parameters
        while (!model->get_parameters().empty())
            model->remove_parameter(model->get_parameters()[0]);
        // and replace them by updated list of parameters
        model->add_parameters(updated_parameters);
    }

    return at_least_one_decomposed;
};
}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
