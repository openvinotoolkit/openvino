// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dict_resolver.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::pass;
using namespace ov::op;

bool DictParameterResolver::run_on_model(const std::shared_ptr<Model>& model) {
    bool changed = false;
    const auto parameters = model->get_parameters();
    ParameterVector new_params;

    for (const auto& p : parameters) {
        bool at_least_one_unused = false;
        if (p->get_output_size() == 1) {
            const auto targets = p->get_output_target_inputs(0);
            for (const auto inp : targets) {
                const auto getitem_node = cast_fw_node(inp.get_node()->shared_from_this(), "aten::__getitem__");
                if (getitem_node) {
                    const auto index_node =
                        ov::as_type_ptr<ov::op::util::FrameworkNode>(getitem_node->get_input_node_shared_ptr(1));
                    if (!index_node) {
                        at_least_one_unused = true;
                        continue;
                    }
                    const auto& attrs = index_node->get_attrs();
                    if (attrs.find("string_value") == attrs.end()) {
                        // index node must contain string value
                        at_least_one_unused = true;
                        continue;
                    }
                    const auto& name = attrs.at("string_value");
                    auto new_param = std::make_shared<v0::Parameter>(getitem_node->get_output_element_type(0),
                                                                     getitem_node->get_output_partial_shape(0));
                    new_param->set_friendly_name(name);
                    getitem_node->output(0).replace(new_param);
                    new_param->output(0).set_names({name});
                    new_params.push_back(new_param);
                    changed = true;
                } else {
                    at_least_one_unused = true;
                }
            }
        }
        if (changed) {
            model->remove_parameter(p);
            if (at_least_one_unused || p->get_output_size() != 1) {
                new_params.push_back(p);
            }
        }
    }
    if (changed) {
        model->add_parameters(new_params);
    }
    return changed;
};

bool DictResultResolver::run_on_model(const std::shared_ptr<Model>& model) {
    bool changed = false;
    const auto results = model->get_results();
    for (const auto& res : results) {
        if (auto dict_construct_node = cast_fw_node(res->get_input_node_shared_ptr(0), "prim::DictConstruct")) {
            const auto inputs = dict_construct_node->input_values();
            if (inputs.size() % 2) {
                // inputs must be divisible by 2
                add_exception_to_fw_node(dict_construct_node,
                                         "prim::DictConstruct: inputs number is not divisible by 2.");
                return false;
            }
            ResultVector new_outputs;
            for (size_t i = 0; i < inputs.size(); i += 2) {
                auto new_output = inputs.at(i + 1);
                const auto& name_node = inputs.at(i);
                auto fw_node = ov::as_type_ptr<ov::op::util::FrameworkNode>(name_node.get_node_shared_ptr());
                if (!fw_node) {
                    add_exception_to_fw_node(
                        dict_construct_node,
                        "prim::DictConstruct: odd inputs must contain constant strings encoded as fw nodes.");
                    return false;
                }
                const auto& attrs = fw_node->get_attrs();
                if (attrs.find("string_value") == attrs.end()) {
                    // fw node must contain string value
                    add_exception_to_fw_node(dict_construct_node, "prim::DictConstruct: unexpected dict key format.");
                    return false;
                }
                const auto& name = attrs.at("string_value");
                new_output.set_names({name});
                new_outputs.push_back(std::make_shared<v0::Result>(new_output));
            }
            bool after_res = false;
            for (const auto& result : results) {
                if (after_res) {
                    // To preserve output order remove results after this one and insert them after new outputs
                    model->remove_result(result);
                    new_outputs.push_back(result);
                }
                if (result == res) {
                    after_res = true;
                    model->remove_result(result);
                }
            }
            model->add_results(new_outputs);
            changed = true;
        }
    }
    return changed;
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
