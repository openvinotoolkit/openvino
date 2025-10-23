// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_cat_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

namespace {
bool process_loop_case(std::shared_ptr<Node> cat, std::shared_ptr<v5::Loop> loop, int64_t axis, bool is_stack) {
    // Concatenation of the list happens inside Loop operation
    // prim::ListConstruct -> Loop [Parameter -> prim::append -> Result] -> aten::cat
    // We recreate the Loop to remove the original merged output and create new output using get_concatenated_slices
    // Also we remove original input which was prim::ListConstruct
    auto body = loop->get_function();
    auto output_index = cat->input(0).get_source_output().get_index();

    int64_t body_result_index = -1;
    for (const auto& desc : loop->get_output_descriptions()) {
        if (desc->m_output_index == output_index) {
            body_result_index = static_cast<int64_t>(desc->m_body_value_index);
            break;
        }
    }
    FRONT_END_GENERAL_CHECK(body_result_index >= 0, "Couldn't find descriptor for output.");

    auto body_result = body->get_results()[body_result_index];
    auto append = cast_fw_node(body_result->get_input_node_shared_ptr(0), "aten::append");
    if (!append) {
        add_exception_to_fw_node(cat, "<aten/quantized>::cat unsupported case: aten::append not found in Loop body.");
        return false;
    }

    auto param = ov::as_type_ptr<v0::Parameter>(append->get_input_node_shared_ptr(0));
    if (!param) {
        add_exception_to_fw_node(cat,
                                 "<aten/quantized>::cat unsupported case: aten::append input is not a body parameter.");
        return false;
    }

    auto body_param_index = body->get_parameter_index(param);
    FRONT_END_GENERAL_CHECK(body_param_index >= 0, "Couldn't find parameter in body parameters.");

    int64_t input_index = -1;
    for (const auto& desc : loop->get_input_descriptions()) {
        if (desc->m_body_parameter_index == static_cast<size_t>(body_param_index)) {
            input_index = static_cast<int64_t>(desc->m_input_index);
            break;
        }
    }
    FRONT_END_GENERAL_CHECK(input_index >= 0, "Couldn't find descriptor for input.");

    auto list_construct = cast_fw_node(loop->get_input_node_shared_ptr(input_index), "prim::ListConstruct");
    if (!list_construct || list_construct->get_input_size() > 0) {
        add_exception_to_fw_node(cat, "<aten/quantized>::cat unsupported case: invalid ListConstruct.");
        return false;
    }

    // Filter output descriptions
    std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>> new_output_descs;
    for (const auto& desc : loop->get_output_descriptions()) {
        if (desc->m_body_value_index != static_cast<size_t>(body_result_index)) {
            new_output_descs.push_back(desc);
        }
    }

    // Filter input descriptions
    std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::InputDescription>> new_input_descs;
    int64_t input_idx = -1;
    for (const auto& desc : loop->get_input_descriptions()) {
        if (desc->m_body_parameter_index != static_cast<size_t>(body_param_index)) {
            new_input_descs.push_back(desc);
        } else {
            input_idx = static_cast<int64_t>(desc->m_input_index);
        }
    }
    FRONT_END_GENERAL_CHECK(input_idx >= 0, "Input description for body parameter not found.");

    // Update inputs
    OutputVector new_inputs;
    for (size_t i = 0; i < loop->get_input_size(); ++i) {
        if (static_cast<int64_t>(i) != input_idx) {
            new_inputs.push_back(loop->input_value(i));
        }
    }

    // Update body
    body->remove_result(body_result);
    for (auto& desc : new_output_descs) {
        if (desc->m_body_value_index > static_cast<size_t>(body_result_index))
            desc->m_body_value_index--;
        if (desc->m_output_index > output_index)
            desc->m_output_index--;
    }
    body->remove_parameter(param);
    for (auto& desc : new_input_descs) {
        if (desc->m_body_parameter_index > static_cast<size_t>(body_param_index))
            desc->m_body_parameter_index--;
        if (desc->m_input_index > static_cast<size_t>(input_index))
            desc->m_input_index--;
    }

    // Create new loop
    auto new_loop = std::make_shared<v5::Loop>();
    new_loop->set_special_body_ports(loop->get_special_body_ports());
    new_loop->set_arguments(new_inputs);
    new_loop->set_friendly_name(loop->get_friendly_name());
    new_loop->set_function(loop->get_function());
    new_loop->set_input_descriptions(0, new_input_descs);
    new_loop->set_output_descriptions(0, new_output_descs);
    new_loop->set_output_size(loop->get_output_size() - 1);

    // Update output mappings
    for (size_t i = 0; i < loop->get_output_size() - 1; ++i) {
        if (i < output_index) {
            replace_output_update_name(loop->output(i), new_loop->output(i));
        } else if (i > output_index) {
            replace_output_update_name(loop->output(i + 1), new_loop->output(i));
        }
    }

    // Handle stack case and create new result
    auto to_append = append->input_value(1);
    if (is_stack) {
        auto axis_constant = v0::Constant::create(element::i32, Shape{}, {axis});
        to_append = std::make_shared<v0::Unsqueeze>(to_append, axis_constant);
    }
    auto new_result = std::make_shared<v0::Result>(to_append);
    body->add_results({new_result});
    auto new_output = new_loop->get_concatenated_slices(new_result, 0, 1, 1, -1, axis);
    copy_runtime_info({loop, cat}, new_loop);
    cat->output(0).replace(new_output);

    return true;
}
}  // namespace

// aten::cat needs a special handling since it takes a Tensor[] as input. We set the inputs of ListConstruct as the
// inputs of cat.
//
// Pytorch IR:                                   OV model:
//     %a    %b     %c          %dim              %a    %b    %c
//      \     |     /             |                \     |    /
//   prim::ListConstruct   prim::Constant        Concat[axis=%dim]
//                    \      /
//                    aten::cat
AtenCatToConcat::AtenCatToConcat() {
    auto aten_cat = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>(
        fw_node_predicate({"aten::cat", "aten::concat", "aten::concatenate", "quantized::cat", "aten::stack"}));

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        const auto root = m.get_match_root();
        bool is_stack = cast_fw_node(root, "aten::stack") != nullptr;
        auto cat = ov::as_type_ptr<ov::op::util::FrameworkNode>(root);
        if (!cat)
            return false;

        int64_t axis;
        if (cat->get_input_size() > 1) {
            auto axis_const = ov::util::get_constant_from_source(cat->input_value(1));
            if (!axis_const) {
                add_exception_to_fw_node(cat, "<aten/quantized>::cat unsupported case: axis is not a constant.");
                return false;
            }
            auto _axis = axis_const->cast_vector<int64_t>();
            if (_axis.size() != 1) {
                add_exception_to_fw_node(cat, "<aten/quantized>::cat unsupported case: axis is not a scalar.");
                return false;
            }
            axis = _axis[0];
        } else {
            const auto& attrs = cat->get_attrs();
            if (attrs.find("axis") == attrs.end()) {
                add_exception_to_fw_node(cat, "<aten/quantized>::cat unsupported case: axis not found in attributes.");
                return false;
            }
            axis = std::stoll(attrs.at("axis"));
        }

        std::shared_ptr<Node> input_node = cat->get_input_node_shared_ptr(0);
        if (auto loop = ov::as_type_ptr<v5::Loop>(input_node)) {
            return process_loop_case(cat, loop, axis, is_stack);
        }

        if (is_stack) {
            // Special case for GPTQ pattern
            auto list_construct = cast_fw_node(input_node, "prim::ListConstruct");
            if (const auto& compression = u4_compression_stack(input_node->input_values(), axis)) {
                copy_runtime_info_and_name(cat, {compression}, {input_node});
                replace_node(cat, compression);
                return true;
            }
        }

        const auto&& tmp_inputs = get_list_as_outputs(cat->get_input_source_output(0));
        OutputVector concat_inputs;
        if (is_stack) {
            auto axis_to_unsqueeze = v0::Constant::create(element::i32, Shape{}, {axis});
            for (const auto& list_input : tmp_inputs) {
                auto unsqueezed_node = std::make_shared<v0::Unsqueeze>(list_input, axis_to_unsqueeze);
                concat_inputs.push_back(unsqueezed_node);
            }
        } else {
            concat_inputs = OutputVector(tmp_inputs.begin(), tmp_inputs.end());
        }
        auto result = std::make_shared<v0::Concat>(concat_inputs, axis);
        copy_runtime_info_and_name(cat, {result});
        replace_node(cat, result);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(aten_cat, "ov::frontend::pytorch::pass::AtenCatToConcat");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
