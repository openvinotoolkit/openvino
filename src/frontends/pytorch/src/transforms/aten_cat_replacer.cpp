// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_cat_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

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
    auto aten_cat = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto cat = cast_fw_node(m.get_match_root(), "aten::cat");
        if (!cat)
            return false;

        int64_t axis;
        if (cat->get_input_size() > 1) {
            auto axis_node = cat->get_input_node_shared_ptr(1);
            auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(axis_node);
            if (!axis_const)
                return false;
            auto _axis = axis_const->cast_vector<int64_t>();
            if (_axis.size() != 1)
                return false;
            axis = _axis[0];
        } else {
            const auto& attrs = cat->get_attrs();
            if (attrs.find("axis") == attrs.end())
                return false;
            axis = std::stoll(attrs.at("axis"));
        }

        std::shared_ptr<Node> input_node = cat->get_input_node_shared_ptr(0);
        if (auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(input_node)) {
            // case when concatenation is done inside the Loop
            auto body = loop->get_function();
            auto output_index = cat->input(0).get_source_output().get_index();
            int64_t body_result_index = -1;
            for (auto out_desc : loop->get_output_descriptions()) {
                if (out_desc->m_output_index == output_index) {
                    body_result_index = static_cast<int64_t>(out_desc->m_body_value_index);
                    break;
                }
            }
            FRONT_END_GENERAL_CHECK(body_result_index >= 0, "Couldn't find descriptor for output.");
            auto body_result = body->get_results()[body_result_index];
            auto append = cast_fw_node(body_result->get_input_node_shared_ptr(0), "aten::append");
            if (!append)
                return false;
            auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(append->get_input_node_shared_ptr(0));
            if (!param)
                return false;
            auto body_param_index = body->get_parameter_index(param);
            FRONT_END_GENERAL_CHECK(body_param_index >= 0, "Couldn't find parameter in body parameters.");
            int64_t input_index = -1;
            for (auto in_desc : loop->get_input_descriptions()) {
                if (in_desc->m_body_parameter_index == static_cast<size_t>(body_param_index)) {
                    input_index = static_cast<int64_t>(in_desc->m_input_index);
                    break;
                }
            }
            FRONT_END_GENERAL_CHECK(input_index >= 0, "Couldn't find descriptor for input.");
            auto list_construct = cast_fw_node(loop->get_input_node_shared_ptr(input_index), "prim::ListConstruct");
            if (!list_construct || list_construct->get_input_size() > 0)
                return false;
            // TODO: Is unsqueeze needed?
            auto new_result = std::make_shared<ov::op::v0::Result>(append->input_value(1));
            body->add_results({new_result});
            auto new_output = loop->get_concatenated_slices(new_result, 0, 1, 1, -1, axis);
            copy_runtime_info(cat, loop);
            cat->output(0).replace(new_output);
            return true;
        }

        OutputVector tmp_inputs;
        NodeVector rt_copy_from{cat};
        while (const auto& input_fw_node = cast_fw_node(input_node, "aten::append")) {
            rt_copy_from.push_back(input_fw_node);
            tmp_inputs.push_back(input_fw_node->input(1).get_source_output());
            input_node = input_fw_node->input(0).get_source_output().get_node_shared_ptr();
        }
        auto list_construct = cast_fw_node(input_node, "prim::ListConstruct");
        if (!list_construct)
            return false;
        rt_copy_from.push_back(list_construct);
        OutputVector inputs;
        for (auto& input : list_construct->inputs()) {
            inputs.push_back(input.get_source_output());
        }
        inputs.insert(inputs.end(), tmp_inputs.rbegin(), tmp_inputs.rend());
        auto result = std::make_shared<ov::op::v0::Concat>(inputs, axis);
        copy_runtime_info(rt_copy_from, result);
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