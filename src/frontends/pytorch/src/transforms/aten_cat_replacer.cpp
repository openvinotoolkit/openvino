// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_cat_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
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

using namespace ov::op;

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
        auto cat = cast_fw_node(m.get_match_root(), {"aten::cat", "aten::concat", "quantized::cat"});
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
            // case when concatenation is done inside the Loop
            auto body = loop->get_function();
            auto output_index = cat->input(0).get_source_output().get_index();
            int64_t body_result_index = -1;
            for (auto& out_desc : loop->get_output_descriptions()) {
                if (out_desc->m_output_index == output_index) {
                    body_result_index = static_cast<int64_t>(out_desc->m_body_value_index);
                    break;
                }
            }
            FRONT_END_GENERAL_CHECK(body_result_index >= 0, "Couldn't find descriptor for output.");
            auto body_result = body->get_results()[body_result_index];
            auto append = cast_fw_node(body_result->get_input_node_shared_ptr(0), "aten::append");
            if (!append) {
                add_exception_to_fw_node(
                    cat,
                    "<aten/quantized>::cat unsupported case: aten::append wasn't found inside prim::Loop body.");
                return false;
            }
            auto param = ov::as_type_ptr<v0::Parameter>(append->get_input_node_shared_ptr(0));
            if (!param) {
                add_exception_to_fw_node(
                    cat,
                    "<aten/quantized>::cat unsupported case: input of aten::append inside prim::Loop "
                    "body is not a body input.");
                return false;
            }
            auto body_param_index = body->get_parameter_index(param);
            FRONT_END_GENERAL_CHECK(body_param_index >= 0, "Couldn't find parameter in body parameters.");
            int64_t input_index = -1;
            for (auto& in_desc : loop->get_input_descriptions()) {
                if (in_desc->m_body_parameter_index == static_cast<size_t>(body_param_index)) {
                    input_index = static_cast<int64_t>(in_desc->m_input_index);
                    break;
                }
            }
            FRONT_END_GENERAL_CHECK(input_index >= 0, "Couldn't find descriptor for input.");
            auto list_construct = cast_fw_node(loop->get_input_node_shared_ptr(input_index), "prim::ListConstruct");
            if (!list_construct || list_construct->get_input_size() > 0) {
                add_exception_to_fw_node(
                    cat,
                    "<aten/quantized>::cat unsupported case: aten::append input outside of prim::Loop "
                    "body is not a prim::ListConstruct.");
                return false;
            }
            auto new_result = std::make_shared<v0::Result>(append->input_value(1));
            body->add_results({new_result});
            auto new_output = loop->get_concatenated_slices(new_result, 0, 1, 1, -1, axis);
            copy_runtime_info(cat, loop);
            cat->output(0).replace(new_output);
            return true;
        }

        const auto&& tmp_inputs = get_list_as_outputs(cat->get_input_source_output(0));
        auto result = std::make_shared<v0::Concat>(OutputVector(tmp_inputs.begin(), tmp_inputs.end()), axis);
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
