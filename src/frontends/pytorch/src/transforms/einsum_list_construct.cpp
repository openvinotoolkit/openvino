// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "einsum_list_construct.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/einsum.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

using namespace ov::pass::pattern;

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::pass;
using namespace ov::op;

AtenEinsumListConstructReplacer::AtenEinsumListConstructReplacer() {
    auto einsum_op = pattern::wrap_type<ov::op::util::FrameworkNode>();
    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto einsum_op = cast_fw_node(m.get_match_root(), "aten::einsum");
        if (!einsum_op) {
            return false;
        }
        const auto& equation_input = einsum_op->input_value(0).get_node_shared_ptr();
        const auto& tensor_list = einsum_op->input_value(1).get_node_shared_ptr();
        std::string equation;
        if (const auto& fw_node_mode = cast_fw_node(equation_input, "prim::Constant")) {
            const auto& attrs = fw_node_mode->get_attrs();
            if (attrs.find("string_value") != attrs.end()) {
                equation = attrs.at("string_value");
            }
        } else {
            add_exception_to_fw_node(einsum_op, "aten::einsum: equation should be string constant.");
            return false;
        }
        // Check if ListConstruct is an input
        if (auto list_construct_node = cast_fw_node(tensor_list, "prim::ListConstruct")) {
            const auto& list_inputs = list_construct_node->input_values();
            OutputVector node_vector;
            // Iterate over values in ListConstruct
            for (const auto& list_input : list_inputs) {
                node_vector.push_back(list_input);
            }

            auto einsum = std::make_shared<v7::Einsum>(node_vector, equation);
            copy_runtime_info_and_name(einsum_op, {einsum}, {equation_input, tensor_list});
            replace_node(einsum_op, einsum);
            return true;
        }
        add_exception_to_fw_node(einsum_op, "aten::einsum: unsupported case.");
        return false;
    };

    auto m =
        std::make_shared<pattern::Matcher>(einsum_op, "ov::frontend::pytorch::pass::AtenEinsumListConstructReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov