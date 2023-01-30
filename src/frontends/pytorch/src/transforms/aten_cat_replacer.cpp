// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_cat_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
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

        auto axis_node = cat->input(1).get_source_output().get_node_shared_ptr();
        auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(axis_node);
        if (!axis_const)
            return false;
        auto axis = axis_const->cast_vector<int64_t>();
        if (axis.size() != 1)
            return false;

        OutputVector tmp_inputs;
        NodeVector rt_copy_from{cat};
        std::shared_ptr<Node> input_node = cat->input(0).get_source_output().get_node_shared_ptr();
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
        auto result = std::make_shared<ov::op::v0::Concat>(inputs, axis[0]);
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