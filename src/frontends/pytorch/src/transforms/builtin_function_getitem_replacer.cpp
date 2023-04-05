// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builtin_function_getitem_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

BuiltinFunctionGetItemReplacer::BuiltinFunctionGetItemReplacer() {
    auto getitem = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {

        auto getitem = cast_fw_node(m.get_match_root(), "<built-in function getitem>");
        if (!getitem)
            return false;

        int64_t axis;
        auto axis_node = getitem->get_input_node_shared_ptr(1);
        auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(axis_node);
        if (!axis_const)
            return false;
        auto _axis = axis_const->cast_vector<int64_t>();
        if (_axis.size() != 1)
            return false;
        axis = _axis[0];

        const auto&& tmp_inputs = get_list_as_outputs(getitem->get_input_source_output(0));

        auto getitem_idx = ov::op::v0::Constant::create(element::i32, Shape{1}, {0});
        auto zero = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
        auto result = std::make_shared<ov::op::v8::Gather>(tmp_inputs[axis], getitem_idx, zero);
        copy_runtime_info(getitem, result);
        replace_node(getitem, result);
        result->set_friendly_name(getitem->get_friendly_name());

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(getitem, "ov::frontend::pytorch::pass::BuiltinFunctionGetItemReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
