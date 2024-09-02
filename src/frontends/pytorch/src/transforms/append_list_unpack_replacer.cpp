// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "append_list_unpack_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

AppendListUnpackReplacer::AppendListUnpackReplacer() {
    auto list_unpack = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto list_unpack = cast_fw_node(m.get_match_root(), "prim::ListUnpack");
        if (!list_unpack)
            return false;

        OutputVector tmp_inputs;
        NodeVector rt_copy_from;
        auto input_node = list_unpack->input_value(0).get_node_shared_ptr();

        // Optional aten::__getitem__ node.
        auto getitem_node = cast_fw_node(input_node, "aten::__getitem__");
        if (getitem_node) {
            rt_copy_from.push_back(getitem_node);
            input_node = getitem_node->input(0).get_source_output().get_node_shared_ptr();
        }

        while (auto append_node = cast_fw_node(input_node, "aten::append")) {
            rt_copy_from.push_back(append_node);
            tmp_inputs.emplace_back(append_node->input(1).get_source_output());
            input_node = append_node->input(0).get_source_output().get_node_shared_ptr();
        }
        OutputVector inputs;
        auto list_construct_node = cast_fw_node(std::move(input_node), "prim::ListConstruct");
        if (!list_construct_node) {
            return false;
        }
        inputs.reserve(list_construct_node->inputs().size() + tmp_inputs.size());
        rt_copy_from.push_back(list_construct_node);
        for (auto& input : list_construct_node->inputs()) {
            inputs.push_back(input.get_source_output());
        }

        inputs.insert(inputs.end(),
                      std::make_move_iterator(tmp_inputs.rbegin()),
                      std::make_move_iterator(tmp_inputs.rend()));
        if (getitem_node) {
            // If aten::__getitem__, expect inputs to be equivalent of pytorch Tensor[][].
            // Tensor selected by aten::__getitem__ index needs to be splitted in axis 0.
            auto getitem_index_const = ov::util::get_constant_from_source(getitem_node->input_value(1));
            if (!getitem_index_const)
                return false;
            auto index_val = getitem_index_const->cast_vector<int64_t>();
            if (index_val.size() != 1) {
                add_exception_to_fw_node(list_unpack, "prim::ListUnpack: index of aten::__getitem__ is not scalar.");
                return false;
            }
            auto index = index_val[0];
            if (index_val[0] < 0) {
                index = inputs.size() + index;
            }
            auto axis_0 = v0::Constant::create(element::i32, Shape{}, {0});
            auto split = std::make_shared<v1::Split>(inputs[index], axis_0, list_unpack->get_output_size());
            NodeVector to_copy_rt{axis_0, split};
            OutputVector res;
            for (auto& output : split->outputs()) {
                auto squeeze = std::make_shared<v0::Squeeze>(output, axis_0);
                to_copy_rt.push_back(squeeze);
                res.push_back(squeeze);
            }
            copy_runtime_info_and_name(list_unpack, std::move(to_copy_rt), rt_copy_from);
            replace_node(list_unpack, res);
            return true;
        } else {
            // Without aten::__getitem__, expect inputs to be equivalent od pytorch Tensor[].
            // Return all inputs.
            replace_node(list_unpack, inputs);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(list_unpack,
                                                          "ov::frontend::pytorch::pass::AppendListUnpackReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
