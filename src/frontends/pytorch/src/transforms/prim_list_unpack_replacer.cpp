// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prim_list_unpack_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/frontend/pytorch/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

PrimListUnpackReplacer::PrimListUnpackReplacer() {
    auto list_unpack = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto list_unpack = cast_fw_node(m.get_match_root(), "prim::ListUnpack");
        if (!list_unpack)
            return false;

        auto input_node = list_unpack->input(0).get_source_output().get_node_shared_ptr();
        if (auto split_with_sizes = cast_fw_node(input_node, "aten::split_with_sizes")) {
            auto split = std::make_shared<opset8::VariadicSplit>(split_with_sizes->get_input_source_output(0),
                                                                 split_with_sizes->get_input_source_output(2),
                                                                 split_with_sizes->get_input_source_output(1));

            copy_runtime_info({list_unpack, input_node}, split);
            replace_node(list_unpack, split);

            return true;
        }

        if (auto chunk = cast_fw_node(input_node, "aten::chunk")) {
            // Using number of ListUnpack outputs instead of 1st input to chunk.
            // TODO: confirm it works for all cases
            auto split = std::make_shared<opset8::Split>(chunk->get_input_source_output(0),
                                                         chunk->get_input_source_output(2),
                                                         list_unpack->get_output_size());

            copy_runtime_info({list_unpack, input_node}, split);
            replace_node(list_unpack, split);

            return true;
        }

        if (auto shape_of = std::dynamic_pointer_cast<opset8::ShapeOf>(input_node)) {
            // case aten::size as input
            // Number of ListUnpack outputs should be equal to rank of input shape.
            auto axis_0 = opset8::Constant::create(element::i64, Shape{}, {0});
            auto split = std::make_shared<opset8::Split>(shape_of, axis_0, list_unpack->get_output_size());

            NodeVector to_copy_rt{axis_0, split};
            OutputVector res;
            for (auto output : split->outputs()) {
                auto squeeze = std::make_shared<opset8::Squeeze>(output, axis_0);
                to_copy_rt.push_back(squeeze);
                res.push_back(squeeze);
            }

            copy_runtime_info({list_unpack, input_node}, to_copy_rt);
            replace_node(list_unpack, res);

            return true;
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(list_unpack,
                                                          "ov::frontend::pytorch::pass::PrimListUnpackReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov