// Copyright (C) 2018-2023 Intel Corporation
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

        auto input_node = list_unpack->input_value(0).get_node_shared_ptr();
        if (auto torch_split = cast_fw_node(input_node, "aten::split")) {
            auto rank = torch_split->input(1).get_partial_shape().rank();
            if (rank.is_dynamic()) {
                return false;
            }
            if (rank.get_length() == 0) {
                // Create split_lenghts tensor from split_size int,
                // allow for last chunk to be smaller if data is not equally divisible.
                auto split_size = torch_split->get_input_source_output(1);
                // Using number of ListUnpack outputs.
                auto num_out_m_1 = opset8::Constant::create(split_size.get_element_type(),
                                                            Shape{1},
                                                            {list_unpack->get_output_size() - 1});
                auto const_neg_1 = opset8::Constant::create(split_size.get_element_type(), Shape{1}, {-1});
                auto split_lenghts_m_1 = std::make_shared<opset8::Tile>(split_size, num_out_m_1);
                NodeVector concat_inputs{split_lenghts_m_1, const_neg_1};
                auto split_lenghts = std::make_shared<opset8::Concat>(concat_inputs, 0);
                auto split = std::make_shared<opset8::VariadicSplit>(torch_split->get_input_source_output(0),
                                                                     torch_split->get_input_source_output(2),
                                                                     split_lenghts);
                copy_runtime_info({list_unpack, input_node}, split);
                replace_node(list_unpack, split);
            } else {
                auto split = std::make_shared<opset8::VariadicSplit>(torch_split->get_input_source_output(0),
                                                                     torch_split->get_input_source_output(2),
                                                                     torch_split->get_input_source_output(1));
                copy_runtime_info({list_unpack, input_node}, split);
                replace_node(list_unpack, split);
            }

            return true;
        }

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

        if (auto unbind = cast_fw_node(input_node, "aten::unbind")) {
            const auto input = unbind->get_input_source_output(0);
            const auto axis = unbind->get_input_source_output(1);
            const auto num_splits = list_unpack->get_output_size();
            auto split = std::make_shared<opset8::Split>(input, axis, num_splits);
            NodeVector to_copy_rt{split};
            OutputVector outputs;
            for (auto output : split->outputs()) {
                const auto squeeze = std::make_shared<opset8::Squeeze>(output, axis);
                outputs.push_back(squeeze);
                to_copy_rt.push_back(squeeze);
            }
            copy_runtime_info({list_unpack, input_node}, to_copy_rt);
            replace_node(list_unpack, outputs);

            return true;
        }
        if (auto where = cast_fw_node(input_node, "aten::where")) {
            const auto input = where->get_input_source_output(0);
            auto non_zero = std::make_shared<opset8::NonZero>(input);
            auto axis = opset8::Constant::create(element::i64, Shape{}, {0});
            const auto num_splits = list_unpack->get_output_size();
            auto split = std::make_shared<opset8::Split>(non_zero, axis, num_splits);
            NodeVector to_copy_rt{split};
            OutputVector outputs;
            for (auto output : split->outputs()) {
                const auto squeeze = std::make_shared<opset8::Squeeze>(output, axis);
                outputs.push_back(squeeze);
                to_copy_rt.push_back(squeeze);
            }
            copy_runtime_info({list_unpack, input_node}, to_copy_rt);
            replace_node(list_unpack, outputs);

            return true;
        }
        if (auto nonzero_numpy = cast_fw_node(input_node, "aten::nonzero_numpy")) {
            const auto input = nonzero_numpy->get_input_source_output(0);
            auto non_zero = std::make_shared<opset8::NonZero>(input);
            auto axis = opset8::Constant::create(element::i64, Shape{}, {0});
            const auto num_splits = list_unpack->get_output_size();
            auto split = std::make_shared<opset8::Split>(non_zero, axis, num_splits);
            NodeVector to_copy_rt{split};
            OutputVector outputs;
            for (auto output : split->outputs()) {
                const auto squeeze = std::make_shared<opset8::Squeeze>(output, axis);
                outputs.push_back(squeeze);
                to_copy_rt.push_back(squeeze);
            }
            copy_runtime_info({list_unpack, input_node}, to_copy_rt);
            replace_node(list_unpack, outputs);

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

        if (auto slice = std::dynamic_pointer_cast<opset8::Slice>(input_node)) {
            // case aten::slice as input
            // Number of ListUnpack outputs should be equal to rank of input shape.
            auto axis_0 = opset8::Constant::create(element::i64, Shape{}, {0});
            auto split = std::make_shared<opset8::Split>(slice, axis_0, list_unpack->get_output_size());

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
