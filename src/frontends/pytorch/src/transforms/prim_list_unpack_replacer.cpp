// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prim_list_unpack_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/variadic_split.hpp"
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
                auto num_out_m_1 = ov::op::v0::Constant::create(split_size.get_element_type(),
                                                                Shape{1},
                                                                {list_unpack->get_output_size() - 1});
                auto const_neg_1 = ov::op::v0::Constant::create(split_size.get_element_type(), Shape{1}, {-1});
                auto split_lenghts_m_1 = std::make_shared<ov::op::v0::Tile>(split_size, num_out_m_1);
                NodeVector concat_inputs{split_lenghts_m_1, const_neg_1};
                auto split_lenghts = std::make_shared<ov::op::v0::Concat>(concat_inputs, 0);
                auto split = std::make_shared<ov::op::v1::VariadicSplit>(torch_split->get_input_source_output(0),
                                                                         torch_split->get_input_source_output(2),
                                                                         split_lenghts);
                copy_runtime_info({list_unpack, input_node}, split);
                replace_node(list_unpack, split);
            } else {
                auto split = std::make_shared<ov::op::v1::VariadicSplit>(torch_split->get_input_source_output(0),
                                                                         torch_split->get_input_source_output(2),
                                                                         torch_split->get_input_source_output(1));
                copy_runtime_info({list_unpack, input_node}, split);
                replace_node(list_unpack, split);
            }

            return true;
        }

        if (auto split_with_sizes = cast_fw_node(input_node, "aten::split_with_sizes")) {
            auto split = std::make_shared<ov::op::v1::VariadicSplit>(split_with_sizes->get_input_source_output(0),
                                                                     split_with_sizes->get_input_source_output(2),
                                                                     split_with_sizes->get_input_source_output(1));

            copy_runtime_info({list_unpack, input_node}, split);
            replace_node(list_unpack, split);

            return true;
        }

        if (auto chunk = cast_fw_node(input_node, "aten::chunk")) {
            // Using number of ListUnpack outputs instead of 1st input to chunk.
            // TODO: confirm it works for all cases
            auto split = std::make_shared<ov::op::v1::Split>(chunk->get_input_source_output(0),
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
            auto split = std::make_shared<ov::op::v1::Split>(input, axis, num_splits);
            NodeVector to_copy_rt{split};
            OutputVector outputs;
            for (auto output : split->outputs()) {
                const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(output, axis);
                outputs.push_back(squeeze);
                to_copy_rt.push_back(squeeze);
            }
            copy_runtime_info({list_unpack, input_node}, to_copy_rt);
            replace_node(list_unpack, outputs);

            return true;
        }
        if (auto where = cast_fw_node(input_node, "aten::where")) {
            const auto input = where->get_input_source_output(0);
            auto non_zero = std::make_shared<ov::op::v3::NonZero>(input);
            auto axis = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
            const auto num_splits = list_unpack->get_output_size();
            auto split = std::make_shared<ov::op::v1::Split>(non_zero, axis, num_splits);
            NodeVector to_copy_rt{split};
            OutputVector outputs;
            for (auto output : split->outputs()) {
                const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(output, axis);
                outputs.push_back(squeeze);
                to_copy_rt.push_back(squeeze);
            }
            copy_runtime_info({list_unpack, input_node}, to_copy_rt);
            replace_node(list_unpack, outputs);

            return true;
        }
        if (auto nonzero_numpy = cast_fw_node(input_node, "aten::nonzero_numpy")) {
            const auto input = nonzero_numpy->get_input_source_output(0);
            auto non_zero = std::make_shared<ov::op::v3::NonZero>(input);
            auto axis = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
            const auto num_splits = list_unpack->get_output_size();
            auto split = std::make_shared<ov::op::v1::Split>(non_zero, axis, num_splits);
            NodeVector to_copy_rt{split};
            OutputVector outputs;
            for (auto output : split->outputs()) {
                const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(output, axis);
                outputs.push_back(squeeze);
                to_copy_rt.push_back(squeeze);
            }
            copy_runtime_info({list_unpack, input_node}, to_copy_rt);
            replace_node(list_unpack, outputs);

            return true;
        }

        if (auto meshgrid = cast_fw_node(input_node, "aten::meshgrid")) {
            // Input - ListConstruct
            auto meshgrid_input_node =
                cast_fw_node(meshgrid->input_value(0).get_node_shared_ptr(), "prim::ListConstruct");
            if (!meshgrid_input_node) {
                return false;
            }
            NodeVector rt_copy_from{list_unpack, input_node, meshgrid_input_node};
            OutputVector meshgrid_inputs;
            for (auto& input : meshgrid_input_node->inputs()) {
                meshgrid_inputs.push_back(input.get_source_output());
            }

            auto meshgrid_attrs = meshgrid->get_attrs();
            if (meshgrid_attrs.find("indexing") == meshgrid_attrs.end()) {
                // Check if "indexing" key is available in meshgrid attributes set in translation.
                return false;
            }
            std::string indexing = meshgrid_attrs.at("indexing");
            if (indexing != "ij" && indexing != "xy") {
                // Check if indexing attribute has correct values.
                return false;
            }

            if (indexing == "xy" && meshgrid_inputs.size() >= 2) {
                std::swap(meshgrid_inputs[0], meshgrid_inputs[1]);
            }
            NodeVector cat_shapes{};
            NodeVector reshapes{};
            auto const_neg_1 = ov::op::v0::Constant::create(element::i64, Shape{1}, {-1});
            auto const_1 = ov::op::v0::Constant::create(element::i64, Shape{1}, {1});
            int input_idx = 0;
            for (auto& input : meshgrid_inputs) {
                auto reshaped_input = std::make_shared<ov::op::v1::Reshape>(input, const_neg_1, false);
                auto shape = std::make_shared<ov::op::v3::ShapeOf>(reshaped_input);
                cat_shapes.push_back(shape);
                NodeVector cat_inputs{};
                for (int i = 0; i < meshgrid_inputs.size(); i++) {
                    cat_inputs.push_back(const_1);
                }
                cat_inputs[input_idx] = shape;
                input_idx++;
                auto input_cat = std::make_shared<ov::op::v0::Concat>(cat_inputs, 0);
                auto reshape_cat = std::make_shared<ov::op::v1::Reshape>(reshaped_input, input_cat, false);
                reshapes.push_back(reshape_cat);
            }
            auto cat = std::make_shared<ov::op::v0::Concat>(cat_shapes, 0);
            NodeVector to_copy_rt{cat};
            to_copy_rt.push_back(cat);
            OutputVector outputs{};
            for (auto& reshape : reshapes) {
                auto out = std::make_shared<ov::op::v3::Broadcast>(reshape, cat, ov::op::BroadcastType::BIDIRECTIONAL);
                to_copy_rt.push_back(out);
                outputs.push_back(out);
            }
            if (indexing == "xy" && meshgrid_inputs.size() >= 2) {
                std::swap(outputs[0], outputs[1]);
            }
            copy_runtime_info(rt_copy_from, to_copy_rt);
            replace_node(list_unpack, outputs);
            return true;
        }

        if (auto shape_of = std::dynamic_pointer_cast<ov::op::v3::ShapeOf>(input_node)) {
            // case aten::size as input
            // Number of ListUnpack outputs should be equal to rank of input shape.
            auto axis_0 = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
            auto split = std::make_shared<ov::op::v1::Split>(shape_of, axis_0, list_unpack->get_output_size());

            NodeVector to_copy_rt{axis_0, split};
            OutputVector res;
            for (auto output : split->outputs()) {
                auto squeeze = std::make_shared<ov::op::v0::Squeeze>(output, axis_0);
                to_copy_rt.push_back(squeeze);
                res.push_back(squeeze);
            }

            copy_runtime_info({list_unpack, input_node}, to_copy_rt);
            replace_node(list_unpack, res);

            return true;
        }

        if (auto slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(input_node)) {
            // case aten::slice as input
            // Number of ListUnpack outputs should be equal to rank of input shape.
            auto axis_0 = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
            auto split = std::make_shared<ov::op::v1::Split>(slice, axis_0, list_unpack->get_output_size());

            NodeVector to_copy_rt{axis_0, split};
            OutputVector res;
            for (auto output : split->outputs()) {
                auto squeeze = std::make_shared<ov::op::v0::Squeeze>(output, axis_0);
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
