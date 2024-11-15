// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prim_list_unpack_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

PrimListUnpackReplacer::PrimListUnpackReplacer() {
    auto list_unpack = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto list_unpack = cast_fw_node(m.get_match_root(), "prim::ListUnpack");
        if (!list_unpack)
            return false;

        auto input_node = list_unpack->input_value(0).get_node_shared_ptr();
        ov::pass::NodeRegistry rg;
        if (auto torch_split = cast_fw_node(input_node, "aten::split")) {
            auto rank = torch_split->input(1).get_partial_shape().rank();
            if (rank.is_dynamic()) {
                add_exception_to_fw_node(torch_split, "aten::split: dynamic rank is not supported.");
                return false;
            }
            std::shared_ptr<Node> split;
            if (rank.get_length() == 0) {
                // Create split_lengths tensor from split_size int,
                // allow for last chunk to be smaller if data is not equally divisible.
                auto split_size = torch_split->get_input_source_output(1);
                // Using number of ListUnpack outputs.
                auto num_out_m_1 =
                    v0::Constant::create(split_size.get_element_type(), Shape{1}, {list_unpack->get_output_size() - 1});
                auto const_neg_1 = v0::Constant::create(split_size.get_element_type(), Shape{1}, {-1});
                auto split_lenghts_m_1 = rg.make<v0::Tile>(split_size, num_out_m_1);
                NodeVector concat_inputs{split_lenghts_m_1, const_neg_1};
                auto split_lenghts = rg.make<v0::Concat>(concat_inputs, 0);
                split = rg.make<v1::VariadicSplit>(torch_split->get_input_source_output(0),
                                                   torch_split->get_input_source_output(2),
                                                   split_lenghts);
            } else {
                split = rg.make<v1::VariadicSplit>(torch_split->get_input_source_output(0),
                                                   torch_split->get_input_source_output(2),
                                                   torch_split->get_input_source_output(1));
            }
            copy_runtime_info_and_name(list_unpack, rg.get(), {input_node});
            replace_node(list_unpack, split);

            return true;
        } else if (auto split_with_sizes = cast_fw_node(input_node, "aten::split_with_sizes")) {
            auto split_lengths = concat_list_construct(split_with_sizes->get_input_source_output(1));
            auto split = rg.make<v1::VariadicSplit>(split_with_sizes->get_input_source_output(0),
                                                    split_with_sizes->get_input_source_output(2),
                                                    split_lengths);

            copy_runtime_info_and_name(list_unpack, rg.get(), {input_node});
            replace_node(list_unpack, split);

            return true;
        } else if (auto chunk = cast_fw_node(input_node, {"aten::chunk", "aten::unsafe_chunk"})) {
            if (list_unpack->get_output_size() == 1) {
                list_unpack->output(0).replace(input_node->input_value(0));
                return true;
            }
            auto input_tensor = chunk->get_input_source_output(0);
            auto chunks = chunk->get_input_source_output(1);
            chunks = rg.make<v0::Convert>(chunks, element::i32);
            auto dim = chunk->get_input_source_output(2);

            auto tensor_0 = v0::Constant::create(element::i32, Shape{1}, {0});
            auto tensor_neg_1 = v0::Constant::create(element::i32, Shape{1}, {-1});

            auto input_shape = rg.make<v3::ShapeOf>(input_tensor, element::i32);
            auto input_dimension = rg.make<v8::Gather>(input_shape, dim, tensor_0);

            auto init_chunk_size = rg.make<v1::Divide>(input_dimension, chunks, true);

            // Add 1 if input is not evenly divisible by chunks
            auto last_chunk_size = rg.make<v1::Mod>(input_dimension, chunks);
            auto is_last_nonzero = rg.make<v1::Greater>(last_chunk_size, tensor_0);
            auto is_last_nonzero_int = rg.make<v0::Convert>(is_last_nonzero, element::i32);

            auto chunk_size = rg.make<v1::Add>(init_chunk_size, is_last_nonzero_int);

            auto split_lengths_even_size =
                v0::Constant::create(element::i32, Shape{1}, {list_unpack->get_output_size() - 1});
            auto split_lengths_even = rg.make<v3::Broadcast>(chunk_size, split_lengths_even_size);

            auto split_lengths = rg.make<v0::Concat>(OutputVector{split_lengths_even, tensor_neg_1}, 0);
            auto sliced_chunks = rg.make<v1::VariadicSplit>(input_tensor, dim, split_lengths);

            copy_runtime_info_and_name(list_unpack, rg.get(), {input_node});
            replace_node(list_unpack, sliced_chunks);

            return true;
        } else if (auto tensor_split = cast_fw_node(input_node, "aten::tensor_split")) {
            auto rank = tensor_split->input(1).get_partial_shape().rank();
            if (rank.is_dynamic()) {
                add_exception_to_fw_node(tensor_split, "aten::tensor_split: dynamic rank is not supported.");
                return false;
            }

            auto const_0 = v0::Constant::create(element::i32, Shape{1}, {0});
            auto const_1 = v0::Constant::create(element::i32, Shape{1}, {1});
            auto const_0_scalar = v0::Constant::create(element::i32, Shape{}, {0});
            auto const_1_scalar = v0::Constant::create(element::i32, Shape{}, {1});
            auto const_max = v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()});
            auto const_neg_1 = v0::Constant::create(element::i32, Shape{1}, {-1});

            auto input = tensor_split->get_input_source_output(0);
            auto indices_or_sections = tensor_split->get_input_source_output(1);
            indices_or_sections = std::make_shared<v0::Convert>(indices_or_sections, element::i32);
            auto dim = rg.make<v0::Unsqueeze>(tensor_split->get_input_source_output(2), const_0);
            auto list_num_outs = v0::Constant::create(element::i32, Shape{1}, {list_unpack->get_output_size()});
            auto list_num_outs_scalar = v0::Constant::create(element::i32, Shape{}, {list_unpack->get_output_size()});

            if (rank.get_length() == 0) {
                auto input_shape = rg.make<v3::ShapeOf>(input, element::i32);
                auto axis_size = rg.make<v8::Gather>(input_shape, dim, const_0);
                auto minimum_split_size = rg.make<v1::Divide>(axis_size, indices_or_sections);
                auto maximum_split_size = rg.make<v1::Add>(minimum_split_size, const_1);
                auto num_splits_with_max_size = rg.make<v1::Mod>(axis_size, indices_or_sections);
                auto num_splits_with_min_size = rg.make<v1::Subtract>(indices_or_sections, num_splits_with_max_size);
                auto splits_max_size = rg.make<v0::Tile>(maximum_split_size, num_splits_with_max_size);
                auto splits_min_size = rg.make<v0::Tile>(minimum_split_size, num_splits_with_min_size);

                auto split_sizes = rg.make<v0::Concat>(OutputVector{splits_max_size, splits_min_size}, 0);
                // Reshape is used to make number of outputs static.
                auto split_sizes_known_length = rg.make<v1::Reshape>(split_sizes, list_num_outs, false);
                auto splits = rg.make<v1::VariadicSplit>(input, dim, split_sizes_known_length);
                copy_runtime_info_and_name(list_unpack, rg.get(), {input_node});
                replace_node(list_unpack, splits->outputs());
                return true;
            } else {
                auto range = rg.make<v4::Range>(const_0_scalar, list_num_outs_scalar, const_1_scalar, element::i32);
                auto range_plus_1 = rg.make<v1::Add>(range, const_1);
                auto sections =
                    rg.make<v0::Concat>(OutputVector{const_0, std::move(indices_or_sections), const_max}, 0);

                auto starts_tensor = rg.make<v8::Slice>(sections, const_0, const_neg_1, const_1, const_0);
                auto starts =
                    rg.make<v1::Split>(starts_tensor, const_0_scalar, list_unpack->get_output_size())->outputs();
                auto stops_tensor = rg.make<v8::Slice>(sections, const_1, const_max, const_1, const_0);
                auto stops =
                    rg.make<v1::Split>(stops_tensor, const_0_scalar, list_unpack->get_output_size())->outputs();
                OutputVector outputs{};
                for (size_t i = 0; i < list_unpack->get_output_size(); i++) {
                    auto slice = rg.make<v8::Slice>(input, starts[i], stops[i], const_1, dim);
                    outputs.push_back(slice);
                }
                copy_runtime_info_and_name(list_unpack, rg.get(), {input_node});
                replace_node(list_unpack, outputs);
                return true;
            }
        } else if (auto broadcast_tensors = cast_fw_node(input_node, "aten::broadcast_tensors")) {
            auto tensors = cast_fw_node(broadcast_tensors->input_value(0).get_node_shared_ptr(), "prim::ListConstruct");
            if (!tensors) {
                add_exception_to_fw_node(input_node,
                                         "aten::broadcast_tensors: only prim::ListConstruct supported as input.");
                return false;
            }
            Output<Node> final_shape_t = v0::Constant::create(element::i32, Shape{}, {0});
            for (auto& input : tensors->inputs()) {
                auto tensor_shape = rg.make<v3::ShapeOf>(input.get_source_output(), element::i32);
                final_shape_t =
                    rg.make<v3::Broadcast>(final_shape_t, tensor_shape, ov::op::BroadcastType::BIDIRECTIONAL);
            }
            auto final_shape = rg.make<v3::ShapeOf>(final_shape_t, element::i32);
            OutputVector outputs;
            for (auto& input : tensors->inputs()) {
                outputs.push_back(rg.make<v3::Broadcast>(input.get_source_output(), final_shape));
            }
            copy_runtime_info_and_name(list_unpack, rg.get(), {input_node});
            replace_node(list_unpack, outputs);
            return true;
        } else if (auto unbind = cast_fw_node(input_node, "aten::unbind")) {
            const auto input = unbind->get_input_source_output(0);
            const auto axis = unbind->get_input_source_output(1);
            const auto num_splits = list_unpack->get_output_size();
            auto split = rg.make<v1::Split>(input, axis, num_splits);
            OutputVector outputs;
            for (auto& output : split->outputs()) {
                const auto squeeze = rg.make<v0::Squeeze>(output, axis);
                outputs.push_back(squeeze);
            }
            copy_runtime_info_and_name(list_unpack, rg.get(), {input_node});
            replace_node(list_unpack, outputs);

            return true;
        } else if (auto where = cast_fw_node(input_node, "aten::where")) {
            const auto input = where->get_input_source_output(0);
            auto non_zero = rg.make<v3::NonZero>(input);
            auto axis = v0::Constant::create(element::i32, Shape{}, {0});
            const auto num_splits = list_unpack->get_output_size();
            auto split = rg.make<v1::Split>(non_zero, axis, num_splits);
            OutputVector outputs;
            for (auto& output : split->outputs()) {
                const auto squeeze = rg.make<v0::Squeeze>(output, axis);
                outputs.push_back(squeeze);
            }
            copy_runtime_info_and_name(list_unpack, rg.get(), {input_node});
            replace_node(list_unpack, outputs);

            return true;
        } else if (auto nonzero_numpy = cast_fw_node(input_node, "aten::nonzero_numpy")) {
            const auto input = nonzero_numpy->get_input_source_output(0);
            auto non_zero = rg.make<v3::NonZero>(input);
            auto axis = v0::Constant::create(element::i32, Shape{}, {0});
            const auto num_splits = list_unpack->get_output_size();
            auto split = rg.make<v1::Split>(non_zero, axis, num_splits);
            OutputVector outputs;
            for (auto& output : split->outputs()) {
                const auto squeeze = rg.make<v0::Squeeze>(output, axis);
                outputs.push_back(squeeze);
            }
            copy_runtime_info_and_name(list_unpack, rg.get(), {input_node});
            replace_node(list_unpack, outputs);

            return true;
        } else if (auto meshgrid = cast_fw_node(input_node, "aten::meshgrid")) {
            // Input - ListConstruct
            auto meshgrid_input_node =
                cast_fw_node(meshgrid->input_value(0).get_node_shared_ptr(), "prim::ListConstruct");
            if (!meshgrid_input_node) {
                add_exception_to_fw_node(input_node, "aten::meshgrid: only prim::ListConstruct supported as input.");
                return false;
            }
            OutputVector meshgrid_inputs;
            for (auto& input : meshgrid_input_node->inputs()) {
                meshgrid_inputs.push_back(input.get_source_output());
            }

            auto meshgrid_attrs = meshgrid->get_attrs();
            if (meshgrid_attrs.find("indexing") == meshgrid_attrs.end()) {
                // Check if "indexing" key is available in meshgrid attributes set in translation.
                add_exception_to_fw_node(input_node, "aten::meshgrid: couldn't find indexing attribute.");
                return false;
            }
            std::string indexing = meshgrid_attrs.at("indexing");
            if (indexing != "ij" && indexing != "xy") {
                // Check if indexing attribute has correct values.
                add_exception_to_fw_node(input_node, "aten::meshgrid: unsupported indexing mode.");
                return false;
            }

            if (indexing == "xy" && meshgrid_inputs.size() >= 2) {
                std::swap(meshgrid_inputs[0], meshgrid_inputs[1]);
            }
            NodeVector cat_shapes{};
            NodeVector reshapes{};
            auto const_neg_1 = v0::Constant::create(element::i32, Shape{1}, {-1});
            auto const_1 = v0::Constant::create(element::i32, Shape{1}, {1});
            int input_idx = 0;
            for (auto& input : meshgrid_inputs) {
                auto reshaped_input = rg.make<v1::Reshape>(input, const_neg_1, false);
                auto shape = rg.make<v3::ShapeOf>(reshaped_input, element::i32);
                cat_shapes.push_back(shape);
                NodeVector cat_inputs(meshgrid_inputs.size(), const_1);
                cat_inputs[input_idx] = shape;
                input_idx++;
                auto input_cat = rg.make<v0::Concat>(cat_inputs, 0);
                auto reshape_cat = rg.make<v1::Reshape>(reshaped_input, input_cat, false);
                reshapes.push_back(reshape_cat);
            }
            auto cat = rg.make<v0::Concat>(cat_shapes, 0);
            OutputVector outputs{};
            for (auto& reshape : reshapes) {
                auto out = rg.make<v3::Broadcast>(reshape, cat, ov::op::BroadcastType::BIDIRECTIONAL);
                outputs.push_back(out);
            }
            if (indexing == "xy" && outputs.size() >= 2) {
                std::swap(outputs[0], outputs[1]);
            }
            copy_runtime_info_and_name(list_unpack, rg.get(), {input_node, meshgrid_input_node});
            replace_node(list_unpack, outputs);
            return true;
        } else if (auto shape_of = std::dynamic_pointer_cast<v3::ShapeOf>(input_node)) {
            // case aten::size as input
            // Number of ListUnpack outputs should be equal to rank of input shape.
            auto axis_0 = v0::Constant::create(element::i32, Shape{}, {0});
            auto split = rg.make<v1::Split>(shape_of, axis_0, list_unpack->get_output_size());

            OutputVector res;
            for (auto& output : split->outputs()) {
                auto squeeze = rg.make<v0::Squeeze>(output, axis_0);
                res.push_back(squeeze);
            }

            copy_runtime_info_and_name(list_unpack, rg.get(), {input_node});
            replace_node(list_unpack, res);

            return true;
        } else if (auto slice = std::dynamic_pointer_cast<v8::Slice>(input_node)) {
            // case aten::slice as input
            // Number of ListUnpack outputs should be equal to rank of input shape.
            auto axis_0 = v0::Constant::create(element::i32, Shape{}, {0});
            auto split = rg.make<v1::Split>(slice, axis_0, list_unpack->get_output_size());

            OutputVector res;
            for (auto& output : split->outputs()) {
                auto squeeze = rg.make<v0::Squeeze>(output, axis_0);
                res.push_back(squeeze);
            }

            copy_runtime_info_and_name(list_unpack, rg.get(), {input_node});
            replace_node(list_unpack, res);

            return true;
        }

        std::stringstream msg;
        msg << "prim::ListUnpack: unsupported input node: " << input_node;
        add_exception_to_fw_node(list_unpack, msg.str());
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
