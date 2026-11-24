// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_index_put_replacer.hpp"

#include <algorithm>
#include <numeric>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/visibility.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::pass {

using namespace ov::op;

AtenIndexPutReplacer::AtenIndexPutReplacer() {
    auto index_op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>(
        fw_node_predicate({"aten::index_put_", "aten.index_put.default", "aten.index_put_.default"}));

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto index_op = m.get_match_root();

        NodeVector rt_copy_from;
        ov::pass::NodeRegistry rg;
        auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
        auto const_1 = v0::Constant::create(element::i32, Shape{1}, {1});
        auto const_max_int = v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()});
        auto const_neg_1 = v0::Constant::create(element::i32, Shape{}, {-1});
        // Fold constant-only subgraphs immediately rather than leaving them to later passes.
        auto try_fold = [&rg](const Output<Node>& output) -> Output<Node> {
            if (auto folded = ov::util::get_constant_from_source(output)) {
                return rg.add(folded);
            }
            return output;
        };
        // Unsqueezes `count` consecutive axes starting at `first_axis`.
        auto unsqueeze_axes = [&](const Output<Node>& x, int64_t first_axis, int64_t count) {
            std::vector<int64_t> axes(count);
            std::iota(axes.begin(), axes.end(), first_axis);
            return try_fold(rg.make<v0::Unsqueeze>(x, v0::Constant::create(element::i64, Shape{axes.size()}, axes)));
        };

        auto input = index_op->input_value(0);
        auto input_shape = rg.make<v3::ShapeOf>(input, element::i32);
        auto indices = index_op->input_value(1);
        auto values = index_op->input_value(2);

        bool accumulate = false;
        if (index_op->get_input_size() > 3) {
            auto acc_const = ov::util::get_constant_from_source(index_op->input_value(3));
            if (!acc_const) {
                add_exception_to_fw_node(index_op, "aten::index_put_: non constant accumulate input is not supported.");
                return false;
            }
            accumulate = acc_const->cast_vector<bool>()[0];
        }

        int64_t indices_list_len;
        OutputVector indices_inputs;

        // Check for SequenceMark
        if (auto seq_mark = ov::as_type_ptr<SequenceMark>(indices.get_node_shared_ptr())) {
            rt_copy_from.push_back(seq_mark);
            indices_inputs = seq_mark->input_values();
            indices_list_len = static_cast<int64_t>(indices_inputs.size());
        } else {
            auto indices_partial_shape = indices.get_partial_shape();
            if (!indices_partial_shape.rank().is_static()) {
                // "We support only indices with static rank."
                add_exception_to_fw_node(index_op, "aten::index_put_: dynamic rank for indices is not supported.");
                return false;
            }
            const auto& indices_first_dim = indices_partial_shape[0];
            if (!indices_first_dim.is_static()) {
                // We support only lists of tensors with static number of elements.
                add_exception_to_fw_node(index_op,
                                         "aten::index_put_: dynamic dynamic number of indices is not supported.");
                return false;
            }
            indices_list_len = indices_first_dim.get_length();
            auto split = rg.make<v1::Split>(indices, const_0, indices_list_len);
            indices_inputs = split->outputs();
        }

        if (indices_list_len == 0) {
            replace_node(index_op, values.get_node_shared_ptr());
            return true;
        }

        // it is possible to have None values among indices_inputs, for example, tensor[:, 2, :, :] = update
        // None value corresponds to `:` that means to update all slices by this dimension
        // compute permutation vector so that all None's are moved to the tail like tensor[2, :, :, :] = update
        // so that it will be possible to use ScatterNDUpdate operation
        // Positions in the index list, used to permute values: each index contributes one
        // dimension to the indexed result.
        std::vector<int64_t> dims_with_none_idx;
        std::vector<int64_t> dims_with_value_idx;
        // Input dimensions covered by each index: a boolean mask covers as many dimensions as its rank.
        std::vector<int64_t> input_dims_with_none_idx;
        std::vector<int64_t> input_dims_with_value_idx;
        int64_t covered_input_dims = 0;
        ov::OutputVector indices_inputs_without_nones;
        for (int64_t idx = 0; idx < indices_list_len; ++idx) {
            const bool is_none = is_none_node(indices_inputs[idx]);
            int64_t span = 1;
            if (is_none) {
                dims_with_none_idx.push_back(idx);
            } else {
                dims_with_value_idx.push_back(idx);
                indices_inputs_without_nones.push_back(indices_inputs[idx]);
                const auto index_rank = indices_inputs[idx].get_partial_shape().rank();
                if (indices_inputs[idx].get_element_type() == element::boolean && index_rank.is_static()) {
                    span = index_rank.get_length();
                }
            }
            auto& input_dims = is_none ? input_dims_with_none_idx : input_dims_with_value_idx;
            for (int64_t i = 0; i < span; ++i) {
                input_dims.push_back(covered_input_dims++);
            }
        }

        std::vector<int64_t> perm_vector_before;
        std::vector<int64_t> perm_vector_after;
        std::vector<int64_t> values_perm;
        if (static_cast<int64_t>(dims_with_none_idx.size()) == indices_list_len) {
            // if there is an expression: tensor[:, :, :] = update
            // the whole tensor is replaced with update
            replace_node(index_op, values.get_node_shared_ptr());
            return true;
        } else if (dims_with_none_idx.size() > 0 &&
                   (indices_list_len - dims_with_none_idx[0]) != static_cast<int64_t>(dims_with_none_idx.size())) {
            // None indices are not placed in the tail, i.e. tensor[2, 10, :, 4, :] = update
            // so we need to transpose input tensor to place None indices to the tail
            // like tensor[2, 10, 4, :, :] = update
            if (!input.get_partial_shape().rank().is_static()) {
                // We support only lists of tensors with static number of elements.
                add_exception_to_fw_node(
                    index_op,
                    "aten::index_put_: input of dynamic rank with None indices not in the tail is not supported.");
                return false;
            }
            auto input_rank = input.get_partial_shape().rank().get_length();
            if (covered_input_dims > input_rank) {
                add_exception_to_fw_node(index_op, "aten::index_put_: indices cover more dimensions than the input.");
                return false;
            }
            // Indexed dimensions first, then None dimensions, then the trailing dimensions in place.
            perm_vector_before = input_dims_with_value_idx;
            perm_vector_before.insert(perm_vector_before.end(),
                                      input_dims_with_none_idx.begin(),
                                      input_dims_with_none_idx.end());
            for (int64_t dim = covered_input_dims; dim < input_rank; ++dim) {
                perm_vector_before.push_back(dim);
            }
            perm_vector_after.resize(input_rank);
            for (int64_t idx = 0; idx < input_rank; ++idx) {
                perm_vector_after[perm_vector_before[idx]] = idx;
            }

            // The same reordering for values: an index contributes a single result dimension.
            values_perm = dims_with_value_idx;
            values_perm.insert(values_perm.end(), dims_with_none_idx.begin(), dims_with_none_idx.end());
            const auto values_result_rank = input_rank - covered_input_dims + indices_list_len;
            for (int64_t dim = indices_list_len; dim < values_result_rank; ++dim) {
                values_perm.push_back(dim);
            }
        }

        // truncate indices with None values
        indices_inputs = indices_inputs_without_nones;
        indices_list_len = static_cast<int64_t>(indices_inputs.size());

        if (!perm_vector_before.empty()) {
            // since None values moved to that tail dimensions
            // it is required to adjust input and update tensors by dimension transposing
            auto perm_before = v0::Constant::create(element::i64, Shape{perm_vector_before.size()}, perm_vector_before);
            input = rg.make<v1::Transpose>(input, perm_before);
            input_shape = rg.make<v3::ShapeOf>(input, element::i32);
            // A scalar broadcasts in any dimension order and must not be transposed.
            const auto values_rank = values.get_partial_shape().rank();
            if (values_rank != 0) {
                // Values broadcast from the right: add missing leading dimensions before transposing.
                const auto perm_size = static_cast<int64_t>(values_perm.size());
                if (values_rank.is_static() && values_rank.get_length() < perm_size) {
                    const auto missing = perm_size - values_rank.get_length();
                    values = missing == 1 ? rg.make<v0::Unsqueeze>(values, const_0)->output(0)
                                          : unsqueeze_axes(values, 0, missing);
                }
                // Without multidimensional masks, values use the same order as the input.
                auto values_order = values_perm == perm_vector_before
                                        ? perm_before
                                        : v0::Constant::create(element::i64, Shape{values_perm.size()}, values_perm);
                values = rg.make<v1::Transpose>(values, values_order);
            }
        }

        // Restore the original dimension order for both masked and integer updates.
        auto finish = [&](Output<Node> result) {
            if (!perm_vector_after.empty()) {
                auto perm_after =
                    v0::Constant::create(element::i64, Shape{perm_vector_after.size()}, perm_vector_after);
                result = rg.make<v1::Transpose>(result, perm_after);
            }
            copy_runtime_info_and_name(index_op, rg.get(), rt_copy_from);
            replace_node(index_op, OutputVector{result});
            return true;
        };

        auto const_indices_list_len = v0::Constant::create(element::i32, Shape{1}, {indices_list_len});

        std::shared_ptr<Node> broadcast_index_shape;
        // Values are broadcast to [broadcast_index_shape, input_shape[sub_data_start:]].
        Output<Node> sub_data_start = const_indices_list_len;
        bool broadcast_values = true;
        Output<Node> index;
        if (indices_list_len > 1) {
            // PyTorch accepts mixed i32/i64 indices. Broadcast arithmetic and Concat
            // require a common type; widen i32 indices to avoid narrowing i64 values.
            const auto i64_index = std::find_if(indices_inputs.begin(), indices_inputs.end(), [](const auto& idx) {
                return idx.get_element_type() == element::i64;
            });
            if (i64_index != indices_inputs.end()) {
                for (auto& idx : indices_inputs) {
                    if (idx.get_element_type() == element::i32) {
                        idx = try_fold(rg.make<v1::ConvertLike>(idx, *i64_index));
                    }
                }
            }
            index = indices_inputs[0];
            for (int i = 1; i < indices_list_len; i++) {
                index = rg.make<v1::Add>(index, indices_inputs[i]);
            }
            broadcast_index_shape = rg.make<v3::ShapeOf>(index, element::i32);
            OutputVector indices_list;
            for (int i = 0; i < indices_list_len; i++) {
                auto broadcast = rg.make<v3::Broadcast>(indices_inputs[i], broadcast_index_shape);
                auto unsqueeze = rg.make<v0::Unsqueeze>(broadcast, const_neg_1);

                // change negative indices to positive indices
                auto const_i = v0::Constant::create(element::i32, Shape{}, {i});
                auto dim_i = rg.make<v8::Gather>(input_shape, const_i, const_0);
                auto dim_i_correct_type = rg.make<v1::ConvertLike>(dim_i, index);
                auto unsqueeze_add = rg.make<v1::Add>(unsqueeze, dim_i_correct_type);
                auto unsqueeze_add_mod = rg.make<v1::Mod>(unsqueeze_add, dim_i_correct_type);

                indices_list.push_back(unsqueeze_add_mod);
            }
            index = rg.make<v0::Concat>(indices_list, -1);
        } else {
            index = indices_inputs[0];
            auto index_dtype = index.get_element_type();
            // Do we need to also check u8?
            if (index_dtype == element::boolean) {
                auto value_pshape_rank = values.get_partial_shape().rank();
                if (value_pshape_rank.is_static() && value_pshape_rank.get_length() == 0) {
                    // The mask covers the leading dimensions, but Select broadcasts from the right.
                    auto mask = index;
                    auto in_rank = input.get_partial_shape().rank();
                    auto mask_rank = mask.get_partial_shape().rank();
                    if (in_rank.is_static() && mask_rank.is_static() && mask_rank.get_length() < in_rank.get_length()) {
                        mask =
                            unsqueeze_axes(mask, mask_rank.get_length(), in_rank.get_length() - mask_rank.get_length());
                    }
                    // A mask selects each element at most once, so accumulation is a plain add.
                    // For booleans, add means logical or, as in aten::add.
                    Output<Node> fill = values;
                    if (accumulate) {
                        auto converted = rg.make<v1::ConvertLike>(values, input);
                        if (input.get_element_type() == element::boolean) {
                            fill = rg.make<v1::LogicalOr>(input, converted);
                        } else {
                            fill = rg.make<v1::Add>(input, converted);
                        }
                    }
                    return finish(masked_fill(rg, input, mask, fill));
                }
                // then apply masked scatter
                auto nonzero = rg.make<v3::NonZero>(index);
                auto input_order = v0::Constant::create(element::i32, Shape{2}, {1, 0});
                index = rg.make<v1::Transpose>(nonzero, input_order);
                // ScatterNDUpdate needs values of shape [number of selected elements, input dimensions not
                // covered by the mask]. Broadcast unless values provably have that shape: dynamic dimensions may
                // be 1 at runtime, as may a static 1 against a dimension that is not 1.
                broadcast_values = false;
                const auto& values_pshape = values.get_partial_shape();
                const auto& index_pshape = index.get_partial_shape();
                const auto& input_pshape = input.get_partial_shape();
                if (values_pshape.rank().is_static() && index_pshape[1].is_static() &&
                    input_pshape.rank().is_static()) {
                    const auto mask_rank = index_pshape[1].get_length();
                    PartialShape expected{index_pshape[0]};
                    for (auto i = mask_rank; i < input_pshape.rank().get_length(); ++i) {
                        expected.push_back(input_pshape[i]);
                    }
                    broadcast_values = !values_pshape.is_static() || values_pshape.size() < expected.size();
                    for (size_t i = 0; !broadcast_values && i < values_pshape.size() && i < expected.size(); ++i) {
                        broadcast_values = values_pshape[i] == 1 && expected[i] != 1;
                    }
                    if (broadcast_values) {
                        auto const_0_1d = v0::Constant::create(element::i32, Shape{1}, {0});
                        auto index_shape = rg.make<v3::ShapeOf>(index, element::i32);
                        broadcast_index_shape = rg.make<v8::Gather>(index_shape, const_0_1d, const_0);
                        sub_data_start = v0::Constant::create(element::i32, Shape{1}, {mask_rank});
                    }
                }
            } else {
                // change negative indices to positive indices
                auto dim_0 = (rg.make<v8::Gather>(input_shape, const_0, const_0));
                auto dim_0_correct_type = (rg.make<v1::ConvertLike>(dim_0, index));
                index = rg.make<v1::Add>(index, dim_0_correct_type);
                index = rg.make<v1::Mod>(index, dim_0_correct_type);
                broadcast_index_shape = rg.make<v3::ShapeOf>(index, element::i32);
                index = rg.make<v0::Unsqueeze>(index, const_neg_1);
            }
        }

        if (broadcast_values) {
            auto sub_data_shape = rg.make<v8::Slice>(input_shape, sub_data_start, const_max_int, const_1);
            auto values_shape = rg.make<v0::Concat>(OutputVector{broadcast_index_shape, sub_data_shape}, 0);
            values = rg.make<v3::Broadcast>(values, values_shape);
        }
        values = rg.make<v1::ConvertLike>(values, input);

        if (accumulate) {
            return finish(rg.make<v15::ScatterNDUpdate>(input, index, values, v15::ScatterNDUpdate::Reduction::SUM));
        }
        return finish(rg.make<v3::ScatterNDUpdate>(input, index, values));
    };

    auto m =
        std::make_shared<ov::pass::pattern::Matcher>(index_op, "ov::frontend::pytorch::pass::AtenIndexPutReplacer");
    this->register_matcher(m, callback);
}

}  // namespace ov::frontend::pytorch::pass
