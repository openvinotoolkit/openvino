// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_index_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/pytorch/visibility.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

namespace {
Output<Node> flatten(ov::pass::NodeRegistry& rg, const Output<Node>& value, size_t axis) {
    // First dimension of output tensor is the product of [d_0, ... d_{axis-1}] dimensions of
    // input tensor. The last dimension is the product of the rest of input tensor dimensions:
    // [d_{axis}, ..., d_n]
    Output<Node> output_shape;
    if (axis == 0) {
        output_shape = v0::Constant::create(element::i32, Shape{2}, {1, -1});
    } else if (axis == 1) {
        output_shape = v0::Constant::create(element::i32, Shape{2}, {0, -1});
    } else {
        const auto value_shape = rg.make<v3::ShapeOf>(value, element::i32);
        const auto value_rank = rg.make<v3::ShapeOf>(value_shape, element::i32);
        const auto axis_node = v0::Constant::create(element::i32, Shape{1}, {axis});
        auto start = v0::Constant::create(element::i32, Shape{1}, {0});
        auto step = v0::Constant::create(element::i32, Shape{1}, {1});
        const auto first_part_dims = rg.make<v8::Slice>(value_shape, start, axis_node, step);
        auto zero = v0::Constant::create(element::i32, {}, {0});
        auto first_part_dims_length = rg.make<v1::ReduceProd>(first_part_dims, zero, true);

        auto remaining_part_length = v0::Constant::create(element::i32, {1}, {-1});

        output_shape = rg.make<v0::Concat>(OutputVector{first_part_dims_length, remaining_part_length}, 0);
    }
    return rg.make<v1::Reshape>(value, output_shape, true);
}
};  // namespace

AtenIndexToSelect::AtenIndexToSelect() {
    auto index_op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto index_op = cast_fw_node(m.get_match_root(), "aten::index");
        if (!index_op) {
            return false;
        }
        ov::pass::NodeRegistry rg;
        auto input_node = index_op->input_value(0);
        auto indicies = index_op->input_value(1).get_node_shared_ptr();
        auto list_indicies = cast_fw_node(indicies, "prim::ListConstruct");
        if (list_indicies) {
            // Multiple tensors as indices. Each tensor could either be
            //   1. prim::Constant()
            //           representing ":" in python indexing. E.g. tensor[:, :]
            //   2. prim::Constant[value=...] or tensor output
            //           representing advanced indexing. E.g. tensor[[0, 1], [2, 0]].
            // For more info on advanced indexing,
            // check https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing

            // Consider a general case of
            //       t: [x_1, y_1, y_2, ..., x_m, ..., y_n]
            // where t is a tensor of rank m+n, {x_i} are axes where tensor index is provided, and {y_i} are axes for
            // ":". Same results can be achieved through transposing t into
            //       t: [x_1, x_2, ..., x_m, y_1, y_2, ..., y_n]
            // and use gather
            //       t: [x_1 * x_2 * ... * x_m, y_1 * y_2 * ... * y_n]
            //       tensor index = \sum_{i=1}^m (ind_i * \prod_{j=i+1}^m (x_j))
            // After gather, reshape and transpose back.
            auto ids = list_indicies->input_values();
            std::vector<size_t> advanced_ids;
            std::vector<bool> is_masked_bool;
            OutputVector masked_indicies;
            // for case when index is bool e.g. x[x>0], replace index with non_zero
            for (size_t i = 0; i < ids.size(); i++) {
                auto const_input = cast_fw_node(ids[i].get_node_shared_ptr(), "prim::Constant");

                // skip dimensions where index is None
                if (const_input) {
                    const auto& attrs = const_input->get_attrs();
                    if (attrs.find("none_value") != attrs.end()) {
                        masked_indicies.push_back(ids[i]);
                        is_masked_bool.push_back(false);
                        continue;
                    }
                }
                auto id_dtype = ids[i].get_element_type();
                if (id_dtype == element::boolean || id_dtype == element::u8) {
                    auto idx = rg.make<v0::Convert>(ids[i], element::u8);
                    auto nonzero = rg.make<v3::NonZero>(idx, element::i32);
                    auto input_order = v0::Constant::create(element::i32, Shape{2}, {1, 0});
                    auto masked_id = rg.make<v1::Transpose>(nonzero, input_order);
                    masked_indicies.push_back(masked_id);
                    is_masked_bool.push_back(true);
                } else {
                    masked_indicies.push_back(ids[i]);
                    is_masked_bool.push_back(false);
                }
                advanced_ids.push_back(i);
            }

            // all indicies prim::Constant(None), return input as is
            if (advanced_ids.size() == 0) {
                index_op->output(0).replace(index_op->get_input_source_output(0));
                return true;
            }
            // perform gather for single element case
            if (advanced_ids.size() == 1) {
                auto index = masked_indicies[advanced_ids[0]];
                if (is_masked_bool[advanced_ids[0]]) {
                    auto gather = rg.make<v8::GatherND>(input_node, index);
                    copy_runtime_info_and_name(index_op, rg.get());
                    replace_node(index_op, gather);
                    return true;
                }
                index = rg.make<v0::Convert>(index, element::i32);
                auto dim = v0::Constant::create(element::i32, Shape{}, {advanced_ids[0]});
                auto gather = rg.make<v8::Gather>(input_node, index, dim);
                copy_runtime_info_and_name(index_op, rg.get());
                replace_node(index_op, gather);
                return true;
            }
            auto adv_idx_count = advanced_ids.size();
            auto rank = input_node.get_partial_shape().rank();
            // index transformation supports only tensors with static rank
            if (rank.is_dynamic()) {
                add_exception_to_fw_node(index_op, "aten::index: dynamic rank for aten::index input is not supported.");
                return false;
            }
            auto input_shape = rg.make<v3::ShapeOf>(input_node, element::i32);
            auto zero = v0::Constant::create(element::i32, Shape{}, {0});
            auto input_dims = rg.make<v1::Split>(input_shape, zero, rank.get_length());
            std::vector<size_t> non_used_dims;
            for (auto i = 0; i < rank.get_length(); i++) {
                if (std::find(advanced_ids.begin(), advanced_ids.end(), i) == advanced_ids.end()) {
                    non_used_dims.push_back(i);
                }
            }
            std::vector<size_t> permutation_dims;
            permutation_dims.insert(permutation_dims.end(), advanced_ids.begin(), advanced_ids.end());
            permutation_dims.insert(permutation_dims.end(), non_used_dims.begin(), non_used_dims.end());
            auto transpose_dims = v0::Constant::create(element::i32, Shape{permutation_dims.size()}, permutation_dims);
            auto transposed_input = rg.make<v1::Transpose>(input_node, transpose_dims);
            auto flatten_input = flatten(rg, transposed_input, adv_idx_count);
            auto cum_adv_index = masked_indicies[advanced_ids[adv_idx_count - 1]];
            cum_adv_index = rg.make<v0::Convert>(cum_adv_index, element::i32);
            auto multiplier = input_dims->output(advanced_ids[adv_idx_count - 1]);
            for (int i = static_cast<int>(adv_idx_count) - 2; i > -1; i--) {
                auto input_id = advanced_ids[i];
                auto m_idx = rg.make<v0::Convert>(masked_indicies[input_id], element::i32);
                auto adv_index = rg.make<v1::Multiply>(m_idx, multiplier);
                cum_adv_index = rg.make<v1::Add>(cum_adv_index, adv_index);
                multiplier = rg.make<v1::Multiply>(multiplier, input_dims->output(input_id));
            }
            std::shared_ptr<Node> gather = rg.make<v8::Gather>(flatten_input, cum_adv_index, zero);
            OutputVector concat_dims;
            // check if all advanced indices are consecutive.
            std::vector<size_t> consequence_dims;
            auto cum_adv_index_shape_tensor = rg.make<v3::ShapeOf>(cum_adv_index, element::i32);
            for (size_t i = advanced_ids[0]; i <= advanced_ids[advanced_ids.size() - 1]; i++) {
                consequence_dims.push_back(i);
            }
            // unfold regular index axes
            if (advanced_ids == consequence_dims) {
                OutputVector folded_adv_idx_shape_vector;
                auto minus_one = v0::Constant::create(element::i32, Shape{1}, {-1});
                folded_adv_idx_shape_vector.push_back(minus_one);
                for (auto i : non_used_dims) {
                    folded_adv_idx_shape_vector.push_back(input_dims->output(i));
                }
                auto folded_adv_idx_shape = rg.make<v0::Concat>(folded_adv_idx_shape_vector, 0);
                gather = rg.make<v1::Reshape>(gather, folded_adv_idx_shape, false);
                std::vector<size_t> adv_idx_permute;
                for (size_t i = 1; i < advanced_ids[0] + 1; i++) {
                    adv_idx_permute.push_back(i);
                }
                adv_idx_permute.push_back(0);
                for (size_t i = advanced_ids[0] + 1; i < (rank.get_length() - adv_idx_count + 1); i++) {
                    adv_idx_permute.push_back(i);
                }
                // Transpose folded advanced indexed axis to its original location.
                auto permute_indicies =
                    v0::Constant::create(element::i32, Shape{adv_idx_permute.size()}, adv_idx_permute);
                gather = rg.make<v1::Transpose>(gather, permute_indicies);
                // unfold advanced index axes
                for (size_t i = 0; i < advanced_ids[0]; i++) {
                    concat_dims.push_back(input_dims->output(i));
                }
                concat_dims.push_back(cum_adv_index_shape_tensor);
                for (auto i : non_used_dims) {
                    if (i < advanced_ids[0]) {
                        continue;
                    }
                    concat_dims.push_back(input_dims->output(i));
                }

            } else {
                concat_dims.push_back(cum_adv_index_shape_tensor);
                for (auto i : non_used_dims) {
                    concat_dims.push_back(input_dims->output(i));
                }
            }
            auto final_shape = rg.make<v0::Concat>(concat_dims, 0);
            gather = rg.make<v1::Reshape>(gather, final_shape, false);
            copy_runtime_info_and_name(index_op, rg.get());
            replace_node(index_op, gather);
            return true;

        } else {
            auto const_input = cast_fw_node(indicies, "prim::Constant");

            if (const_input) {
                // index is None, stay input as is
                const auto& attrs = const_input->get_attrs();
                if (attrs.find("none_value") != attrs.end()) {
                    index_op->output(0).replace(index_op->get_input_source_output(0));
                    return true;
                }
            }
            auto index_dtype = indicies->get_output_element_type(0);
            if (index_dtype == element::boolean || index_dtype == element::u8) {
                auto nonzero = rg.make<v3::NonZero>(indicies, element::i32);
                auto input_order = v0::Constant::create(element::i32, Shape{2}, {1, 0});
                auto masked_id = rg.make<v1::Transpose>(nonzero, input_order);
                auto gather = rg.make<v8::GatherND>(input_node, masked_id);
                copy_runtime_info_and_name(index_op, rg.get());
                replace_node(index_op, gather);
                return true;
            }
            if (index_dtype != element::i32) {
                indicies = rg.make<ov::op::v0::Convert>(indicies, element::i32);
            }
            auto dim = v0::Constant::create(element::i32, Shape{}, {0});
            auto gather = rg.make<v8::Gather>(input_node, indicies, dim);
            copy_runtime_info_and_name(index_op, rg.get());
            replace_node(index_op, gather);
            return true;
        }
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(index_op, "ov::frontend::pytorch::pass::AtenIndexToSelect");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
