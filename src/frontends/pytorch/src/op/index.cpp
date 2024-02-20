// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

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

OutputVector index_on_list(ov::pass::NodeRegistry& rg,
                           const Output<Node>& data,
                           std::deque<Output<Node>> ids,
                           int64_t rank) {
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
    std::vector<size_t> advanced_ids;
    std::vector<bool> is_masked_bool;
    OutputVector masked_indicies;
    // for case when index is bool e.g. x[x>0], replace index with non_zero
    for (size_t i = 0; i < ids.size(); i++) {
        // skip dimensions where index is None
        bool is_none = false;
        if (!ids[i].get_node_shared_ptr()) {
            is_none = true;
        }
        if (auto const_input = cast_fw_node(ids[i].get_node_shared_ptr(), "prim::Constant")) {
            const auto& attrs = const_input->get_attrs();
            if (attrs.find("none_value") != attrs.end()) {
                is_none = true;
            }
        }
        if (is_none) {
            masked_indicies.push_back(ids[i]);
            is_masked_bool.push_back(false);
            continue;
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
        return {data};
    }
    // perform gather for single element case
    if (advanced_ids.size() == 1) {
        auto index = masked_indicies[advanced_ids[0]];
        if (is_masked_bool[advanced_ids[0]]) {
            auto gather = rg.make<v8::GatherND>(data, index);
            return {gather};
        }
        index = rg.make<v0::Convert>(index, element::i32);
        auto dim = v0::Constant::create(element::i32, Shape{}, {advanced_ids[0]});
        auto gather = rg.make<v8::Gather>(data, index, dim);
        return {gather};
    }
    auto adv_idx_count = advanced_ids.size();
    auto input_shape = rg.make<v3::ShapeOf>(data, element::i32);
    auto zero = v0::Constant::create(element::i32, Shape{}, {0});
    auto input_dims = rg.make<v1::Split>(input_shape, zero, rank);
    std::vector<size_t> non_used_dims;
    for (auto i = 0; i < rank; i++) {
        if (std::find(advanced_ids.begin(), advanced_ids.end(), i) == advanced_ids.end()) {
            non_used_dims.push_back(i);
        }
    }
    std::vector<size_t> permutation_dims;
    permutation_dims.insert(permutation_dims.end(), advanced_ids.begin(), advanced_ids.end());
    permutation_dims.insert(permutation_dims.end(), non_used_dims.begin(), non_used_dims.end());
    auto transpose_dims = v0::Constant::create(element::i32, Shape{permutation_dims.size()}, permutation_dims);
    auto transposed_input = rg.make<v1::Transpose>(data, transpose_dims);
    auto flatten_input = flatten(rg, transposed_input, adv_idx_count);
    auto cum_adv_index = masked_indicies[advanced_ids.back()];
    cum_adv_index = rg.make<v0::Convert>(cum_adv_index, element::i32);
    auto multiplier = input_dims->output(advanced_ids.back());
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
    for (size_t i = advanced_ids[0]; i <= advanced_ids[advanced_ids.back()]; i++) {
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
        for (size_t i = advanced_ids[0] + 1; i < (rank - adv_idx_count + 1); i++) {
            adv_idx_permute.push_back(i);
        }
        // Transpose folded advanced indexed axis to its original location.
        auto permute_indicies = v0::Constant::create(element::i32, Shape{adv_idx_permute.size()}, adv_idx_permute);
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
        size_t i = 0;
        auto one = v0::Constant::create(element::i32, Shape{1}, {1});
        while (i < non_used_dims.size() && non_used_dims[i] < advanced_ids[0]) {
            concat_dims.push_back(one);
            i++;
        }
        concat_dims.push_back(cum_adv_index_shape_tensor);
        for (; i < non_used_dims.size(); i++) {
            concat_dims.push_back(input_dims->output(non_used_dims[i]));
        }
    }
    auto final_shape = rg.make<v0::Concat>(concat_dims, 0);
    gather = rg.make<v1::Reshape>(gather, final_shape, false);
    return {gather};
}
}  // namespace

OutputVector translate_index(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    if (context.input_is_none(1)) {
        return {x};
    }
    auto indices = context.get_input(1);
    auto index_dtype = context.get_input_type(1);
    if (index_dtype.is<type::List>()) {
        auto list_elems = get_list_as_outputs(indices);
        ov::pass::NodeRegistry rg;
        auto rank = x.get_partial_shape().rank();
        // index transformation supports only tensors with static rank
        PYTORCH_OP_CONVERSION_CHECK(rank.is_static(), "Dynamic rank for aten::index input is not supported.");
        auto res = index_on_list(rg, x, list_elems, rank.get_length());
        context.mark_nodes(rg.get());
        return res;
    }
    auto index_ov_type = indices.get_element_type();
    if (index_ov_type.is_dynamic()) {
        if (simplified_type_interpret(index_dtype).is<element::Type>()) {
            index_ov_type = index_dtype.as<element::Type>();
        }
    }
    if (index_ov_type == element::boolean || index_ov_type == element::u8) {
        auto nonzero = context.mark_node(std::make_shared<v3::NonZero>(indices, element::i32));
        auto input_order = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
        auto masked_id = context.mark_node(std::make_shared<v1::Transpose>(nonzero, input_order));
        auto gather = context.mark_node(std::make_shared<v8::GatherND>(x, masked_id));
        return {gather};
    }
    if (index_ov_type != element::i32) {
        indices = context.mark_node(std::make_shared<ov::op::v0::Convert>(indices, element::i32));
    }
    auto dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    return {context.mark_node(std::make_shared<v8::Gather>(x, indices, dim))};
};

OutputVector translate_index_fx(const NodeContext& context) {
    num_inputs_check(context, 2, context.get_input_size());
    auto x = context.get_input(0);
    std::deque<Output<Node>> list_elems;
    for (size_t i = 1; i < context.get_input_size(); i++) {
        Output<Node> index;
        if (!context.input_is_none(i)) {
            index = context.get_input(static_cast<int>(i));
        }
        list_elems.push_back(index);
    }
    ov::pass::NodeRegistry rg;
    auto rank = x.get_partial_shape().rank();
    if (rank.is_dynamic()) {
        rank = context.get_decoder()->get_input_shape(0).rank();
    }
    // index transformation supports only tensors with static rank
    PYTORCH_OP_CONVERSION_CHECK(rank.is_static(), "Dynamic rank for aten::index input is not supported.");
    auto res = index_on_list(rg, x, list_elems, rank.get_length());
    context.mark_nodes(rg.get());
    return res;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
