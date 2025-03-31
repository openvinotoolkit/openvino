// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gemm_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>
#include <utility>
#include <algorithm>

#include "intel_gpu/op/gemm.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gemm)

layout gemm_inst::calc_output_layout(gemm_node const& node, kernel_impl_params const& impl_param) {
    auto prim = impl_param.typed_desc<gemm>();

    auto input0_layout = impl_param.get_input_layout(0);
    auto input1_layout = impl_param.get_input_layout(1);

    auto input0_shape = input0_layout.get_shape();
    auto input1_shape = input1_layout.get_shape();

    auto input0_transpose_order = prim->input0_transpose_order;
    auto input1_transpose_order = prim->input1_transpose_order;

    bool reordered = prim->input_rank > 4 || prim->weight_rank > 4;
    size_t output_rank = std::max(prim->input_rank, prim->weight_rank);
    size_t input_rank = reordered ? output_rank : prim->input_rank;
    size_t weight_rank = reordered ? output_rank : prim->weight_rank;

    auto update_input_shape = [&output_rank](const ov::Shape& input_shape, size_t rank, std::vector<int64_t> input_order, bool first_input) {
        auto input_shape_update = ov::Shape();
        auto _input_shape_update = ov::Shape(input_shape.begin(), input_shape.begin() + std::min(rank, input_shape.size()));
        if (_input_shape_update.size() == input_order.size() && input_order.size() > 1) {
            for (auto idx : input_order) {
                input_shape_update.push_back(_input_shape_update[idx]);
            }
        } else {
            input_shape_update = _input_shape_update;
        }
        if (input_shape_update.size() == 1) {
            first_input ? input_shape_update.insert(input_shape_update.begin(), 1)
                        : input_shape_update.insert(input_shape_update.end(), 1);
            output_rank = std::max(output_rank, rank + 1);
        }
        input_shape_update.insert(input_shape_update.begin(), output_rank - input_shape_update.size(), 1);
        return input_shape_update;
    };

    auto transpose_shape = [](const ov::Shape& shape, const std::vector<int64_t>& order) {
        auto shape_transposed = ov::Shape(shape);
        auto rank_diff = shape.size() - order.size();
        for (size_t i = 0; i < order.size(); i++) {
            size_t idx = static_cast<size_t>(order[i]);
            shape_transposed[i + rank_diff] = shape[idx + rank_diff];
        }

        return shape_transposed;
    };

    auto input0_shape_update = update_input_shape(input0_shape, input_rank, input0_transpose_order, true);
    auto input1_shape_update = update_input_shape(input1_shape, weight_rank, input1_transpose_order, false);

    ov::Shape bias_shape(output_rank);
    if (prim->input_size() == 3) {
        bias_shape = impl_param.get_input_layout(2).get_shape();
        bias_shape = update_input_shape(bias_shape, weight_rank, input1_transpose_order, false);
    }

    auto output_shape = input0_shape_update;
    for (size_t i = 0; i < output_rank; ++i) {
        output_shape[i] = std::max(std::max(input0_shape_update[i], input1_shape_update[i]), bias_shape[i]);
    }

    size_t M = *(input0_shape_update.end() - 2);
    size_t N = input1_shape_update.back();

    output_shape[output_rank - 2] = M;
    output_shape[output_rank - 1] = N;

    size_t ones_to_add = 4 - std::min(output_shape.size(), static_cast<size_t>(4));
    output_shape.insert(output_shape.begin(), ones_to_add, 1);

    if (prim->output_transpose_order.size() > 0)
        output_shape = transpose_shape(output_shape, prim->output_transpose_order);

    auto output_type = input0_layout.data_type;
    if ((output_type == data_types::u8 || output_type == data_types::i8) && prim->output_data_types[0])
        output_type = *prim->output_data_types[0];

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    auto output_format = input0_layout.format;

    if (node.get_preferred_impl_type() == impl_types::onednn && node.get_preferred_output_fmt() != format::any) {
        output_format = node.get_preferred_output_fmt();
    }

    return layout(output_shape, output_type, output_format, prim->output_paddings[0]);
}

template<typename ShapeType>
std::vector<layout> gemm_inst::calc_output_layouts(gemm_node const& node, const kernel_impl_params& impl_param) {
    auto prim = impl_param.typed_desc<gemm>();
    auto input0_layout = impl_param.get_input_layout(0);
    auto input1_layout = impl_param.get_input_layout(1);

    auto default_out_dt = data_type_traits::is_floating_point(input0_layout.data_type) ? input0_layout.data_type : data_types::f32;
    auto output_type = prim->output_data_types[0].value_or(default_out_dt);

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    ov::intel_gpu::op::Gemm op;
    op.set_transpose_a(false);
    op.set_transpose_b(false);

    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        input1_layout.get<ShapeType>()
    };

    std::vector<ShapeType> output_shapes = ov::intel_gpu::op::shape_infer(&op,
                                                                          input_shapes,
                                                                          prim->input0_transpose_order,
                                                                          prim->input1_transpose_order,
                                                                          prim->output_transpose_order);

    cldnn::format output_format = input0_layout.format;
    if (output_shapes[0].size() > output_format.dimension()) {
        // This happened when input0.rank=2, input1.rank=5, output0.rank=5.
        // Output should use format like bfzyx, but it was taken from input0_layout which is bfyx.
        // Therefore, adjust output_format to proper rank.(say, bfzyx)
        output_format = cldnn::format::adjust_to_rank(output_format, output_shapes[0].size());
    }
    if (node.get_preferred_output_fmt() != format::any)
        output_format = node.get_preferred_output_fmt();

    return { layout{output_shapes[0], output_type, output_format, prim->output_paddings[0]} };
}

template std::vector<layout> gemm_inst::calc_output_layouts<ov::PartialShape>(gemm_node const& node, const kernel_impl_params& impl_param);

std::vector<layout> gemm_inst::transform_input_layouts(const std::shared_ptr<const gemm> primitive,
                                                       const std::vector<layout>& input_layouts,
                                                       const bool allow_new_shape_infer) {
    auto get_transposed_input_shape = [&](const ov::PartialShape& input_pshape, size_t input_rank, size_t output_rank, bool transpose, bool first_input) {
        ov::PartialShape transposed_input_pshape;

        if (input_rank == 1) {
            if (input_pshape.is_static()) {
                auto input_shape = input_pshape.to_shape();
                transposed_input_pshape = ov::PartialShape{ static_cast<int64_t>(*std::max_element(input_shape.begin(), input_shape.end())) };
            } else {
                transposed_input_pshape = ov::PartialShape::dynamic(input_rank);
            }
        } else {
            if (input_pshape.is_static()) {
                OPENVINO_ASSERT(input_pshape.size() >= input_rank, "[GPU] Requested input rank in gemm primitive is greater than actual shape");
                std::vector<ov::Dimension> dims(input_pshape.begin(), input_pshape.begin() + input_rank);
                transposed_input_pshape = ov::PartialShape(dims);
            } else {
                transposed_input_pshape = input_pshape;
            }
        }

        if (transposed_input_pshape.size() == 1) {
            first_input ? transposed_input_pshape.insert(transposed_input_pshape.begin(), 1)
                        : transposed_input_pshape.insert(transposed_input_pshape.end(), 1);

            if (transpose) {
                std::swap(transposed_input_pshape[0], transposed_input_pshape[1]);
            }
        }
        size_t ones_to_add = std::max(output_rank, static_cast<size_t>(4)) - transposed_input_pshape.size();
        transposed_input_pshape.insert(transposed_input_pshape.begin(), ones_to_add, 1ul);

        return transposed_input_pshape;
    };

    auto input0_pshape = input_layouts[0].get_partial_shape();
    auto input1_pshape = input_layouts[1].get_partial_shape();

    bool reordered = primitive->input_rank > 4 || primitive->weight_rank > 4;
    size_t output_rank = std::max(primitive->input_rank, primitive->weight_rank);
    // No need to get output_rank for rank>4 inputs when allow_new_shape_infer=true
    size_t input_rank = (reordered && !allow_new_shape_infer) ? output_rank : primitive->input_rank;
    size_t weight_rank = (reordered && !allow_new_shape_infer) ? output_rank : primitive->weight_rank;

    auto transposed_input0_pshape = get_transposed_input_shape(input0_pshape, input_rank, output_rank, primitive->transpose_input0, true);
    auto transposed_input1_pshape = get_transposed_input_shape(input1_pshape, weight_rank, output_rank, primitive->transpose_input1, false);

    std::vector<layout> layouts = input_layouts;
    // Format update for rank > 4 case
    if (layouts[0].format.dimension() < transposed_input0_pshape.size())
        layouts[0].format = cldnn::format::get_default_format(transposed_input0_pshape.size());
    layouts[0].set_partial_shape(transposed_input0_pshape);
    layouts[1].set_partial_shape(transposed_input1_pshape);

    if (primitive->input_size() == 3) {
        auto bias_pshape = input_layouts[2].get_partial_shape();
        auto updated_bias_pshape = get_transposed_input_shape(bias_pshape, weight_rank, output_rank, primitive->transpose_input1, false);
        layouts[2].set_partial_shape(updated_bias_pshape);
    }

    return layouts;
}

layout gemm_inst::transform_output_layout(const std::shared_ptr<const gemm> primitive,
                                          const std::vector<layout>& input_layouts,
                                          const layout& output_layout) {
    auto transpose_pshape = [](const ov::PartialShape& pshape, const std::vector<int64_t>& transpose_order) {
        ov::PartialShape transposed_pshape = pshape;
        auto rank_diff = pshape.size() - transpose_order.size();
        for (size_t i = 0; i < transpose_order.size(); ++i) {
            size_t idx = static_cast<size_t>(transpose_order[i]);
            transposed_pshape[i + rank_diff] = std::move(pshape[idx + rank_diff]);
        }
        return transposed_pshape;
    };

    auto updated_output_layout = output_layout;
    auto output_rank = output_layout.get_partial_shape().size();
    if (output_rank < 4) {
        ov::PartialShape transposed_input0_pshape = transpose_pshape(input_layouts[0].get_partial_shape(), primitive->input0_transpose_order);
        ov::PartialShape transposed_input1_pshape = transpose_pshape(input_layouts[1].get_partial_shape(), primitive->input1_transpose_order);

        auto M = (transposed_input0_pshape.size() > 1) ? transposed_input0_pshape[transposed_input0_pshape.size() - 2]
                                                       : transposed_input0_pshape[0];
        auto N = transposed_input1_pshape[transposed_input1_pshape.size() - 1];

        auto output_pshape = transposed_input0_pshape;
        for (size_t i = 0; i != primitive->input_size(); ++i) {
            auto input_pshape = (i == 0) ? transposed_input0_pshape :
                                (i == 1) ? transposed_input1_pshape :
                                input_layouts[i].get_partial_shape();
            for (size_t j = 0; j != input_pshape.size(); ++j) {
                if (input_pshape[j].get_max_length() != input_pshape[j].get_min_length())
                    ov::Dimension::merge(output_pshape[j], output_pshape[j], input_pshape[j]);
            }
        }

        auto get_spatial_idx = [](cldnn::format format, size_t spatial_idx) {
            const size_t idx = (format::is_grouped(format) ? 3 : 2) + (format.spatial_num() - 1 - spatial_idx);
            return idx;
        };

        output_pshape[get_spatial_idx(updated_output_layout.format, 0)] = std::move(N);
        output_pshape[get_spatial_idx(updated_output_layout.format, 1)] = std::move(M);

        if (primitive->output_transpose_order.size() > 0) {
            output_pshape = transpose_pshape(output_pshape, primitive->output_transpose_order);
        }

        updated_output_layout.set_partial_shape(output_pshape);
    }
    return updated_output_layout;
}

std::string gemm_inst::to_string(gemm_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto alpha = desc->alpha;
    auto beta = desc->beta;
    auto transpose_input0 = desc->transpose_input0 ? " true" : "false";
    auto transpose_input1 = desc->transpose_input1 ? " true" : "false";
    auto indirect_input0 = desc->indirect_a ? " true" : "false";
    auto indirect_input1 = desc->indirect_b ? " true" : "false";
    std::stringstream primitive_description;

    json_composite gemm_info;
    for (size_t i = 0; i < node.get_inputs_count(); i++) {
        gemm_info.add("input_" + std::to_string(i), node.input(i).id());
    }
    gemm_info.add("beam_table", (desc->beam_table.is_valid() ? desc->beam_table.pid : "N/A"));
    gemm_info.add("alpha", alpha);
    gemm_info.add("beta", beta);
    gemm_info.add("trasnpose_input0", transpose_input0);
    gemm_info.add("transpose_input1", transpose_input1);
    gemm_info.add("indirect_input0", indirect_input0);
    gemm_info.add("indirect_input1", indirect_input1);
    gemm_info.add("trasnpose_order_input0", desc->input0_transpose_order);
    gemm_info.add("trasnpose_order_input1", desc->input1_transpose_order);
    gemm_info.add("trasnpose_order_output", desc->output_transpose_order);
    node_info->add("gemm info", gemm_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gemm_inst::typed_primitive_inst(network& network, gemm_node const& node) : parent(network, node) {}
}  // namespace cldnn
