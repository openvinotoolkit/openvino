// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gemm_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <utility>
#include <algorithm>

#include "matmul_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gemm)

layout gemm_inst::calc_output_layout(gemm_node const& node, kernel_impl_params const& impl_param) {
    auto prim = impl_param.typed_desc<gemm>();

    auto input0_layout = impl_param.get_input_layout(0);
    auto input1_layout = impl_param.get_input_layout(1);

    auto input0_shape = input0_layout.get_shape();
    auto input1_shape = input1_layout.get_shape();

    bool transpose_input0 = prim->transpose_input0;
    bool transpose_input1 = prim->transpose_input1;

    bool reordered = prim->input_rank > 4 || prim->weight_rank > 4;
    size_t output_rank = std::max(prim->input_rank, prim->weight_rank);
    size_t input_rank = reordered ? output_rank : prim->input_rank;
    size_t weight_rank = reordered ? output_rank : prim->weight_rank;

    auto update_input_shape = [&output_rank](const ov::Shape& input_shape, size_t rank, bool transpose, bool first_input) {
        auto input_shape_update = ov::Shape(input_shape.begin(), input_shape.begin() + std::min(rank, input_shape.size()));
        if (input_shape_update.size() == 1) {
            first_input ? input_shape_update.insert(input_shape_update.begin(), 1)
                        : input_shape_update.insert(input_shape_update.end(), 1);
            if (transpose) {
                std::swap(input_shape_update[0], input_shape_update[1]);
            }
            output_rank = std::max(output_rank, rank + 1);
        }
        input_shape_update.insert(input_shape_update.begin(), output_rank - input_shape_update.size(), 1);
        return input_shape_update;
    };

    auto input0_shape_update = update_input_shape(input0_shape, input_rank, transpose_input0, true);
    auto input1_shape_update = update_input_shape(input1_shape, weight_rank, transpose_input1, false);

    ov::Shape bias_shape(output_rank);
    if (prim->input_size() == 3) {
        bias_shape = impl_param.get_input_layout(2).get_shape();
        bias_shape = update_input_shape(bias_shape, weight_rank, transpose_input1, false);
    }

    auto output_shape = input0_shape_update;
    for (size_t i = 0; i < output_rank; ++i) {
        output_shape[i] = std::max(std::max(input0_shape_update[i], input1_shape_update[i]), bias_shape[i]);
    }

    size_t M = !transpose_input0 ? *(input0_shape_update.end() - 2) : input0_shape_update.back();
    size_t N = !transpose_input1 ? input1_shape_update.back() : *(input1_shape_update.end() - 2);

    output_shape[output_rank - 2] = M;
    output_shape[output_rank - 1] = N;

    size_t ones_to_add = 4 - std::min(output_shape.size(), static_cast<size_t>(4));
    output_shape.insert(output_shape.begin(), ones_to_add, 1);

    auto output_type = input0_layout.data_type;
    if ((output_type == data_types::u8 || output_type == data_types::i8) && prim->output_data_types[0])
        output_type = *prim->output_data_types[0];

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    auto output_format = input0_layout.format;

    return layout(output_shape, output_type, output_format, prim->output_paddings[0]);
}

template<typename ShapeType>
std::vector<layout> gemm_inst::calc_output_layouts(gemm_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto prim = impl_param.typed_desc<gemm>();
    auto input0_layout = impl_param.get_input_layout(0);
    auto input1_layout = impl_param.get_input_layout(1);

    auto default_out_dt = data_type_traits::is_floating_point(input0_layout.data_type) ? input0_layout.data_type : data_types::f32;
    auto output_type = prim->output_data_types[0].value_or(default_out_dt);

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    ov::op::v0::MatMul op;
    op.set_transpose_a(prim->transpose_input0);
    op.set_transpose_b(prim->transpose_input1);

    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        input1_layout.get<ShapeType>()
    };

    ov::op::v0::shape_infer(&op, input_shapes, output_shapes);

    return { layout{output_shapes[0], output_type, input0_layout.format, prim->output_paddings[0]} };
}

template std::vector<layout> gemm_inst::calc_output_layouts<ov::PartialShape>(gemm_node const& node, const kernel_impl_params& impl_param);

std::vector<size_t> gemm_inst::extend_input_shape_to_6d(kernel_impl_params const& orig_impl_param, int32_t input_idx) {
    ov::PartialShape ps = orig_impl_param.get_input_layout(input_idx).get_partial_shape();

    if (ps.size() < 4) {
        ps.insert(ps.begin(), 4 - ps.size(), ov::Dimension(1));
    }

    layout l(ps, data_types::i32, format::get_default_format(ps.size()));
    return l.transform(format::bfwzyx).to_shape();
}

std::vector<size_t> gemm_inst::extend_output_shape_to_6d(kernel_impl_params const& orig_impl_param, int32_t output_idx) {
    ov::PartialShape ps = orig_impl_param.get_output_layout(output_idx).get_partial_shape();

    if (ps.size() < 4) {
        ps.insert(ps.begin(), 4 - ps.size(), ov::Dimension(1));
    }

    layout l(ps, data_types::i32, format::get_default_format(ps.size()));
    return l.transform(format::bfwzyx).to_shape();
}

std::vector<layout> gemm_inst::transform_input_layouts(const std::shared_ptr<const gemm> primitive,
                                                       const std::vector<layout>& input_layouts,
                                                       const layout& output_layout) {
    auto get_updated_input_shape = [&](const ov::PartialShape& input_pshape, size_t input_rank, size_t output_rank, bool transpose, bool first_input) {
        ov::PartialShape updated_input_pshape;

        if (input_rank == 1) {
            if (input_pshape.is_static()) {
                auto input_shape = input_pshape.to_shape();
                updated_input_pshape = ov::PartialShape{ static_cast<int64_t>(*std::max_element(input_shape.begin(), input_shape.end())) };
            } else {
                updated_input_pshape = ov::PartialShape::dynamic(input_rank);
            }
        } else {
            if (input_pshape.is_static()) {
                OPENVINO_ASSERT(input_pshape.size() >= input_rank, "[GPU] Requested input rank in gemm primitive is greater than actual shape");
                std::vector<ov::Dimension> dims(input_pshape.begin(), input_pshape.begin() + input_rank);
                updated_input_pshape = ov::PartialShape(dims);
            } else {
                updated_input_pshape = input_pshape;
            }
        }

        if (updated_input_pshape.size() == 1) {
            first_input ? updated_input_pshape.insert(updated_input_pshape.begin(), 1)
                        : updated_input_pshape.insert(updated_input_pshape.end(), 1);

            if (transpose) {
                std::swap(updated_input_pshape[0], updated_input_pshape[1]);
            }
        }
        size_t ones_to_add = std::max(output_rank, static_cast<size_t>(4)) - updated_input_pshape.size();
        updated_input_pshape.insert(updated_input_pshape.begin(), ones_to_add, 1ul);

        return updated_input_pshape;
    };

    auto input0_pshape = input_layouts[0].get_partial_shape();
    auto input1_pshape = input_layouts[1].get_partial_shape();

    bool reordered = primitive->input_rank > 4 || primitive->weight_rank > 4;
    size_t output_rank = std::max(primitive->input_rank, primitive->weight_rank);
    size_t input_rank = reordered ? output_rank : primitive->input_rank;
    size_t weight_rank = reordered ? output_rank : primitive->weight_rank;

    auto updated_input0_pshape = get_updated_input_shape(input0_pshape, input_rank, output_rank, primitive->transpose_input0, true);
    auto updated_input1_pshape = get_updated_input_shape(input1_pshape, weight_rank, output_rank, primitive->transpose_input1, false);

    std::vector<layout> layouts = input_layouts;
    layouts[0].set_partial_shape(updated_input0_pshape);
    layouts[1].set_partial_shape(updated_input1_pshape);

    if (primitive->input_size() == 3) {
        auto bias_pshape = input_layouts[2].get_partial_shape();
        auto updated_bias_pshape = get_updated_input_shape(bias_pshape, weight_rank, output_rank, primitive->transpose_input1, false);
        layouts[2].set_partial_shape(updated_bias_pshape);
    }

    return layouts;
}

layout gemm_inst::transform_output_layout(const std::shared_ptr<const gemm> primitive,
                                          const std::vector<layout>& input_layouts,
                                          const layout& output_layout) {
    auto updated_output_layout = output_layout;
    auto output_rank = output_layout.get_partial_shape().size();
    if (output_rank < 4) {
        auto input0_pshape = input_layouts[0].get_partial_shape();
        auto input1_pshape = input_layouts[1].get_partial_shape();

        auto M = !primitive->transpose_input0 ? input0_pshape[input0_pshape.size() - 2] : input0_pshape[input0_pshape.size() - 1];
        auto N = !primitive->transpose_input1 ? input1_pshape[input1_pshape.size() - 1] : input1_pshape[input1_pshape.size() - 2];

        auto output_pshape = input_layouts[0].get_partial_shape();
        for (size_t i = 0; i != input_layouts.size(); ++i) {
            auto input_pshape = input_layouts[i].get_partial_shape();
            for (size_t j = 0; j != input_pshape.size(); ++j) {
                ov::Dimension::merge(output_pshape[j], output_pshape[j], input_pshape[j]);
            }
        }

        auto get_spatial_idx = [](cldnn::format format, size_t spatial_idx) {
            const size_t idx = (format::is_grouped(format) ? 3 : 2) + (format.spatial_num() - 1 - spatial_idx);
            return idx;
        };

        output_pshape[get_spatial_idx(updated_output_layout.format, 0)] = N;
        output_pshape[get_spatial_idx(updated_output_layout.format, 1)] = M;
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
    std::stringstream primitive_description;

    json_composite gemm_info;
    for (size_t i = 0; i < node.inputs_count(); i++) {
        gemm_info.add("input_" + std::to_string(i), node.input(i).id());
    }
    gemm_info.add("alpha", alpha);
    gemm_info.add("beta", beta);
    gemm_info.add("trasnpose_input0", transpose_input0);
    gemm_info.add("transpose_input1", transpose_input1);
    node_info->add("gemm info", gemm_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gemm_inst::typed_primitive_inst(network& network, gemm_node const& node) : parent(network, node) {}
}  // namespace cldnn
