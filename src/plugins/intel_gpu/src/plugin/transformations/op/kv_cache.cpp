// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/kv_cache.hpp"
#include "gather_shape_inference.hpp"
#include "concat_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const Output<Node>& beam_idx,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 int64_t gather_axis,
                 const ov::element::Type output_type)
    : Op({past, new_token_data, beam_idx})
    , m_concat_axis(concat_axis)
    , m_gather_axis(gather_axis)
    , m_indirect(true)
    , m_compressed(false)
    , m_output_type(output_type) {
    m_variable = past_variable;
    if (m_indirect)
        set_output_size(2);
    validate_and_infer_types();
}

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 const ov::element::Type output_type)
    : Op({past, new_token_data})
    , m_concat_axis(concat_axis)
    , m_gather_axis(0)
    , m_indirect(false)
    , m_compressed(false)
    , m_output_type(output_type) {
    m_variable = past_variable;
    validate_and_infer_types();
}

KVCache::KVCache(const OutputVector& inputs,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 int64_t gather_axis,
                 bool combine_scales_and_zp,
                 const QuantizationConfig& config,
                 const std::vector<uint64_t>& scales_zp_output_order,
                 const ov::element::Type output_type)
    : Op(inputs)
    , m_concat_axis(concat_axis)
    , m_gather_axis(gather_axis)
    , m_indirect(true)
    , m_compressed(true)
    , m_combine_scales_and_zp(combine_scales_and_zp)
    , m_quantization_config(config)
    , m_scales_zp_output_order(scales_zp_output_order)
    , m_output_type(output_type) {
    OPENVINO_ASSERT(m_quantization_config.quantization_dt == ov::element::i8,
                    "[GPU] Only I8 data type is currently supported for KV-cache compression");

    m_variable = past_variable;
    size_t output_size = 3;
    if (config.is_asymmetric_quantization() && !combine_scales_and_zp)
        output_size++; // add zp output

    set_output_size(output_size);
    validate_and_infer_types();
}

bool KVCache::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("concat_axis", m_concat_axis);
    visitor.on_attribute("gather_axis", m_gather_axis);
    visitor.on_attribute("indirect", m_indirect);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("compressed", m_compressed);
    return true;
}

void KVCache::validate_and_infer_types() {
    auto output_type = m_output_type;
    if (m_compressed) {
        output_type = m_quantization_config.quantization_dt;
    } else if (m_output_type == ov::element::undefined) {
        output_type = get_input_element_type(0);
    }

    std::vector<ov::PartialShape> input_shapes = {m_variable->get_info().data_shape, get_input_partial_shape(1)};
    if (m_indirect) {
        input_shapes.push_back(get_input_partial_shape(2));
    }

    if (m_compressed) {
        input_shapes.push_back(get_input_partial_shape(3));

        if (m_quantization_config.is_asymmetric_quantization() && !m_combine_scales_and_zp)
            input_shapes.push_back(get_input_partial_shape(4));
    }

    auto shapes = m_compressed ? shape_infer(this, input_shapes, m_quantization_config, m_scales_zp_output_order, m_combine_scales_and_zp)
                               : shape_infer(this, input_shapes);

    size_t out_ports = 0;
    set_output_type(out_ports++, output_type, shapes[0]);

    if (m_indirect) {
        set_output_type(out_ports++, get_input_element_type(2), shapes[1]);
    }

    if (m_compressed) {
        set_output_type(out_ports++, m_quantization_config.scale_dt, shapes[2]);

        if (m_quantization_config.is_asymmetric_quantization() && !m_combine_scales_and_zp) {
            set_output_type(out_ports++, m_quantization_config.zp_dt, shapes[3]);
        }
    }
}

std::shared_ptr<Node> KVCache::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return std::make_shared<KVCache>(new_args.at(0),
                                         new_args.at(1),
                                         m_variable,
                                         m_concat_axis,
                                         m_output_type);

    } else if (new_args.size() == 3) {
        return std::make_shared<KVCache>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         m_variable,
                                         m_concat_axis,
                                         m_gather_axis,
                                         m_output_type);
    } else {
        return std::make_shared<KVCache>(new_args,
                                         m_variable,
                                         m_concat_axis,
                                         m_gather_axis,
                                         m_combine_scales_and_zp,
                                         m_quantization_config,
                                         m_scales_zp_output_order,
                                         m_output_type);
    }
}

std::vector<ov::PartialShape> shape_infer(const KVCache* op,
                                          const std::vector<ov::PartialShape>& input_shapes,
                                          const ov::op::internal::QuantizationConfig& config,
                                          const std::vector<uint64_t>& scales_zp_output_order,
                                          bool combine_scales_and_zp) {
    std::vector<ov::PartialShape> out_shapes = shape_infer(op, input_shapes);

    if (op->get_output_size() >= 3) {
        ov::intel_gpu::op::DynamicQuantize op;
        auto quantized_data_shapes =
            ov::intel_gpu::op::DynamicQuantize::shape_infer(&op, { input_shapes[1] }, config, scales_zp_output_order, combine_scales_and_zp);

        const auto scales_concat_axis = 2;
        ov::PartialShape compression_scale_shape = input_shapes[3];
        compression_scale_shape[scales_concat_axis] += quantized_data_shapes[1][scales_concat_axis];
        out_shapes[2] = compression_scale_shape;

        // add zp output
        if (quantized_data_shapes.size() == 3) {
            ov::PartialShape compression_zp_shape = input_shapes[4];
            compression_zp_shape[scales_concat_axis] += quantized_data_shapes[2][scales_concat_axis];
            out_shapes[3] = compression_zp_shape;
        }
    }

    return out_shapes;
}

std::vector<ov::PartialShape> shape_infer(const KVCache* op, const std::vector<ov::PartialShape>& input_shapes) {
    std::vector<ov::PartialShape> out_shapes;
    out_shapes.resize(op->get_output_size());

    const auto& gather_axis = op->get_gather_axis();
    const auto& concat_axis = ov::util::normalize(op->get_concat_axis(), input_shapes[0].size());
    if (op->get_output_size() >= 2) {
        out_shapes[0] = input_shapes[0];
        out_shapes[0][gather_axis] = input_shapes[2][0];
        out_shapes[0][concat_axis] += input_shapes[1][concat_axis];

        std::vector<ov::Dimension> dims(out_shapes[0].size(), 1);
        dims[gather_axis] = out_shapes[0][gather_axis];
        dims[concat_axis] = out_shapes[0][concat_axis];
        out_shapes[1] = dims;
    } else {
        out_shapes[0] = input_shapes[0];
        out_shapes[0][concat_axis] += input_shapes[1][concat_axis];
    }

    return out_shapes;
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
