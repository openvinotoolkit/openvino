// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/kv_cache_compressed.hpp"
#include "gather_shape_inference.hpp"
#include "concat_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"

namespace ov::intel_gpu::op {

KVCache::KVCache(const OutputVector& inputs,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 bool indirect,
                 bool trim,
                 int64_t concat_axis,
                 int64_t gather_axis,
                 const ov::element::Type output_type)
    : Op(inputs)
    , m_concat_axis(concat_axis)
    , m_gather_axis(gather_axis)
    , m_indirect(indirect)
    , m_trim(trim)
    , m_output_type(output_type) {
    m_variable = past_variable;
}

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const Output<Node>& beam_idx,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 int64_t gather_axis,
                 const ov::element::Type output_type)
    : KVCache({past, new_token_data, beam_idx}, past_variable, true, false, concat_axis, gather_axis, output_type) {
    if (m_indirect)
        set_output_size(2);
    validate_and_infer_types();
}

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 const ov::element::Type output_type)
    : KVCache({past, new_token_data}, past_variable, false, false, concat_axis, 0, output_type) {
    validate_and_infer_types();
}

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const Output<Node>& past_seq_len,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 const ov::element::Type output_type)
    : KVCache({past, new_token_data, past_seq_len}, past_variable, false, true, concat_axis, 0, output_type) {
    validate_and_infer_types();
}

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const Output<Node>& beam_idx,
                 const Output<Node>& past_seq_len,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 int64_t gather_axis,
                 const ov::element::Type output_type)
    : KVCache({past, new_token_data, beam_idx, past_seq_len}, past_variable, true, true, concat_axis, gather_axis, output_type) {
    if (m_indirect)
        set_output_size(2);
    validate_and_infer_types();
}

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const Output<Node>& past_seq_len,
                 const Output<Node>& dst_idx,
                 const Output<Node>& update_data,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 const ov::element::Type output_type)
    : KVCache({past, new_token_data, past_seq_len, dst_idx, update_data}, past_variable, false, true, concat_axis, 0, output_type) {
    m_update_kv = true;
    validate_and_infer_types();
}

bool KVCache::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("concat_axis", m_concat_axis);
    visitor.on_attribute("gather_axis", m_gather_axis);
    visitor.on_attribute("indirect", m_indirect);
    visitor.on_attribute("trim", m_trim);
    visitor.on_attribute("update_kv", m_update_kv);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void KVCache::validate_and_infer_types() {
    auto output_type = m_output_type;
    if (m_output_type == ov::element::dynamic) {
        output_type = get_input_element_type(0);
    }

    std::vector<ov::PartialShape> input_shapes = {m_variable->get_info().data_shape, get_input_partial_shape(1)};
    if (m_indirect) {
        input_shapes.push_back(get_input_partial_shape(2));
    }

    auto shapes = shape_infer(this, input_shapes);

    size_t out_ports = 0;
    set_output_type(out_ports++, output_type, shapes[0]);

    if (m_indirect) {
        set_output_type(out_ports++, get_input_element_type(2), shapes[1]);
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
        if (m_trim) {
            return std::make_shared<KVCache>(new_args.at(0),
                                             new_args.at(1),
                                             new_args.at(2),
                                             m_variable,
                                             m_concat_axis,
                                             m_output_type);
        } else {
            return std::make_shared<KVCache>(new_args.at(0),
                                             new_args.at(1),
                                             new_args.at(2),
                                             m_variable,
                                             m_concat_axis,
                                             m_gather_axis,
                                             m_output_type);
        }
    } else if (new_args.size() == 4) {
        return std::make_shared<KVCache>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         m_variable,
                                         m_concat_axis,
                                         m_gather_axis,
                                         m_output_type);
    } else if (new_args.size() == 5) {
        return std::make_shared<KVCache>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         new_args.at(4),
                                         m_variable,
                                         m_concat_axis,
                                         m_output_type);
    } else {
        OPENVINO_ASSERT(false);
    }
}

std::vector<ov::PartialShape> shape_infer(const KVCache* op, const std::vector<ov::PartialShape>& input_shapes) {
    std::vector<ov::PartialShape> out_shapes;
    out_shapes.resize(op->get_output_size());

    const auto& gather_axis = op->get_gather_axis();
    const auto& concat_axis = ov::util::normalize(op->get_concat_axis(), input_shapes[0].size());
    const auto trim_length = op->get_trim() ? op->get_trim_length() : 0;
    if (trim_length > 0) {
        OPENVINO_ASSERT(!op->get_indirect(), "Indirect KVCache should not perform trim");
    }
    // We update output shape with input1 shape by default, as input1 is always new, and in some situations, input0 shape
    // has zeros in some dimensions. For example to concat input0 [-1, 0, 0, 0] + input1 [-1, 4, -1, 128] along axis 2,
    // we could (and should) infer dim value of axis 1 and 3 in this case.
    if (op->get_output_size() >= 2) {
        out_shapes[0] = input_shapes[1];
        out_shapes[0][gather_axis] = input_shapes[2][0];
        out_shapes[0][concat_axis] += input_shapes[0][concat_axis] - trim_length;

        std::vector<ov::Dimension> dims(out_shapes[0].size(), 1);
        dims[gather_axis] = out_shapes[0][gather_axis];
        dims[concat_axis] = out_shapes[0][concat_axis];
        out_shapes[1] = dims;
    } else {
        out_shapes[0] = input_shapes[1];
        out_shapes[0][concat_axis] += input_shapes[0][concat_axis] - trim_length;
    }

    return out_shapes;
}

KVCacheCompressed::KVCacheCompressed(const OutputVector& inputs,
                                     const std::shared_ptr<ov::op::util::Variable>& past_variable,
                                     bool trim,
                                     int64_t concat_axis,
                                     int64_t gather_axis,
                                     const QuantizationAttrs& quantization_attrs,
                                     const ov::element::Type output_type)
    : KVCache(inputs, past_variable, true, trim, concat_axis, gather_axis, output_type)
    , m_compressed(true)
    , m_quantization_attrs(quantization_attrs) {
    OPENVINO_ASSERT(quantization_attrs.quantization_dt == ov::element::i8,
                    "[GPU] Only I8 data type is currently supported for KV-cache compression");

    m_variable = past_variable;
    size_t output_size = 3;
    if (quantization_attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
        quantization_attrs.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar)
        output_size++; // add zp output

    set_output_size(output_size);
    validate_and_infer_types();
}

void KVCacheCompressed::validate_and_infer_types() {
    std::vector<ov::PartialShape> input_shapes = {m_variable->get_info().data_shape, get_input_partial_shape(1)};
    input_shapes.push_back(get_input_partial_shape(2));
    const auto compress_input_offset = m_trim ? 4 : 3;
    input_shapes.push_back(get_input_partial_shape(compress_input_offset + 0));

    if (m_quantization_attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
        m_quantization_attrs.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar)
        input_shapes.push_back(get_input_partial_shape(compress_input_offset + 1));

    auto shapes = shape_infer(this, input_shapes);

    size_t out_ports = 0;
    set_output_type(out_ports++, m_quantization_attrs.quantization_dt, shapes[0]);
    set_output_type(out_ports++, get_input_element_type(2), shapes[1]);
    set_output_type(out_ports++, m_quantization_attrs.scale_dt, shapes[2]);

    if (m_quantization_attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
        m_quantization_attrs.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
        set_output_type(out_ports++, m_quantization_attrs.zp_dt, shapes[3]);
    }
}

std::shared_ptr<Node> KVCacheCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<KVCacheCompressed>(new_args,
                                               m_variable,
                                               m_trim,
                                               m_concat_axis,
                                               m_gather_axis,
                                               m_quantization_attrs,
                                               m_output_type);
}

std::vector<ov::PartialShape> shape_infer(const KVCacheCompressed* op,
                                          const std::vector<ov::PartialShape>& input_shapes) {
    std::vector<ov::PartialShape> out_shapes = shape_infer(static_cast<const KVCache*>(op), input_shapes);

    const auto trim_length = op->get_trim() ? op->get_trim_length() : 0;
    if (trim_length > 0) {
        OPENVINO_ASSERT(!op->get_indirect(), "Compressed KVCache should not perform trim");
    }

    if (op->get_output_size() >= 3) {
        ov::op::internal::DynamicQuantize dq_op;
        dq_op.set_attrs(op->get_quantization_attrs());

        auto quantized_data_shapes =
            ov::op::internal::DynamicQuantize::shape_infer(&dq_op, { input_shapes[1] });

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

}  // namespace ov::intel_gpu::op
