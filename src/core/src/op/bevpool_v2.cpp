// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bevpool_v2.hpp"

#include <algorithm>
#include <cmath>

#include "itt.hpp"
#include "openvino/core/type/float16.hpp"

namespace {
std::vector<int64_t> to_i64_vector(const ov::Tensor& t) {
    const auto count = ov::shape_size(t.get_shape());
    std::vector<int64_t> values(count);

    switch (t.get_element_type()) {
    case ov::element::Type_t::i32: {
        const auto* src = t.data<const int32_t>();
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<int64_t>(src[i]);
        }
        break;
    }
    case ov::element::Type_t::i64: {
        const auto* src = t.data<const int64_t>();
        std::copy(src, src + count, values.begin());
        break;
    }
    case ov::element::Type_t::u32: {
        const auto* src = t.data<const uint32_t>();
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<int64_t>(src[i]);
        }
        break;
    }
    case ov::element::Type_t::u64: {
        const auto* src = t.data<const uint64_t>();
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<int64_t>(src[i]);
        }
        break;
    }
    default:
        values.clear();
        break;
    }

    return values;
}

template <typename T>
bool evaluate_bevpool_v2(const ov::Tensor& cf,
                        const ov::Tensor& dw,
                        const ov::Tensor& idx,
                        const ov::Tensor& itv,
                        ov::Tensor& output,
                        uint32_t input_channels,
                        uint32_t output_channels,
                        uint32_t image_width,
                        uint32_t image_height,
                        uint32_t feature_width,
                        uint32_t feature_height,
                        const ov::op::v15::Bound& d_bound) {
    const auto& cf_shape = cf.get_shape();
    if (cf_shape.size() != 4 || cf_shape[0] == 0) {
        return false;
    }

    const auto idx_values = to_i64_vector(idx);
    const auto itv_values = to_i64_vector(itv);
    if (itv_values.size() % 3 != 0) {
        return false;
    }

    const auto dw_len = static_cast<int64_t>(ov::shape_size(dw.get_shape()));
    const auto cf_len = static_cast<int64_t>(ov::shape_size(cf_shape));
    const auto out_len = static_cast<int64_t>(ov::shape_size(output.get_shape()));

    const auto feature_area = static_cast<int64_t>(image_width) * static_cast<int64_t>(image_height);
    const auto depth_bins = static_cast<int64_t>((d_bound.max - d_bound.min) / d_bound.step);
    const auto depth_span = depth_bins * feature_area;
    const auto out_plane = static_cast<int64_t>(feature_width) * static_cast<int64_t>(feature_height);
    if (feature_area <= 0 || depth_bins <= 0 || depth_span <= 0 || out_plane <= 0) {
        return false;
    }

    auto* out_data = output.data<T>();
    std::fill(out_data, out_data + out_len, T{0});

    const auto* cf_data = cf.data<const T>();
    const auto* dw_data = dw.data<const T>();

    const auto interval_count = itv_values.size() / 3;
    for (size_t interval = 0; interval < interval_count; ++interval) {
        const auto start = itv_values[interval * 3 + 0];
        const auto end = itv_values[interval * 3 + 1];
        const auto bev_base = itv_values[interval * 3 + 2];
        if (start < 0 || end < start || static_cast<size_t>(end) > idx_values.size()) {
            continue;
        }

        for (uint32_t c = 0; c < output_channels; ++c) {
            float acc = 0.f;
            for (int64_t i = start; i < end; ++i) {
                const auto dw_index = idx_values[static_cast<size_t>(i)];
                if (dw_index < 0 || dw_index >= dw_len) {
                    continue;
                }

                const auto camera_idx = dw_index / depth_span;
                const auto feature_idx = dw_index % feature_area;
                const auto cf_offset = (camera_idx * feature_area + feature_idx) * static_cast<int64_t>(input_channels) +
                                       static_cast<int64_t>(c);
                if (cf_offset < 0 || cf_offset >= cf_len) {
                    continue;
                }

                acc += static_cast<float>(cf_data[cf_offset]) * static_cast<float>(dw_data[dw_index]);
            }

            const auto out_index = bev_base + static_cast<int64_t>(c) * out_plane;
            if (out_index >= 0 && out_index < out_len) {
                out_data[out_index] = static_cast<T>(acc);
            }
        }
    }

    return true;
}
}  // namespace

namespace ov {
namespace op {
namespace v15 {

BevPoolV2::BevPoolV2(const OutputVector& inputs,
                     uint32_t input_channels,
                     uint32_t output_channels,
                     uint32_t image_width,
                     uint32_t image_height,
                     uint32_t feature_width,
                     uint32_t feature_height,
                     const Bound& x_bound,
                     const Bound& y_bound,
                     const Bound& z_bound,
                     const Bound& d_bound)
    : Op(inputs),
      m_input_channels(input_channels),
      m_output_channels(output_channels),
      m_image_width(image_width),
      m_image_height(image_height),
      m_feature_width(feature_width),
      m_feature_height(feature_height),
      m_x_bound(x_bound),
      m_y_bound(y_bound),
      m_z_bound(z_bound),
      m_d_bound(d_bound) {
    constructor_validate_and_infer_types();
}

bool BevPoolV2::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("input_channels", m_input_channels);
    visitor.on_attribute("output_channels", m_output_channels);
    visitor.on_attribute("image_width", m_image_width);
    visitor.on_attribute("image_height", m_image_height);
    visitor.on_attribute("feature_width", m_feature_width);
    visitor.on_attribute("feature_height", m_feature_height);

    visitor.on_attribute("x_bound_min", m_x_bound.min);
    visitor.on_attribute("x_bound_max", m_x_bound.max);
    visitor.on_attribute("x_bound_step", m_x_bound.step);

    visitor.on_attribute("y_bound_min", m_y_bound.min);
    visitor.on_attribute("y_bound_max", m_y_bound.max);
    visitor.on_attribute("y_bound_step", m_y_bound.step);

    visitor.on_attribute("z_bound_min", m_z_bound.min);
    visitor.on_attribute("z_bound_max", m_z_bound.max);
    visitor.on_attribute("z_bound_step", m_z_bound.step);

    visitor.on_attribute("d_bound_min", m_d_bound.min);
    visitor.on_attribute("d_bound_max", m_d_bound.max);
    visitor.on_attribute("d_bound_step", m_d_bound.step);

    return true;
}

void BevPoolV2::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 4, "BevPoolV2 expects exactly 4 inputs: cf, dw, idx, itv.");

    NODE_VALIDATION_CHECK(this, m_input_channels > 0, "input_channels must be greater than zero.");
    NODE_VALIDATION_CHECK(this, m_output_channels > 0, "output_channels must be greater than zero.");
    NODE_VALIDATION_CHECK(this, m_image_width > 0, "image_width must be greater than zero.");
    NODE_VALIDATION_CHECK(this, m_image_height > 0, "image_height must be greater than zero.");
    NODE_VALIDATION_CHECK(this, m_feature_width > 0, "feature_width must be greater than zero.");
    NODE_VALIDATION_CHECK(this, m_feature_height > 0, "feature_height must be greater than zero.");

    NODE_VALIDATION_CHECK(this, m_x_bound.step > 0.f, "x_bound_step must be greater than zero.");
    NODE_VALIDATION_CHECK(this, m_y_bound.step > 0.f, "y_bound_step must be greater than zero.");
    NODE_VALIDATION_CHECK(this, m_z_bound.step > 0.f, "z_bound_step must be greater than zero.");
    NODE_VALIDATION_CHECK(this, m_d_bound.step > 0.f, "d_bound_step must be greater than zero.");
    NODE_VALIDATION_CHECK(this, m_x_bound.max >= m_x_bound.min, "x_bound_max must be greater than or equal to x_bound_min.");
    NODE_VALIDATION_CHECK(this, m_y_bound.max >= m_y_bound.min, "y_bound_max must be greater than or equal to y_bound_min.");
    NODE_VALIDATION_CHECK(this, m_z_bound.max >= m_z_bound.min, "z_bound_max must be greater than or equal to z_bound_min.");
    NODE_VALIDATION_CHECK(this, m_d_bound.max >= m_d_bound.min, "d_bound_max must be greater than or equal to d_bound_min.");

    const auto cf_et = get_input_element_type(0);
    const auto dw_et = get_input_element_type(1);
    const auto idx_et = get_input_element_type(2);
    const auto itv_et = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          cf_et.is_real(),
                          "Input 0 (cf) must be a floating-point tensor. Got: ",
                          cf_et);
    NODE_VALIDATION_CHECK(this,
                          dw_et.is_real(),
                          "Input 1 (dw) must be a floating-point tensor. Got: ",
                          dw_et);
    NODE_VALIDATION_CHECK(this,
                          cf_et.compatible(dw_et),
                          "Input 0 (cf) and input 1 (dw) element types must be compatible. Got: ",
                          cf_et,
                          " and ",
                          dw_et);
    NODE_VALIDATION_CHECK(this,
                          idx_et.is_integral_number(),
                          "Input 2 (idx) must be an integer tensor. Got: ",
                          idx_et);
    NODE_VALIDATION_CHECK(this,
                          itv_et.is_integral_number(),
                          "Input 3 (itv) must be an integer tensor. Got: ",
                          itv_et);

    const auto cf_ps = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
                          cf_ps.rank().compatible(4),
                          "Input 0 (cf) rank must be compatible with 4 [N, H, W, C]. Got: ",
                          cf_ps);

    if (cf_ps.rank().is_static()) {
        const auto height_dim = cf_ps[1];
        const auto width_dim = cf_ps[2];
        const auto channels_dim = cf_ps[3];

        if (channels_dim.is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  channels_dim.get_length() == static_cast<int64_t>(m_input_channels),
                                  "Input 0 (cf) channel dimension does not match input_channels attribute. Got: ",
                                  channels_dim,
                                  ", expected: ",
                                  m_input_channels);
        }

        if (height_dim.is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  height_dim.get_length() == static_cast<int64_t>(m_image_height),
                                  "Input 0 (cf) height does not match image_height attribute. Got: ",
                                  height_dim,
                                  ", expected: ",
                                  m_image_height);
        }

        if (width_dim.is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  width_dim.get_length() == static_cast<int64_t>(m_image_width),
                                  "Input 0 (cf) width does not match image_width attribute. Got: ",
                                  width_dim,
                                  ", expected: ",
                                  m_image_width);
        }
    }

    const auto itv_ps = get_input_partial_shape(3);
    if (itv_ps.rank().is_static()) {
        const auto itv_rank = itv_ps.rank().get_length();
        NODE_VALIDATION_CHECK(this,
                              itv_rank == 1 || itv_rank == 2,
                              "Input 3 (itv) must be 1D ([3 * intervals]) or 2D ([intervals, 3]). Got rank: ",
                              itv_rank);

        if (itv_rank == 2) {
            NODE_VALIDATION_CHECK(this,
                                  itv_ps[1].compatible(3),
                                  "Input 3 (itv) second dimension must be 3 for [start, end, bev_offset]. Got: ",
                                  itv_ps);
        } else if (itv_ps[0].is_static()) {
            const auto v = itv_ps[0].get_length();
            NODE_VALIDATION_CHECK(this,
                                  v % 3 == 0,
                                  "Input 3 (itv) 1D length must be divisible by 3. Got: ",
                                  v);
        }
    }

    const auto output_et = cf_et;

    Dimension batch = Dimension::dynamic();
    if (cf_ps.rank().is_static() && cf_ps.rank().get_length() >= 1) {
        batch = cf_ps[0];
    }

    PartialShape output_ps{batch,
                           static_cast<int64_t>(m_output_channels),
                           static_cast<int64_t>(m_feature_height),
                           static_cast<int64_t>(m_feature_width)};
    set_output_type(0, output_et, output_ps);
}

std::shared_ptr<Node> BevPoolV2::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<BevPoolV2>(new_args,
                                       m_input_channels,
                                       m_output_channels,
                                       m_image_width,
                                       m_image_height,
                                       m_feature_width,
                                       m_feature_height,
                                       m_x_bound,
                                       m_y_bound,
                                       m_z_bound,
                                       m_d_bound);
}

bool BevPoolV2::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v15_BevPoolV2_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 4);

    const auto& cf = inputs[0];
    const auto batch = cf.get_shape().at(0);
    outputs[0].set_shape({batch,
                          static_cast<size_t>(m_output_channels),
                          static_cast<size_t>(m_feature_height),
                          static_cast<size_t>(m_feature_width)});

    switch (cf.get_element_type()) {
    case element::Type_t::f32:
        return evaluate_bevpool_v2<float>(inputs[0],
                                          inputs[1],
                                          inputs[2],
                                          inputs[3],
                                          outputs[0],
                                          m_input_channels,
                                          m_output_channels,
                                          m_image_width,
                                          m_image_height,
                                          m_feature_width,
                                          m_feature_height,
                                          m_d_bound);
    case element::Type_t::f16:
        return evaluate_bevpool_v2<ov::float16>(inputs[0],
                                                inputs[1],
                                                inputs[2],
                                                inputs[3],
                                                outputs[0],
                                                m_input_channels,
                                                m_output_channels,
                                                m_image_width,
                                                m_image_height,
                                                m_feature_width,
                                                m_feature_height,
                                                m_d_bound);
    default:
        return false;
    }
}

bool BevPoolV2::has_evaluate() const {
    OV_OP_SCOPE(v15_BevPoolV2_has_evaluate);

    const auto cf_et = get_input_element_type(0);
    const auto dw_et = get_input_element_type(1);
    const auto idx_et = get_input_element_type(2);
    const auto itv_et = get_input_element_type(3);

    const bool fp_supported = (cf_et == element::f16 || cf_et == element::f32);
    return fp_supported && cf_et.compatible(dw_et) && idx_et.is_integral_number() && itv_et.is_integral_number();
}

uint32_t BevPoolV2::get_input_channels() const {
    return m_input_channels;
}

uint32_t BevPoolV2::get_output_channels() const {
    return m_output_channels;
}

uint32_t BevPoolV2::get_image_width() const {
    return m_image_width;
}

uint32_t BevPoolV2::get_image_height() const {
    return m_image_height;
}

uint32_t BevPoolV2::get_feature_width() const {
    return m_feature_width;
}

uint32_t BevPoolV2::get_feature_height() const {
    return m_feature_height;
}

const Bound& BevPoolV2::get_x_bound() const {
    return m_x_bound;
}

const Bound& BevPoolV2::get_y_bound() const {
    return m_y_bound;
}

const Bound& BevPoolV2::get_z_bound() const {
    return m_z_bound;
}

const Bound& BevPoolV2::get_d_bound() const {
    return m_d_bound;
}

void BevPoolV2::set_input_channels(uint32_t value) {
    m_input_channels = value;
}

void BevPoolV2::set_output_channels(uint32_t value) {
    m_output_channels = value;
}

void BevPoolV2::set_image_width(uint32_t value) {
    m_image_width = value;
}

void BevPoolV2::set_image_height(uint32_t value) {
    m_image_height = value;
}

void BevPoolV2::set_feature_width(uint32_t value) {
    m_feature_width = value;
}

void BevPoolV2::set_feature_height(uint32_t value) {
    m_feature_height = value;
}

void BevPoolV2::set_x_bound(const Bound& value) {
    m_x_bound = value;
}

void BevPoolV2::set_y_bound(const Bound& value) {
    m_y_bound = value;
}

void BevPoolV2::set_z_bound(const Bound& value) {
    m_z_bound = value;
}

void BevPoolV2::set_d_bound(const Bound& value) {
    m_d_bound = value;
}

}  // namespace v15
}  // namespace op
}  // namespace ov
