// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_residual.hpp"

#include <array>
#include <limits>

#include "gated_residual_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/util/container_util.hpp"
#include "openvino/util/math_util.hpp"

namespace ov::op::internal {

// Named (not anonymous) namespace: helper names must not collide with other ops in unity builds.
namespace gated_residual_detail {

constexpr GatedResidual::ReadLayout make_read_layout(bool compressed, bool has_zp, bool write_gate) {
    GatedResidual::ReadLayout l;
    l.compressed = compressed;
    l.has_zp = compressed && has_zp;
    l.write_gate = write_gate;
    const size_t base = compressed ? (l.has_zp ? 8 : 6) : 4;
    l.w_inject = base;
    l.scale_inject = base + 1;
    l.zp_inject = base + 2;
    l.input_count = base + (write_gate ? (compressed ? (l.has_zp ? 3 : 2) : 1) : 0);
    return l;
}

/// The form selected by the input count; empty when no form has that many inputs.
static std::optional<GatedResidual::ReadLayout> read_layout(size_t input_count, bool write_gate) {
    for (const bool compressed : {false, true}) {
        for (const bool has_zp : {false, true}) {
            const auto l = make_read_layout(compressed, has_zp, write_gate);
            if (l.has_zp == has_zp && l.input_count == input_count) {
                return l;
            }
        }
    }
    return std::nullopt;
}

static bool is_compressed_weight_type(const ov::element::Type& t) {
    static const auto compressed_types =
        ov::util::make_array(ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8);
    return ov::util::contains(compressed_types, t);
}

}  // namespace gated_residual_detail

std::optional<GatedResidual::ReadLayout> GatedResidual::get_read_layout() const {
    return gated_residual_detail::read_layout(get_input_size(), m_config.produce_write_gate);
}

GatedResidual::GatedResidual(const ov::OutputVector& args, const Config& config) : Op(args), m_config(config) {
    OV_OP_SCOPE(internal_GatedResidual_GatedResidual);
    constructor_validate_and_infer_types();
}

const GatedResidual::Config& GatedResidual::get_config() const {
    return m_config;
}

bool GatedResidual::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_GatedResidual_visit_attributes);
    visitor.on_attribute("mode", m_config.mode);
    visitor.on_attribute("hc_count", m_config.hc_count);
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.on_attribute("lowrank", m_config.lowrank);
    visitor.on_attribute("epsilon", m_config.epsilon);
    visitor.on_attribute("produce_write_gate", m_config.produce_write_gate);
    return true;
}

void GatedResidual::validate_and_infer_types() {
    OV_OP_SCOPE(internal_GatedResidual_validate_and_infer_types);
    const auto input_size = get_input_size();
    const auto out_et = get_input_element_type(0);

    // Every branch below tests only against WRITE, so an unnamed value must not slip through as READ.
    NODE_VALIDATION_CHECK(this,
                          ov::util::contains(std::array{Mode::READ, Mode::WRITE, Mode::FINAL_MIX}, m_config.mode),
                          "GatedResidual got an unknown mode: ",
                          static_cast<int>(m_config.mode));

    NODE_VALIDATION_CHECK(this, m_config.hc_count > 0, "GatedResidual requires a positive branch count C.");
    NODE_VALIDATION_CHECK(this, m_config.hidden_size > 0, "GatedResidual requires a positive hidden size D.");
    // C, D, L and C*D must fit in int64, the dimension type of the shape system.
    constexpr auto max_dim = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    NODE_VALIDATION_CHECK(
        this,
        m_config.hidden_size <= max_dim && m_config.lowrank <= max_dim && m_config.hc_count <= max_dim,
        "GatedResidual C, D and L must fit in int64.");
    size_t cd = 0;
    NODE_VALIDATION_CHECK(this,
                          !ov::util::mul_overflow(m_config.hc_count, m_config.hidden_size, cd) && cd <= max_dim,
                          "GatedResidual C*D overflows: C = ",
                          m_config.hc_count,
                          ", D = ",
                          m_config.hidden_size);
    NODE_VALIDATION_CHECK(this,
                          out_et.is_dynamic() || out_et.is_real(),
                          "GatedResidual residual must have a floating-point element type, got ",
                          out_et);

    if (m_config.mode == Mode::WRITE) {
        NODE_VALIDATION_CHECK(this, input_size == 3, "GatedResidual (WRITE) expects 3 inputs, got ", input_size);
        auto merged_et = out_et;
        for (size_t i = 1; i < input_size; ++i) {
            NODE_VALIDATION_CHECK(this,
                                  ov::element::Type::merge(merged_et, merged_et, get_input_element_type(i)),
                                  "GatedResidual (WRITE) inputs must share the residual element type.");
        }
        const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
        set_output_type(0, merged_et, output_shapes[0]);
        return;
    }

    // READ / FINAL_MIX
    const auto layout = gated_residual_detail::read_layout(input_size, m_config.produce_write_gate);
    NODE_VALIDATION_CHECK(this,
                          layout.has_value(),
                          "GatedResidual (READ/FINAL_MIX) expects 4 inputs (5 with W_inject), 6 / 8 with compressed "
                          "weights (without / with zero "
                          "points), or 8 / 11 with compressed weights and W_inject, got ",
                          input_size);
    const bool compressed = layout->compressed;
    NODE_VALIDATION_CHECK(this,
                          m_config.mode != Mode::FINAL_MIX || !m_config.produce_write_gate,
                          "GatedResidual (FINAL_MIX) must not produce a write gate.");
    NODE_VALIDATION_CHECK(this, m_config.lowrank > 0, "GatedResidual (READ/FINAL_MIX) requires a positive low rank L.");

    const auto wd_et = get_input_element_type(layout->w_down);
    const auto wu_et = get_input_element_type(layout->w_up);
    NODE_VALIDATION_CHECK(this,
                          wd_et.is_dynamic() || wu_et.is_dynamic() || wd_et == wu_et,
                          "GatedResidual W_down and W_up must share one element type, got ",
                          wd_et,
                          " and ",
                          wu_et);
    const bool wd_quant = gated_residual_detail::is_compressed_weight_type(wd_et);
    NODE_VALIDATION_CHECK(this,
                          wd_et.is_dynamic() || (compressed == wd_quant),
                          "GatedResidual takes i4/u4/i8/u8 weights exactly with the compressed input forms.");
    if (compressed) {
        const auto check_types = [&](size_t scale_idx, size_t zp_idx, const char* name) {
            NODE_VALIDATION_CHECK(
                this,
                get_input_element_type(scale_idx).is_dynamic() || get_input_element_type(scale_idx).is_real(),
                "GatedResidual ",
                name,
                " scale must be a floating-point type (f16, bf16 or f32), got ",
                get_input_element_type(scale_idx));
            if (layout->has_zp) {
                NODE_VALIDATION_CHECK(
                    this,
                    get_input_element_type(zp_idx).is_dynamic() || get_input_element_type(zp_idx) == wd_et,
                    "GatedResidual ",
                    name,
                    " zero point must have the weight element type ",
                    wd_et);
            }
        };
        check_types(layout->scale_down, layout->zp_down, "W_down");
        check_types(layout->scale_up, layout->zp_up, "W_up");
        if (m_config.produce_write_gate) {
            NODE_VALIDATION_CHECK(this,
                                  get_input_element_type(layout->w_inject).is_dynamic() ||
                                      get_input_element_type(layout->w_inject) == wd_et,
                                  "GatedResidual W_inject must have the weight element type ",
                                  wd_et);
            check_types(layout->scale_inject, layout->zp_inject, "W_inject");
        }
    } else if (m_config.produce_write_gate) {
        const auto inject_et = get_input_element_type(layout->w_inject);
        NODE_VALIDATION_CHECK(this,
                              inject_et.is_dynamic() || !gated_residual_detail::is_compressed_weight_type(inject_et),
                              "GatedResidual takes an i4/u4/i8/u8 W_inject exactly with the compressed input forms.");
    }

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_size(output_shapes.size());
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        set_output_type(i, out_et, output_shapes[i]);
    }
}

std::shared_ptr<Node> GatedResidual::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(internal_GatedResidual_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<GatedResidual>(new_args, m_config);
}

}  // namespace ov::op::internal

namespace ov {

std::ostream& operator<<(std::ostream& s, const ov::op::internal::GatedResidual::Mode& mode) {
    return s << as_string(mode);
}

template <>
OPENVINO_API EnumNames<ov::op::internal::GatedResidual::Mode>& EnumNames<ov::op::internal::GatedResidual::Mode>::get() {
    static auto enum_names = EnumNames<ov::op::internal::GatedResidual::Mode>(
        "ov::op::internal::GatedResidual::Mode",
        {
            {"read", ov::op::internal::GatedResidual::Mode::READ},
            {"write", ov::op::internal::GatedResidual::Mode::WRITE},
            {"final_mix", ov::op::internal::GatedResidual::Mode::FINAL_MIX},
        });
    return enum_names;
}

}  // namespace ov
