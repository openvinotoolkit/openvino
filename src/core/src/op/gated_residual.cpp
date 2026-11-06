// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_residual.hpp"

#include "itt.hpp"

namespace ov::op::internal {

namespace {

/// Rank is dynamic, or last dimension is dynamic, or it equals `expected`.
bool last_dim_matches(const ov::PartialShape& ps, size_t expected) {
    return ps.rank().is_dynamic() ||
           (ps.rank().get_length() > 0 && ps[ps.rank().get_length() - 1].compatible(static_cast<int64_t>(expected)));
}

/// Rank-2 weight [rows, cols] where each static dimension must match.
bool matrix_matches(const ov::PartialShape& ps, size_t rows, size_t cols) {
    return ps.rank().is_dynamic() || (ps.rank().get_length() == 2 && ps[0].compatible(static_cast<int64_t>(rows)) &&
                                      ps[1].compatible(static_cast<int64_t>(cols)));
}

}  // namespace

GatedResidual::GatedResidual(const ov::OutputVector& args,
                             Mode mode,
                             size_t hc_count,
                             size_t hidden_size,
                             size_t lowrank,
                             float epsilon,
                             bool norm_gain_plus_one,
                             bool produce_write_gate)
    : Op(args),
      m_mode(mode),
      m_hc_count(hc_count),
      m_hidden_size(hidden_size),
      m_lowrank(lowrank),
      m_epsilon(epsilon),
      m_norm_gain_plus_one(norm_gain_plus_one),
      m_produce_write_gate(produce_write_gate) {
    constructor_validate_and_infer_types();
}

bool GatedResidual::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_GatedResidual_visit_attributes);
    auto mode = static_cast<int64_t>(m_mode);
    visitor.on_attribute("mode", mode);
    m_mode = static_cast<Mode>(mode);
    visitor.on_attribute("hc_count", m_hc_count);
    visitor.on_attribute("hidden_size", m_hidden_size);
    visitor.on_attribute("lowrank", m_lowrank);
    visitor.on_attribute("epsilon", m_epsilon);
    visitor.on_attribute("norm_gain_plus_one", m_norm_gain_plus_one);
    visitor.on_attribute("produce_write_gate", m_produce_write_gate);
    return true;
}

void GatedResidual::validate_and_infer_types() {
    OV_OP_SCOPE(internal_GatedResidual_validate_and_infer_types);
    const auto input_size = get_input_size();
    const auto out_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this, m_hc_count > 0, "GatedResidual requires a positive branch count C.");
    NODE_VALIDATION_CHECK(this, m_hidden_size > 0, "GatedResidual requires a positive hidden size D.");
    NODE_VALIDATION_CHECK(this,
                          out_et.is_dynamic() || out_et.is_real(),
                          "GatedResidual residual must have a floating-point element type, got ",
                          out_et);

    const auto& residual_ps = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
                          residual_ps.rank().is_dynamic() || residual_ps.rank().get_length() >= 2,
                          "GatedResidual requires the residual input rank to be at least 2.");
    NODE_VALIDATION_CHECK(this,
                          last_dim_matches(residual_ps, m_hc_count * m_hidden_size),
                          "GatedResidual residual last dimension must be C*D = ",
                          m_hc_count * m_hidden_size,
                          ", got ",
                          residual_ps);

    if (m_mode == Mode::WRITE) {
        NODE_VALIDATION_CHECK(this, input_size == 3, "GatedResidual (WRITE) expects 3 inputs, got ", input_size);
        NODE_VALIDATION_CHECK(this,
                              last_dim_matches(get_input_partial_shape(WriteInputIdx::SUBLAYER_Y), m_hidden_size),
                              "GatedResidual (WRITE) sub-layer output last dimension must be D = ",
                              m_hidden_size);
        NODE_VALIDATION_CHECK(this,
                              last_dim_matches(get_input_partial_shape(WriteInputIdx::WRITE_GATE), m_hc_count),
                              "GatedResidual (WRITE) write gate last dimension must be C = ",
                              m_hc_count);
        auto merged_et = out_et;
        for (size_t i = 1; i < input_size; ++i) {
            NODE_VALIDATION_CHECK(this,
                                  ov::element::Type::merge(merged_et, merged_et, get_input_element_type(i)),
                                  "GatedResidual (WRITE) inputs must share the residual element type.");
        }
        // R' has exactly the residual layout, its width is known even when the residual's is not.
        auto output_ps = residual_ps;
        if (output_ps.rank().is_static()) {
            output_ps[output_ps.size() - 1] = static_cast<int64_t>(m_hc_count * m_hidden_size);
        }
        set_output_type(0, merged_et, output_ps);
        return;
    }

    // READ / FINAL_MIX
    const auto layout = read_layout(input_size, m_produce_write_gate);
    NODE_VALIDATION_CHECK(this,
                          layout.has_value(),
                          "GatedResidual (READ/FINAL_MIX) expects 4 inputs (5 with W_inject), 6 / 8 with compressed "
                          "weights (without / with zero "
                          "points), or 8 / 11 with compressed weights and W_inject, got ",
                          input_size);
    const bool compressed = layout->compressed;
    const bool has_zp = layout->has_zp;
    NODE_VALIDATION_CHECK(this,
                          m_mode != Mode::FINAL_MIX || !m_produce_write_gate,
                          "GatedResidual (FINAL_MIX) must not produce a write gate.");
    NODE_VALIDATION_CHECK(this, m_lowrank > 0, "GatedResidual (READ/FINAL_MIX) requires a positive low rank L.");

    const size_t cd = m_hc_count * m_hidden_size;
    const auto& gain_ps = get_input_partial_shape(ReadInputIdx::NORM_WEIGHT);
    NODE_VALIDATION_CHECK(this,
                          gain_ps.rank().is_dynamic() ||
                              (gain_ps.rank().get_length() == 1 && gain_ps[0].compatible(static_cast<int64_t>(cd))),
                          "GatedResidual norm weight must have shape [C*D] = [",
                          cd,
                          "], got ",
                          gain_ps);
    NODE_VALIDATION_CHECK(this,
                          matrix_matches(get_input_partial_shape(ReadInputIdx::W_DOWN), m_lowrank, cd),
                          "GatedResidual W_down must have shape [L, C*D] = [",
                          m_lowrank,
                          ", ",
                          cd,
                          "], got ",
                          get_input_partial_shape(ReadInputIdx::W_DOWN));
    NODE_VALIDATION_CHECK(this,
                          matrix_matches(get_input_partial_shape(ReadInputIdx::W_UP), cd, m_lowrank),
                          "GatedResidual W_up must have shape [C*D, L] = [",
                          cd,
                          ", ",
                          m_lowrank,
                          "], got ",
                          get_input_partial_shape(ReadInputIdx::W_UP));

    const auto wd_et = get_input_element_type(ReadInputIdx::W_DOWN);
    const auto wu_et = get_input_element_type(ReadInputIdx::W_UP);
    NODE_VALIDATION_CHECK(this,
                          wd_et.is_dynamic() || wu_et.is_dynamic() || wd_et == wu_et,
                          "GatedResidual W_down and W_up must share one element type, got ",
                          wd_et,
                          " and ",
                          wu_et);
    const bool wd_quant = is_compressed_weight_type(wd_et);
    NODE_VALIDATION_CHECK(this,
                          wd_et.is_dynamic() || (compressed == wd_quant),
                          "GatedResidual takes i4/u4/i8/u8 weights exactly with the compressed input forms.");
    if (compressed) {
        // scale/zp are [G, N] (N contiguous), unlike FC's [N, G]; the group size along K is K / G (not an attribute).
        auto check_group = [&](size_t scale_idx, size_t zp_idx, size_t n, size_t k, const char* name) {
            const auto& sps = get_input_partial_shape(scale_idx);
            NODE_VALIDATION_CHECK(
                this,
                sps.rank().is_dynamic() || (sps.rank().get_length() == 2 && sps[1].compatible(static_cast<int64_t>(n))),
                "GatedResidual ",
                name,
                " scale must have shape [G, ",
                n,
                "], got ",
                sps);
            NODE_VALIDATION_CHECK(
                this,
                get_input_element_type(scale_idx).is_dynamic() || get_input_element_type(scale_idx).is_real(),
                "GatedResidual ",
                name,
                " scale must be a floating-point type (f16, bf16 or f32), got ",
                get_input_element_type(scale_idx));
            if (has_zp) {
                const auto& zps = get_input_partial_shape(zp_idx);
                NODE_VALIDATION_CHECK(this,
                                      zps.rank().is_dynamic() || sps.rank().is_dynamic() || zps == sps,
                                      "GatedResidual ",
                                      name,
                                      " zero point must match the scale shape, got ",
                                      zps);
                NODE_VALIDATION_CHECK(
                    this,
                    get_input_element_type(zp_idx).is_dynamic() || get_input_element_type(zp_idx) == wd_et,
                    "GatedResidual ",
                    name,
                    " zero point must have the weight element type ",
                    wd_et);
            }
            if (sps.rank().is_static() && sps[0].is_static()) {
                const auto g = static_cast<size_t>(sps[0].get_length());
                NODE_VALIDATION_CHECK(this,
                                      g != 0 && k % g == 0,
                                      "GatedResidual ",
                                      name,
                                      ": K = ",
                                      k,
                                      " is not divisible by the scale group count G = ",
                                      g);
            }
        };
        check_group(ReadInputIdx::SCALE_DOWN, ReadInputIdx::ZP_DOWN, m_lowrank, cd, "W_down");
        check_group(ReadInputIdx::SCALE_UP, ReadInputIdx::ZP_UP, cd, m_lowrank, "W_up");
        if (m_produce_write_gate) {
            NODE_VALIDATION_CHECK(this,
                                  get_input_element_type(layout->w_inject).is_dynamic() ||
                                      get_input_element_type(layout->w_inject) == wd_et,
                                  "GatedResidual W_inject must have the weight element type ",
                                  wd_et);
            check_group(layout->scale_inject, layout->zp_inject, m_hc_count, cd, "W_inject");
        }
    }
    if (m_produce_write_gate) {
        const size_t inject_idx = layout->w_inject;
        if (!compressed) {
            const auto inject_et = get_input_element_type(inject_idx);
            NODE_VALIDATION_CHECK(
                this,
                inject_et.is_dynamic() || !is_compressed_weight_type(inject_et),
                "GatedResidual takes an i4/u4/i8/u8 W_inject exactly with the compressed input forms.");
        }
        NODE_VALIDATION_CHECK(this,
                              matrix_matches(get_input_partial_shape(inject_idx), m_hc_count, cd),
                              "GatedResidual W_inject must have shape [C, C*D] = [",
                              m_hc_count,
                              ", ",
                              cd,
                              "], got ",
                              get_input_partial_shape(inject_idx));
    }

    auto mixed_ps = residual_ps;
    if (mixed_ps.rank().is_static()) {
        mixed_ps[mixed_ps.rank().get_length() - 1] = ov::Dimension(static_cast<int64_t>(m_hidden_size));
    }
    set_output_size(m_produce_write_gate ? 2 : 1);
    set_output_type(0, out_et, mixed_ps);

    if (m_produce_write_gate) {
        auto gate_ps = residual_ps;
        if (gate_ps.rank().is_static()) {
            gate_ps[gate_ps.rank().get_length() - 1] = ov::Dimension(static_cast<int64_t>(m_hc_count));
        }
        set_output_type(1, out_et, gate_ps);
    }
}

std::shared_ptr<Node> GatedResidual::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<GatedResidual>(new_args,
                                           m_mode,
                                           m_hc_count,
                                           m_hidden_size,
                                           m_lowrank,
                                           m_epsilon,
                                           m_norm_gain_plus_one,
                                           m_produce_write_gate);
}

}  // namespace ov::op::internal
