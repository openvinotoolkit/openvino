// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "intel_gpu/runtime/device_info.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// @brief GPU-internal Gated Residual (Hyper Connection) operation.
///
/// The op models the two halves of the gated-residual mechanism used by
/// Qwen4-Exp / Qwen3.8-Flash-Next style models, where the residual stream is widened
/// from [B,T,D] to [B,T,C,D] (physically stored as [B,T,C*D]).
///
/// READ / FINAL_MIX (branch merge, executed *before* a sub-layer):
///   Rbar_c = rms_norm(R_c) * gain_c                (branch-wise grouped RMSNorm, group width D)
///   G      = sigmoid(W_up * SiLU(W_down * Rbar / C))
///   mixed  = (1/C) * sum_c G_c (*) Rbar_c
///   S      = 2 * sigmoid(W_inject * Rbar / C)      (optional, only when produce_write_gate)
///
/// WRITE (branch update, executed *after* a sub-layer):
///   R'_c = R_c + S_c * Y
///
/// Notes on semantics that must not be "optimized away":
///  - the normalization is per-branch over D channels, *not* a single RMSNorm over C*D;
///  - all three 1/C factors are part of the model semantics;
///  - the read gate is per-(branch, channel), the write gate is a per-branch scalar in (0,2);
///  - `gain = 1 + w` when `norm_gain_plus_one` is set (checkpoint stores w initialized to 0).
class GatedResidual : public ov::op::Op {
public:
    OPENVINO_OP("GatedResidual", "gpu_opset");

    enum class Mode : int32_t {
        READ = 0,       ///< branch merge, optionally also produces the write gate
        WRITE = 1,      ///< branch update using a precomputed write gate
        FINAL_MIX = 2,  ///< model-tail mixer: READ without write-gate generation
    };

    /// @brief Input indices for Mode::READ / Mode::FINAL_MIX
    struct ReadInputIdx {
        static constexpr size_t RESIDUAL = 0;     ///< [B,T,C*D]
        static constexpr size_t NORM_WEIGHT = 1;  ///< [C*D]
        static constexpr size_t W_DOWN = 2;       ///< [L, C*D]
        static constexpr size_t W_UP = 3;         ///< [C*D, L]
        static constexpr size_t W_INJECT = 4;     ///< [C, C*D], only when produce_write_gate and the weights are not compressed

        // Compressed weights (W_DOWN / W_UP are u4 or u8). scale/zp are [G, N] (N contiguous), unlike FC's [N, G]: one sub-group
        // block read gives every lane the scale of its own output channel.
        static constexpr size_t SCALE_DOWN = 4;  ///< f16 [G_d, L]
        static constexpr size_t SCALE_UP = 5;    ///< f16 [G_u, C*D]
        static constexpr size_t ZP_DOWN = 6;     ///< same type as the weights (u4 packed / u8) [G_d, L]
        static constexpr size_t ZP_UP = 7;       ///< same type as the weights (u4 packed / u8) [G_u, C*D]

        // The write-gate inputs are always the last inputs; with compressed weights W_inject is compressed the same way.
        static constexpr size_t W_INJECT_Q = 8;    ///< u4 / u8 [C, C*D]
        static constexpr size_t SCALE_INJECT = 9;  ///< f16 [G_i, C]
        static constexpr size_t ZP_INJECT = 10;    ///< same type as the weights (u4 packed / u8) [G_i, C]
    };

    static constexpr size_t compressed_input_count = 8;
    static constexpr size_t compressed_write_gate_input_count = 11;
    static constexpr size_t int4_group_size = 32;  ///< the only group size along K the kernels support for u4 weights

    /// @brief Input indices for Mode::WRITE
    struct WriteInputIdx {
        static constexpr size_t RESIDUAL = 0;    ///< [B,T,C*D]
        static constexpr size_t SUBLAYER_Y = 1;  ///< [B,T,D]
        static constexpr size_t WRITE_GATE = 2;  ///< [B,T,C]
    };

    GatedResidual() = default;

    GatedResidual(const ov::OutputVector& args,
                  Mode mode,
                  size_t hc_count,
                  size_t hidden_size,
                  size_t lowrank,
                  float epsilon,
                  bool norm_gain_plus_one,
                  bool produce_write_gate);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    Mode get_mode() const {
        return m_mode;
    }
    size_t get_hc_count() const {
        return m_hc_count;
    }
    size_t get_hidden_size() const {
        return m_hidden_size;
    }
    size_t get_lowrank() const {
        return m_lowrank;
    }
    float get_epsilon() const {
        return m_epsilon;
    }
    bool get_norm_gain_plus_one() const {
        return m_norm_gain_plus_one;
    }
    bool get_produce_write_gate() const {
        return m_produce_write_gate;
    }

    /// Geometry the kernels can execute; shared by the fusion pass and the impl's validate_impl.
    static constexpr size_t supported_hc_count = 4;
    static constexpr size_t max_lowrank = 512;        // kept in registers by gr_read.cl
    static constexpr size_t max_total_width = 16384;  // C*D must fit into SLM as f16

    static constexpr bool is_geometry_supported(Mode mode, size_t hc_count, size_t hidden_size, size_t lowrank) {
        if (hc_count != supported_hc_count || hidden_size == 0 || hidden_size % 16 != 0 || hc_count * hidden_size > max_total_width) {
            return false;
        }
        return mode == Mode::WRITE || (lowrank != 0 && lowrank % 16 == 0 && lowrank <= max_lowrank);
    }

    /// Compressed weights run on the systolic path only.
    static bool is_compression_arch_supported(const cldnn::device_info& info) {
        return info.arch >= cldnn::gpu_arch::xe2 && info.supports_immad && info.supports_intel_subgroups && info.supports_intel_subgroups_short;
    }

    /// Group size along K, derived from the weight columns K and the scale / zp rows G; 0 when K is not divisible by G.
    static constexpr size_t derive_group_size(size_t k, size_t g) {
        return (g == 0 || k % g != 0) ? 0 : k / g;
    }

    /// Group sizes the kernels support: 32 for u4, per-channel (group size == K) for u8.
    static constexpr bool is_group_size_supported(size_t bits, size_t k, size_t group_size) {
        return group_size != 0 && (bits == 4 ? group_size == int4_group_size : group_size == k);
    }

    /// Shared by the fusion pass and the impl's validate_impl. N/K/G are the weight rows / columns / scale rows of W_down and
    /// W_up; bits is 4 (u4) or 8 (u8). The group size is K / G.
    static bool is_weight_compression_supported(size_t bits, size_t n_down, size_t k_down, size_t g_down, size_t n_up, size_t k_up, size_t g_up) {
        if (bits != 4 && bits != 8) {
            return false;
        }
        auto matrix_ok = [bits](size_t n, size_t k, size_t g) {
            if (n == 0 || n % 16 != 0 || k == 0 || k % 32 != 0) {
                return false;
            }
            if (!is_group_size_supported(bits, k, derive_group_size(k, g))) {
                return false;
            }
            return (k * bits / 8) % 16 == 0;
        };
        return matrix_ok(n_down, k_down, g_down) && matrix_ok(n_up, k_up, g_up);
    }

    /// Compressed W_inject [C, K = C*D] with a [G, C] scale / zp. It has only C (< 16) output rows, so the N % 16 rule of
    /// is_weight_compression_supported does not apply; the group rules are the same as for W_down / W_up.
    static bool is_write_gate_compression_supported(size_t bits, size_t hc_count, size_t k, size_t g) {
        if ((bits != 4 && bits != 8) || hc_count == 0 || k == 0 || k % 32 != 0) {
            return false;
        }
        return is_group_size_supported(bits, k, derive_group_size(k, g)) && (k * bits / 8) % 16 == 0;
    }

private:
    Mode m_mode = Mode::READ;
    size_t m_hc_count = 4;
    size_t m_hidden_size = 0;
    size_t m_lowrank = 0;
    float m_epsilon = 1e-6f;
    bool m_norm_gain_plus_one = true;
    bool m_produce_write_gate = false;
};

}  // namespace ov::intel_gpu::op
