// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <utility>

#include "intel_gpu/runtime/layout.hpp"
#include "lstm_seq_inst.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

struct LSTMSeqImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::lstm_seq")
    explicit LSTMSeqImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<lstm_seq>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        for (size_t idx = 0; idx < node.get_dependencies().size(); idx++) {
            in_fmts[idx] = format::bfyx;
        }
        out_fmts[0] = format::ybfx;
        for (size_t idx = 1; idx < node.get_outputs_count(); idx++) {
            out_fmts[idx] = format::bfyx;
        }

        return {in_fmts, out_fmts};
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<lstm_seq>());

        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        const auto& info = engine.get_device_info();

        // XeTLA LSTM optimized for Xe2 architectures
        if (!check_cm_jit_support(engine, config) || info.arch != gpu_arch::xe2 || !config.get_use_cm()) {
            return false;
        }

        if (node.has_fused_primitives()) {
            return false;
        }

        const auto& lstm_node = node.as<lstm_seq>();
        const auto& lstm_prim = lstm_node.get_primitive();
        if (lstm_prim->clip > 0.0f) {
            return false;
        }

        if (lstm_prim->activations.size() != 3 || lstm_prim->activations[0] != activation_func::logistic ||
            lstm_prim->activations[1] != activation_func::hyperbolic_tan || lstm_prim->activations[2] != activation_func::hyperbolic_tan) {
            return false;
        }

        auto in_layouts = node.get_input_layouts();
        if (node.is_dynamic()) {
            return false;
        }
        const uint32_t expected_inputs = 7;
        if (in_layouts.size() != expected_inputs) {
            return false;
        }
        {
            auto& seq_lengths = in_layouts[expected_inputs - 1];
            if (seq_lengths.format != format::bfyx || seq_lengths.data_type != data_types::i32) {
                return false;
            }
            in_layouts.pop_back();
        }

        auto out_layouts = node.get_output_layouts();
        for (auto& layout : in_layouts) {
            if (layout.format != format::bfyx || layout.data_type != data_types::f16) {
                return false;
            }
        }
        for (auto& layout : out_layouts) {
            if (layout.data_type != data_types::f16) {
                return false;
            }
        }

        auto num_gates = 4;
        auto batch_size = in_layouts[0].get_shape()[0];
        auto input_size = in_layouts[0].get_shape()[2];
        auto hidden_size = in_layouts[3].get_shape()[1] / num_gates;
        auto num_dir = in_layouts[3].get_shape()[0];
        return hidden_size == 128 && batch_size == 1 && num_dir == 2 && (input_size == 64 || input_size == 256);
    }
};

}  // namespace ov::intel_gpu::cm
