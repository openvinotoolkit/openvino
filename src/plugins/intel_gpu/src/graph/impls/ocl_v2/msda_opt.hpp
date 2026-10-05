// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <memory>
#include <utility>

#include "msda_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct MSDAOptImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::msda::opt")
    explicit MSDAOptImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    // Raw linear indexing requires contiguous, unpadded plain buffers. Request
    // bfyx for every input; the graph may pack higher-rank tensors into bfyx.
    // All logical dimensions come from validated element counts, not packed axes.
    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        return {std::vector<format::type>(5, format::bfyx), {format::bfyx}};
    }

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        if (!node.is_type<msda>() || node.get_dependencies().size() != 5 || node.get_outputs_count() != 1)
            return false;
        const auto plain = [](const cldnn::layout& l) {
            const bool contiguous = l.format == format::bfyx || l.format == format::bfzyx || l.format == format::bfwzyx;
            return !l.is_dynamic() && contiguous && !l.get_padding() && !l.get_padding().is_dynamic();
        };
        for (size_t i = 0; i < 5; ++i) {
            if (!plain(node.get_input_layout(i)))
                return false;
        }
        const auto& out = node.get_output_layout(0);
        if (!plain(out))
            return false;
        const auto value = node.get_input_layout(0);
        const auto loc = node.get_input_layout(3);
        const auto weight = node.get_input_layout(4);
        if ((value.data_type != data_types::f16 && value.data_type != data_types::f32) || loc.data_type != value.data_type ||
            weight.data_type != value.data_type || out.data_type != value.data_type || node.get_input_layout(1).data_type != data_types::i32 ||
            node.get_input_layout(2).data_type != data_types::i32)
            return false;

        // Kernel pointer arithmetic uses signed 32-bit offsets.
        constexpr auto max_index = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
        if (value.count() > max_index || loc.count() > max_index || weight.count() > max_index || out.count() > max_index)
            return false;

        const auto v = value.get_shape();
        const auto s = node.get_input_layout(1).get_shape();
        const auto c = loc.get_shape();
        const auto w = weight.get_shape();
        const auto o = out.get_shape();
        if (v.size() != 4 || s.empty() || c.size() < 2 || w.size() < 2 || o.size() < 3)
            return false;
        const uint64_t batch = v[0], keys = v[1], heads = v[2], channels = v[3];
        const uint64_t levels = s[0], queries = w[1];
        if (!batch || !keys || !heads || !channels || !levels || !queries || node.get_input_layout(1).count() != levels * 2 ||
            node.get_input_layout(2).count() != levels || c[0] != batch || c[1] != queries || w[0] != batch || o[0] != batch || o[1] != queries ||
            o[2] != heads * channels)
            return false;

        const uint64_t weight_count = weight.count();
        const uint64_t denominator = batch * queries * heads * levels;
        if (!denominator || weight_count % denominator != 0)
            return false;
        const uint64_t points = weight_count / denominator;
        return points != 0 && loc.count() == 2 * weight_count && out.count() == batch * queries * heads * channels;
    }
};

}  // namespace ov::intel_gpu::ocl
