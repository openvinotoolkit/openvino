// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <utility>

#include "group_normalization_base.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct GroupNormalizationRef : public GroupNormalizationBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::group_norm::ref")
    explicit GroupNormalizationRef(shape_types shape_type, ValidateFunc vf = nullptr) : GroupNormalizationBase(shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        // GroupNormalization-12 defines data, scale, bias and output as floating point types.
        // Scale and bias may have a different floating point type from data.
        static constexpr std::array supported_types = {
            ov::element::f32,
            ov::element::f16,
        };

        const auto& in0_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        const auto& in0_pshape = in0_layout.get_partial_shape();
        if (in0_pshape.rank().is_dynamic() || in0_pshape.size() < 2 || in0_pshape.size() > 6) {
            return false;
        }

        // The kernel picks the number of index arguments from the rank, so the format dimension must match it.
        const size_t fmt_dims = std::max<size_t>(in0_pshape.size(), 4);
        if (!is_supported_format(in0_layout.format, fmt_dims) || !is_supported_format(out_layout.format, fmt_dims)) {
            return false;
        }

        // Only data, scale and bias. The dependencies of fused ops follow them and are converted by the fused ops code.
        for (size_t i = 0; i < node.get_primitive()->input_size(); i++) {
            if (!one_of(node.get_input_layout(i).data_type, supported_types)) {
                return false;
            }
        }
        if (!one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        // The kernel computes buffer offsets in 32 bits. Dynamic shapes cannot be checked here and share this limit.
        constexpr size_t max_elements = size_t{1} << 32;
        for (const auto* l : {&in0_layout, &out_layout}) {
            if (l->is_static() && l->get_linear_size() >= max_elements) {
                return false;
            }
        }

        return fused_ops_are_one_of<eltwise, activation, reorder>(node.get_fused_primitives());
    }

private:
    // The kernel indexes through the GET_INDEX macros, so it works for other simple and blocked formats whose
    // dimension matches the rank. But format propagation keeps a node in any format some impl accepts. Accepting
    // formats the opt impls reject would make this ref replace their reorder + opt path, so only the formats of the
    // opt impls (plus rank 6) are accepted.
    static bool is_supported_format(const format& fmt, size_t dims) {
        static constexpr std::array supported_fmts = {
            format::bfyx,
            format::bfzyx,
            format::bfwzyx,
            format::b_fs_yx_fsv16,
        };

        return fmt.dimension() == dims && one_of(fmt, supported_fmts);
    }
};

}  // namespace ov::intel_gpu::ocl
