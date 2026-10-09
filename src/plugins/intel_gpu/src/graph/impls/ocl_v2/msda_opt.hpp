// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <utility>

#include "msda_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct MSDAOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::msda::opt")
    explicit MSDAOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        std::vector<format::type> in_fmts;
        for (size_t i = 0; i < node.get_dependencies().size(); i++)
            in_fmts.push_back(format::get_default_format(node.get_input_layout(i).get_rank()));
        return {in_fmts, {format::get_default_format(node.get_output_layout(0).get_rank())}};
    }

    // The op validation guarantees the input ranks and dimensions; the kernel
    // needs plain layouts, f16/f32 data, i32 level sizes and offsets, and
    // element counts that fit its 32-bit offsets.
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<msda>());
        static constexpr std::array supported_types = {ov::element::f16, ov::element::f32};
        const auto plain = [](const layout& l) {
            return l.format == format::get_default_format(l.get_rank()) && !l.get_padding();
        };
        const auto fits_int32 = [](const layout& l) {
            return l.count() <= static_cast<size_t>(std::numeric_limits<int32_t>::max());
        };
        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto& in_layout = node.get_input_layout(i);
            const bool index_input = i == 1 || i == 2;
            if (!plain(in_layout) || !fits_int32(in_layout) ||
                (index_input ? in_layout.data_type != ov::element::i32 : !one_of(in_layout.data_type, supported_types))) {
                return false;
            }
        }
        const auto& out_layout = node.get_output_layout(0);
        return plain(out_layout) && fits_int32(out_layout) && one_of(out_layout.data_type, supported_types);
    }
};

}  // namespace ov::intel_gpu::ocl
