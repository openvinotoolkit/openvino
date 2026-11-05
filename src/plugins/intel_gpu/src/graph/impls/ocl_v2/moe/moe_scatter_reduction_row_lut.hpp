// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/moe_scatter_reduction.hpp"
#include "moe_scatter_reduction_base.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {
struct MoeScatterReductionRowLut : public MoeScatterReductionBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::moe_scatter_reduction::row_lut")
    explicit MoeScatterReductionRowLut(shape_types shape_type, ValidateFunc vf = nullptr) : MoeScatterReductionBase(shape_type, std::move(vf)) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        if (!node.as<moe_scatter_reduction>().get_primitive()->use_row_lut) {
            return false;
        }
        const auto& data_layout = node.get_input_layout(0);
        const auto& weights_layout = node.get_input_layout(1);
        const auto& out_layout = node.get_output_layout(0);
        const auto& data_shape = data_layout.get_partial_shape();
        if (data_shape.rank().get_length() != 2 || data_shape[1].is_dynamic()) {
            return false;
        }
        const auto hidden_size = static_cast<size_t>(data_shape[1].get_length());
        const auto max_wgs = node.get_program().get_engine().get_device_info().max_work_group_size;
        if (hidden_size % (block_size * get_dispatch(max_wgs, hidden_size).second) != 0) {
            return false;
        }
        for (const auto& l : {data_layout, weights_layout, out_layout}) {
            if (l.format != format::bfyx || !one_of(l.data_type, {ov::element::f16, ov::element::f32})) {
                return false;
            }
        }
        return data_layout.data_type == weights_layout.data_type && data_layout.data_type == out_layout.data_type &&
               node.get_input_layout(2).data_type == ov::element::i32;
    }

    // Work-group size and vector blocks per work-item covering one hidden row.
    static std::pair<size_t, size_t> get_dispatch(size_t max_wgs, size_t hidden_size) {
        const size_t blocks = (hidden_size + block_size - 1) / block_size;
        const size_t batches_per_thread = (blocks + max_wgs - 1) / max_wgs;
        return {(blocks + batches_per_thread - 1) / batches_per_thread, batches_per_thread};
    }

    static constexpr size_t block_size = 4;
};
}  // namespace ov::intel_gpu::ocl
