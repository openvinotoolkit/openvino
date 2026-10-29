// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "openvino/core/type/element_type.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// GGUF block formats decoded by the gather_gguf kernel (gather_gguf.cl). Kept in sync with the
// per-format decoders in that kernel; other gguf_* types fall back to the FE dense f16 path.
inline bool gather_gguf_supported_type(ov::element::Type_t t) {
    switch (t) {
    case ov::element::Type_t::gguf_q4_0:
    case ov::element::Type_t::gguf_q4_1:
    case ov::element::Type_t::gguf_q8_0:
    case ov::element::Type_t::gguf_q4_k:
    case ov::element::Type_t::gguf_q5_k:
    case ov::element::Type_t::gguf_q6_k:
        return true;
    default:
        return false;
    }
}

struct GatherGGUFRef : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::gather_gguf::ref")
    explicit GatherGGUFRef(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                              const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        const auto& in0_layout = node.get_input_layout(0);
        const auto& in1_layout = node.get_input_layout(1);
        const auto& out_layout = node.get_output_layout(0);

        if (!ov::element::is_gguf_block(in0_layout.data_type) ||
            !gather_gguf_supported_type(in0_layout.data_type)) {
            return false;
        }
        if (in1_layout.data_type != ov::element::i32 && in1_layout.data_type != ov::element::i64) {
            return false;
        }
        if (out_layout.data_type != ov::element::f16) {
            return false;
        }
        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
