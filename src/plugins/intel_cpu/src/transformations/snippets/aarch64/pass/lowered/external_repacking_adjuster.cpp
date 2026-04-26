// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "external_repacking_adjuster.hpp"

#include <cstddef>

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p32x1b_x16_x16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x32p16x1b_x32_x32_neon.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::pass::aarch64 {
namespace {

size_t get_packed_size(size_t N, size_t K, const ov::element::Type& precision) {
    if (precision == ov::element::f16) {
        return kai_get_rhs_packed_size_rhs_pack_kxn_x16p32x1b_x16_x16_neon(N, K);
    }
    if (precision == ov::element::f32) {
        return kai_get_rhs_packed_size_rhs_pack_kxn_x32p16x1b_x32_x32_neon(N, K);
    }
    OPENVINO_THROW("Unsupported precision for aarch64 GEMM weights repacking: ", precision.get_type_name());
}

ov::snippets::VectorDims get_repacked_offsets(const ov::snippets::VectorDims& planar_shape,
                                              size_t target_rank,
                                              const ov::element::Type& precision) {
    OPENVINO_ASSERT(planar_shape.size() >= 2, "GEMM weights must have rank >= 2");
    OPENVINO_ASSERT(target_rank >= planar_shape.size(), "Incorrect target rank for repacked GEMM weights offsets");

    const auto K = *++planar_shape.rbegin();
    const auto N = *planar_shape.rbegin();
    OPENVINO_ASSERT(!ov::snippets::utils::is_dynamic_value(N) && !ov::snippets::utils::is_dynamic_value(K),
                    "N and K shape should not be dynamic for pre-packed aarch64 GEMM weights");

    const auto packed_bytes = get_packed_size(N, K, precision);
    OPENVINO_ASSERT(packed_bytes % precision.size() == 0, "Unexpected packed weights byte size alignment");

    auto allocation_shape = planar_shape;
    allocation_shape[allocation_shape.size() - 2] = 1;
    allocation_shape[allocation_shape.size() - 1] = packed_bytes / precision.size();

    ov::snippets::VectorDims shape_for_offset(target_rank - allocation_shape.size(), 1);
    shape_for_offset.insert(shape_for_offset.end(), allocation_shape.begin(), allocation_shape.end());

    ov::snippets::VectorDims dst_offsets;
    ov::snippets::utils::init_strides(shape_for_offset, target_rank, precision.size(), 0, dst_offsets);
    return dst_offsets;
}

}  // namespace

GemmExternalRepackingAdjuster::GemmExternalRepackingAdjuster(
    const ov::snippets::lowered::LinearIRCPtr& linear_ir,
    const CPURuntimeConfigurator* configurator)
    : ov::snippets::lowered::pass::RuntimeOptimizer(configurator) {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_configurator->get_config());
    const auto& params = linear_ir->get_parameters();
    for (const auto& [idx, input_repacker] : cpu_config->input_repackers) {
        OPENVINO_ASSERT(idx < params.size(), "Incorrect index of repacked input");
        OPENVINO_ASSERT(input_repacker.already_repacked(),
                        "Runtime aarch64 GEMM weights repacking is not supported");
        m_repacked_inputs.push_back(idx);
    }
}

bool GemmExternalRepackingAdjuster::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::GemmExternalRepackingAdjuster")
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_configurator->get_config());
    for (const auto idx : m_repacked_inputs) {
        const auto& shape = cpu_config->io_shapes[idx];
        const auto& layout = cpu_config->io_layouts[idx];
        const auto& precision = linear_ir.get_parameters()[idx]->get_node()->get_output_element_type(0);
        const auto planar_shape = ov::snippets::utils::get_planar_vdims(shape, layout);

        cpu_config->io_data_offsets[idx] =
            get_repacked_offsets(planar_shape, cpu_config->io_data_offsets[idx].size(), precision);
        cpu_config->input_repackers.erase(idx);
    }
    return true;
}

}  // namespace ov::intel_cpu::pass::aarch64
