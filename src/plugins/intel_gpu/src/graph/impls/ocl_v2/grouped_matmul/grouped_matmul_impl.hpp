// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grouped_matmul_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct GroupedMatmulImpl : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::grouped_matmul")

    explicit GroupedMatmulImpl(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        const auto& device_info = node.get_program().get_engine().get_device_info();
        // OneDNN grouped matmul requires systolic (IMMAD) support
        if (!device_info.supports_immad) {
            return false;
        }

        static const std::vector<format> supported_fmts = {
            format::bfyx,
        };

        static const std::vector<ov::element::Type_t> supported_activation_types = {
            ov::element::f16,
        };

        static const std::vector<ov::element::Type_t> supported_weight_types = {
            ov::element::f16,
            ov::element::u4,
            ov::element::i4,
            ov::element::i8,
            ov::element::u8,
        };

        const size_t input_idx = grouped_matmul::InputIdx::INPUT;
        if (!one_of(node.get_input_layout(input_idx).format, supported_fmts) ||
            !one_of(node.get_input_layout(input_idx).data_type, supported_activation_types)) {
            const auto layer_id = node.get_kernel_impl_params()->typed_desc<grouped_matmul>()->id;
            DO_NOT_USE_THIS_KERNEL(layer_id);
        }

        const size_t weight_idx = grouped_matmul::InputIdx::WEIGHT;
        if (!one_of(node.get_input_layout(weight_idx).format, supported_fmts) ||
            !one_of(node.get_input_layout(weight_idx).data_type, supported_weight_types)) {
            const auto layer_id = node.get_kernel_impl_params()->typed_desc<grouped_matmul>()->id;
            DO_NOT_USE_THIS_KERNEL(layer_id);
        }

        const auto& output_layout = node.get_output_layout(0);
        if (!one_of(output_layout.format, supported_fmts) ||
            !one_of(output_layout.data_type, supported_activation_types)) {
            const auto layer_id = node.get_kernel_impl_params()->typed_desc<grouped_matmul>()->id;
            DO_NOT_USE_THIS_KERNEL(layer_id);
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
