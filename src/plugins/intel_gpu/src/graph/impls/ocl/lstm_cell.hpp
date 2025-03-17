// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_cell_inst.h"
#include "registry/implementation_manager.hpp"
#include "intel_gpu/runtime/layout.hpp"

#include <memory>
namespace cldnn {
namespace ocl {

struct LSTMCellImplementationManager: public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::lstm_cell")
    LSTMCellImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<lstm_cell>());

        const auto& input_layout = node.get_input_layout(0);
        const auto& output_layout = node.get_output_layout(0);

        auto input_fmt = input_layout.format;
        auto output_fmt = output_layout.format;
        auto in_dt = input_layout.data_type;
        auto out_dt = output_layout.data_type;
        static const std::vector<format::type> supported_formats = {
            format::bfyx,
            format::fyxb,
        };
        static const std::vector<ov::element::Type_t> supported_data_types = {
            data_types::f32,
            data_types::f16,
        };

        if (!one_of(in_dt, supported_data_types) || !one_of(out_dt, supported_data_types)) {
            return false;
        }

        return one_of(input_fmt.value, supported_formats) && one_of(output_fmt.value, supported_formats);
    }
};

}  // namespace ocl
}  // namespace cldnn
