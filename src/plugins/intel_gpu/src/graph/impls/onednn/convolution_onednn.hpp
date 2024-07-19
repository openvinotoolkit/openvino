// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_inst.h"
#include "intel_gpu/runtime/format.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/utils.hpp"

#include "impls/registry/implementation_manager.hpp"

#include "utils.hpp"

#include <memory>

namespace cldnn {
namespace onednn {

struct ConvolutionImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ConvolutionImplementationOnednn")
    ConvolutionImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<convolution>());
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad)
            return false;

        const auto& conv_node = node.as<convolution>();
        if (!is_supported_format(node.get_preferred_input_fmt(0)))
            return false;

        auto in_dt = conv_node.get_input_layout(0).data_type;
        auto wei_dt = conv_node.weights().get_output_layout().data_type;
        auto out_dt = conv_node.get_output_layout(false).data_type;

        bool f16_conv = everyone_is(data_types::f16, in_dt, wei_dt) && one_of(out_dt, {data_types::f16, data_types::f32, data_types::u8, data_types::i8});
        bool u8s8_conv = one_of(in_dt, {data_types::i8, data_types::u8}) &&
                         wei_dt == data_types::i8 &&
                         one_of(out_dt, {data_types::i32, data_types::f16, data_types::f32, data_types::u8, data_types::i8});

        if (!f16_conv && !u8s8_conv)
            return false;

        if (!is_supported_post_ops(conv_node))
            return false;

        if (conv_node.get_primitive()->deformable_mode)
            return false;

        // oneDNN doesn't support asymmetric weights quantization
        if (conv_node.weights_zero_points_term())
            return false;

        return ImplementationManager::validate(node);
    }

    in_out_fmts_t query_formats(const program_node& node) const override;

    bool support_shapes(const kernel_impl_params& params) const override {
        return true;
    }
};

}  // namespace onednn
}  // namespace cldnn
