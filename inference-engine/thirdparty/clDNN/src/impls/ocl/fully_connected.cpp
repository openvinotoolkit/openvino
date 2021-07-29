// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "fully_connected_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "fully_connected/fully_connected_kernel_selector.h"
#include "fully_connected/fully_connected_params.h"

#include "network_impl.h"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_runner.h"

#include "cldnn/primitives/reorder.hpp"
#include "cldnn/primitives/input_layout.hpp"
#include <memory>

namespace cldnn {
namespace ocl {

struct fully_connected_impl : typed_primitive_impl_ocl<fully_connected> {
    using parent = typed_primitive_impl_ocl<fully_connected>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<fully_connected_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<fully_connected>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = instance.weights_memory();
        args.bias = instance.bias_term() ? instance.bias_memory() : nullptr;

        return args;
    }

public:
    static primitive_impl* create(const fully_connected_node& arg) {
        auto fc_params = get_weights_bias_default_params<kernel_selector::fully_connected_params>(arg);
        auto fc_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::fully_connected_optional_params>(
                arg.get_program());
        fc_optional_params.allowInputReordering = true;

        const auto primitive = arg.get_primitive();

        if (primitive->input_size != 3)
            fc_params.output = fc_params.output.FlattenFeatureAndSpatials();

        if (arg.get_output_layout().data_type == data_types::i8 ||
            arg.get_output_layout().data_type == data_types::u8) {
            fc_params.quantization = kernel_selector::QuantizationType::SYMMETRIC;
        } else {
            fc_params.quantization = kernel_selector::QuantizationType::NONE;
        }

        fc_optional_params.tuningParams.runner =
            std::make_shared<gpu::kernel_runner>(arg.get_program().get_engine(), arg.get_program().get_id(), true);

        auto& kernel_selector = kernel_selector::fully_connected_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(fc_params, fc_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto fc = new fully_connected_impl(arg, best_kernels[0]);

        return fc;
    }
};

namespace detail {

attach_fully_connected_impl::attach_fully_connected_impl() {
    implementation_map<fully_connected>::add(impl_types::ocl, fully_connected_impl::create, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
