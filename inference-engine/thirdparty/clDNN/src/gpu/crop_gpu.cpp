// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "crop_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"
#include "error_handler.h"

namespace cldnn {
namespace gpu {

struct crop_gpu : typed_primitive_gpu_impl<crop> {
    using parent = typed_primitive_gpu_impl<crop>;
    using parent::parent;

protected:
    bool optimized_out(crop_inst& instance) const override {
        return parent::optimized_out(instance) || _outer.can_be_optimized();
    }

public:
    static primitive_impl* create(const crop_node& arg) {
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg, 1);
        auto ew_optional_params =
            get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        ew_params.operations.push_back(
            {{kernel_selector::eltwise_params::InputType::Buffer(0)}, kernel_selector::eltwise_mode::ASSIGN});

        const auto& input_layout = arg.input().get_output_layout();
        ew_params.inputs[0] = convert_data_tensor(input_layout, 1, arg.get_primitive()->offsets);

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto crop = new crop_gpu(arg, best_kernels[0]);

        return crop;
    }
};

namespace detail {

attach_crop_gpu::attach_crop_gpu() {
    auto val_fw = crop_gpu::create;

    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::yxfb), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::yxfb), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::yxfb), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::yxfb), val_fw);

    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::bfyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);

    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::byxf), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::byxf), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::byxf), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::byxf), val_fw);

    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::fyxb), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::fyxb), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::fyxb), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::fyxb), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::fyxb), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::fyxb), val_fw);

    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::bfzyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx), val_fw);

    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::bfwzyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfwzyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfwzyx), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfwzyx), val_fw);

    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv16), val_fw);

    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_zyx_fsv16), val_fw);

    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_yx_bsv16_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bs_fs_yx_bsv16_fsv16), val_fw);

    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<crop>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bs_fs_zyx_bsv16_fsv16), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
