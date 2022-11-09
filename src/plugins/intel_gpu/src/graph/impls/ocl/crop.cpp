// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "crop_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct crop_impl : typed_primitive_impl_ocl<crop> {
    using parent = typed_primitive_impl_ocl<crop>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<crop_impl>(*this);
    }

    explicit crop_impl(const crop_impl& other) : parent(other),
        _can_be_optimized(other._can_be_optimized) {}

    crop_impl(const crop_node& arg, const kernel_selector::kernel_data& kd) : parent(arg, kd) {
        set_node_params(arg);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<crop>());
        const auto& node = arg.as<crop>();
        _can_be_optimized = node.can_be_optimized();
    }

protected:
    bool optimized_out(crop_inst& instance) const override {
        return parent::optimized_out(instance) || _can_be_optimized;
    }

public:
    static primitive_impl* create(const crop_node& arg, const kernel_impl_params& impl_param) {
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(impl_param, 1);
        auto ew_optional_params = get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        ew_params.operations.push_back(
            {{kernel_selector::eltwise_params::InputType::Buffer(0)}, kernel_selector::eltwise_mode::ASSIGN});
        ew_params.inputs[0] = convert_data_tensor(impl_param.get_input_layout(), 1, impl_param.input_offsets[0]);

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto crop = new crop_impl(arg, best_kernels[0]);

        return crop;
    }

private:
    bool _can_be_optimized;
};

namespace detail {

attach_crop_impl::attach_crop_impl() {
    implementation_map<crop>::add(impl_types::ocl, crop_impl::create, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i64, format::yxfb),
        std::make_tuple(data_types::i32, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i64, format::byxf),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
        std::make_tuple(data_types::f32, format::fyxb),
        std::make_tuple(data_types::f16, format::fyxb),
        std::make_tuple(data_types::i64, format::fyxb),
        std::make_tuple(data_types::i32, format::fyxb),
        std::make_tuple(data_types::i8, format::fyxb),
        std::make_tuple(data_types::u8, format::fyxb),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i64, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),

        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i64, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv32_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
