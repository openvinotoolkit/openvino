// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenation_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "concatenation/concatenation_kernel_selector.h"
#include "concatenation/concatenation_kernel_base.h"

#include <initializer_list>

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::concat_axis convert_axis(concatenation::concatenation_axis axis) {
    switch (axis) {
        case concatenation::along_x:
            return kernel_selector::concat_axis::X;
        case concatenation::along_y:
            return kernel_selector::concat_axis::Y;
        case concatenation::along_z:
            return kernel_selector::concat_axis::Z;
        case concatenation::along_w:
            return kernel_selector::concat_axis::W;
        case concatenation::along_f:
            return kernel_selector::concat_axis::FEATURE;
        case concatenation::along_b:
            return kernel_selector::concat_axis::BATCH;
        default:
            return kernel_selector::concat_axis::X;
    }
}
}  // namespace

struct concatenation_impl : typed_primitive_impl_ocl<concatenation> {
    using parent = typed_primitive_impl_ocl<concatenation>;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<concatenation_impl>(*this);
    }

    concatenation_impl(const concatenation_node& arg, const kernel_selector::kernel_data& kd) : parent(arg, kd) {
        if (!_outer.can_be_optimized()) {
            CLDNN_ERROR_NOT_EQUAL(_outer.id(),
                                  "Input count",
                                  _outer.inputs_count(),
                                  "kds size",
                                  kd.kernels.size(),
                                  "Error - not enough kernels for concatenation");
        }
    }

protected:
    bool optimized_out(concatenation_inst& instance) const override {
        return parent::optimized_out(instance) || _outer.can_be_optimized();
    }

public:
    static primitive_impl* create(const concatenation_node& arg) {
        if (arg.can_be_optimized()) {
            return new concatenation_impl(arg, {});
        }

        auto concat_params = get_default_params<kernel_selector::concatenation_params>(arg);
        auto concat_optional_params =
            get_default_optional_params<kernel_selector::concatenation_optional_params>(arg.get_program());
        auto axis = arg.get_primitive()->axis;

        concat_params.inputs.resize(arg.inputs_count());
        for (size_t i = 0; i < arg.inputs_count(); ++i) {
            const layout& input_layout = arg.input(i).get_output_layout();
            concat_params.inputs[i] = convert_data_tensor(input_layout);
        }

        concat_params.axis = convert_axis(axis);
        concat_optional_params.kernelPerInput = true;

        auto& kernel_selector = kernel_selector::concatenation_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(concat_params, concat_optional_params);
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        concatenation_impl* concat = new concatenation_impl(arg, best_kernels[0]);

        return concat;
    }
};

namespace detail {

attach_concatenation_impl::attach_concatenation_impl() {
    implementation_map<concatenation>::add(impl_types::ocl, concatenation_impl::create, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::i32, format::yxfb),
        std::make_tuple(data_types::i64, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::i64, format::byxf),
        std::make_tuple(data_types::f32, format::fyxb),
        std::make_tuple(data_types::f16, format::fyxb),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfzyx),
        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i64, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i64, format::bfwzyx),
        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
