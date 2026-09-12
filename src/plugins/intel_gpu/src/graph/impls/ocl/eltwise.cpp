// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include <utility>

#include "common_utils/eltwise_kernel_params.hpp"
#include "eltwise_inst.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"

namespace cldnn {
namespace ocl {

struct eltwise_impl : typed_primitive_impl_ocl<eltwise> {
    using parent = typed_primitive_impl_ocl<eltwise>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::eltwise_kernel_selector;
    using kernel_params_t = kernel_selector::eltwise_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::eltwise_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<eltwise_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && !_kernel_data.kernelName.empty()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<eltwise>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto params = get_default_params<kernel_selector::eltwise_params>(impl_param, is_shape_agnostic);
        return lower_eltwise_params(impl_param, std::move(params));
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        return canonicalize_eltwise_shapes(impl_params);
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

namespace detail {

attach_eltwise_impl::attach_eltwise_impl() {
    auto dyn_types = {
        data_types::f32,
        data_types::f16,
        data_types::bf16,
        data_types::i8,
        data_types::u8,
        data_types::i16,
        data_types::u16,
        data_types::i32,
        data_types::u32,
        data_types::i64
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx,
        format::b_fs_yx_fsv16,
    };

    implementation_map<eltwise>::add(impl_types::ocl,
                                     shape_types::dynamic_shape,
                                     typed_primitive_impl_ocl<eltwise>::create<eltwise_impl>,
                                     dyn_types,
                                     dyn_formats);

    implementation_map<eltwise>::add(impl_types::ocl, shape_types::static_shape, typed_primitive_impl_ocl<eltwise>::create<eltwise_impl>, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::bf16, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::i16, format::yxfb),
        std::make_tuple(data_types::u16, format::yxfb),
        std::make_tuple(data_types::u32, format::yxfb),
        std::make_tuple(data_types::i32, format::yxfb),
        std::make_tuple(data_types::i64, format::yxfb),

        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::bf16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u16, format::bfyx),
        std::make_tuple(data_types::i16, format::bfyx),
        std::make_tuple(data_types::u32, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),

        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::bf16, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
        std::make_tuple(data_types::i16, format::byxf),
        std::make_tuple(data_types::u16, format::byxf),
        std::make_tuple(data_types::u32, format::byxf),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::i64, format::byxf),

        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::bf16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::bf16, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i16, format::bfzyx),
        std::make_tuple(data_types::u16, format::bfzyx),
        std::make_tuple(data_types::u32, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfzyx),

        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::bf16, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
        std::make_tuple(data_types::i16, format::bfwzyx),
        std::make_tuple(data_types::u16, format::bfwzyx),
        std::make_tuple(data_types::u32, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i64, format::bfwzyx),

        std::make_tuple(data_types::f32, format::bfuwzyx),
        std::make_tuple(data_types::f16, format::bfuwzyx),
        std::make_tuple(data_types::bf16, format::bfuwzyx),
        std::make_tuple(data_types::i8, format::bfuwzyx),
        std::make_tuple(data_types::u8, format::bfuwzyx),
        std::make_tuple(data_types::i16, format::bfuwzyx),
        std::make_tuple(data_types::u16, format::bfuwzyx),
        std::make_tuple(data_types::u32, format::bfuwzyx),
        std::make_tuple(data_types::i32, format::bfuwzyx),
        std::make_tuple(data_types::i64, format::bfuwzyx),

        std::make_tuple(data_types::f32, format::bfvuwzyx),
        std::make_tuple(data_types::f16, format::bfvuwzyx),
        std::make_tuple(data_types::bf16, format::bfvuwzyx),
        std::make_tuple(data_types::i8, format::bfvuwzyx),
        std::make_tuple(data_types::u8, format::bfvuwzyx),
        std::make_tuple(data_types::i16, format::bfvuwzyx),
        std::make_tuple(data_types::u16, format::bfvuwzyx),
        std::make_tuple(data_types::u32, format::bfvuwzyx),
        std::make_tuple(data_types::i32, format::bfvuwzyx),
        std::make_tuple(data_types::i64, format::bfvuwzyx),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::bf16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i64, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::bf16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv32),
        std::make_tuple(data_types::bf16, format::bs_fs_zyx_bsv16_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv16_fsv32),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv16_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::bf16, format::bs_fs_yx_bsv16_fsv16),

        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv2),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv2),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv2),
        std::make_tuple(data_types::bf16, format::b_fs_zyx_fsv2),
        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv2),

        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv8_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv8_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv8_fsv2),
        std::make_tuple(data_types::bf16, format::bs_fs_zyx_bsv8_fsv2),
        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv8_fsv2),

        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv16_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv2),
        std::make_tuple(data_types::bf16, format::bs_fs_zyx_bsv16_fsv2),
        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv2),

        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::bf16, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),

        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::bf16, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::bf16, format::b_fs_zyx_fsv32),

        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),
        std::make_tuple(data_types::bf16, format::fs_b_yx_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::bf16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::bf16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv32_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv32),
        std::make_tuple(data_types::bf16, format::bs_fs_yx_bsv16_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv16_fsv32),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv16_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::bf16, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv4_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::bf16, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv8_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::bf16, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv8_fsv2),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::bf16, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv4_fsv2),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::bf16, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::bf16, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv32_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::eltwise_impl)
