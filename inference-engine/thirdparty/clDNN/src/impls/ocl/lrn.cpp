// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lrn_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "lrn/lrn_kernel_selector.h"
#include "lrn/lrn_kernel_base.h"

namespace cldnn {
namespace ocl {

struct lrn_impl : typed_primitive_impl_ocl<lrn> {
    using parent = typed_primitive_impl_ocl<lrn>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<lrn_impl>(*this);
    }

    static primitive_impl* create(const lrn_node& arg) {
        auto lrn_params = get_default_params<kernel_selector::lrn_params>(arg);
        auto lrn_optional_params = get_default_optional_params<kernel_selector::lrn_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();

        lrn_params.alpha = primitive->alpha;
        lrn_params.beta = primitive->beta;
        lrn_params.k = primitive->k;
        lrn_params.localSize = primitive->size;
        lrn_params.divMode = kernel_selector::kernel_divider_mode::FIXED;
        lrn_params.normMode = primitive->norm_region == lrn_norm_region_within_channel
                                  ? kernel_selector::lrn_mode::WITHIN_CHANNEL
                                  : kernel_selector::lrn_mode::ACROSS_CHANNEL;

        auto& kernel_selector = kernel_selector::lrn_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(lrn_params, lrn_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto lrn = new lrn_impl(arg, best_kernels[0]);

        return lrn;
    }
};

namespace detail {

attach_lrn_impl::attach_lrn_impl() {
    implementation_map<lrn>::add(impl_types::ocl, lrn_impl::create, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
