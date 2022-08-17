// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "mvn/mvn_kernel_selector.h"
#include "mvn/mvn_kernel_base.h"

#include <algorithm>

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct mvn_impl : typed_primitive_impl_ocl<mvn> {
    using parent = typed_primitive_impl_ocl<mvn>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<mvn_impl>(*this);
    }

public:
    static primitive_impl* create(const mvn_node& arg) {
        auto mvn_params = get_default_params<kernel_selector::mvn_params>(arg);
        auto mvn_optional_params = get_default_optional_params<kernel_selector::mvn_optional_params>(arg.get_program());

        mvn_params.mvnMode = arg.get_primitive()->across_channels ? kernel_selector::mvn_mode::ACROSS_CHANNELS
                                                                  : kernel_selector::mvn_mode::WITHIN_CHANNELS;
        mvn_params.mvnNormalizeVariance = arg.get_primitive()->normalize_variance;
        mvn_params.epsilon = arg.get_primitive()->epsilon;

        mvn_params.mvnEpsMode = arg.get_primitive()->eps_inside_sqrt ? kernel_selector::mvn_eps_mode::INSIDE_SQRT
                                                                     : kernel_selector::mvn_eps_mode::OUTSIDE_SQRT;

        auto& kernel_selector = kernel_selector::mvn_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(mvn_params, mvn_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto mvn = new mvn_impl(arg, best_kernels[0]);

        return mvn;
    }
};

namespace detail {

attach_mvn_impl::attach_mvn_impl() {
    implementation_map<mvn>::add(impl_types::ocl, mvn_impl::create, {
        MAKE_TUPLE4(bfyx,                   f32, f16, u8, i8),
        MAKE_TUPLE4(bfzyx,                  f32, f16, u8, i8),
        MAKE_TUPLE2(byxf,                   f32, f16),
        MAKE_TUPLE2(yxfb,                   f32, f16),
        MAKE_TUPLE4(b_fs_yx_fsv16,          f32, f16, u8, i8),
        MAKE_TUPLE4(b_fs_zyx_fsv16,         f32, f16, u8, i8),
        MAKE_TUPLE1(bs_fs_yx_bsv32_fsv16,        f16),
        MAKE_TUPLE2(bs_fs_yx_bsv32_fsv32,             u8, i8),
        MAKE_TUPLE2(bs_fs_zyx_bsv16_fsv16,  f32, f16),

        // TODO: uncomment this code when fsv32 optimizations for MVN will be implemented
        // MAKE_TUPLE4(b_fs_yx_fsv32,          f32, f16, u8, i8),
        // MAKE_TUPLE4(b_fs_zyx_fsv32,         f32, f16, u8, i8),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
