// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
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
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),

        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),

        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
