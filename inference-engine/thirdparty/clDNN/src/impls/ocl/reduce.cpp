// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "reduce/reduce_kernel_selector.h"
#include "reduce/reduce_kernel_ref.h"
#include "reduce/reduce_kernel_b_fs_yx_fsv16.h"
#include "cldnn/runtime/error_handler.hpp"
#include "data_inst.h"

using namespace cldnn;

namespace cldnn {
namespace ocl {
namespace {
kernel_selector::reduce_mode cldnn_2_reduce_mode(reduce_mode mode) {
    switch (mode) {
        case reduce_mode::max:
            return kernel_selector::reduce_mode::MAX;
        case reduce_mode::min:
            return kernel_selector::reduce_mode::MIN;
        case reduce_mode::mean:
            return kernel_selector::reduce_mode::MEAN;
        case reduce_mode::prod:
            return kernel_selector::reduce_mode::PROD;
        case reduce_mode::sum:
            return kernel_selector::reduce_mode::SUM;
        case reduce_mode::logical_and:
            return kernel_selector::reduce_mode::AND;
        case reduce_mode::logical_or:
            return kernel_selector::reduce_mode::OR;
        case reduce_mode::sum_square:
            return kernel_selector::reduce_mode::SUM_SQUARE;
        case reduce_mode::l1:
            return kernel_selector::reduce_mode::L1;
        case reduce_mode::l2:
            return kernel_selector::reduce_mode::L2;
        case reduce_mode::log_sum:
            return kernel_selector::reduce_mode::LOG_SUM;
        case reduce_mode::log_sum_exp:
            return kernel_selector::reduce_mode::LOG_SUM_EXP;
        default:
            assert(0);
            return kernel_selector::reduce_mode::MAX;
    }
}
}  // namespace
struct reduce_impl : typed_primitive_impl_ocl<reduce> {
    using parent = typed_primitive_impl_ocl<reduce>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reduce_impl>(*this);
    }

public:
    static primitive_impl* create(const reduce_node& arg) {
        auto reduce_params = get_default_params<kernel_selector::reduce_params>(arg);
        auto reduce_optional_params = get_default_optional_params<kernel_selector::reduce_optional_params>(arg.get_program());

        reduce_params.reduceAxes = arg.get_primitive()->axes;
        reduce_params.keepDims = arg.get_primitive()->keep_dims;
        reduce_params.reduceMode = cldnn_2_reduce_mode(arg.get_primitive()->mode);

        auto& kernel_selector = kernel_selector::reduce_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(reduce_params, reduce_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto reduce = new reduce_impl(arg, best_kernels[0]);

        return reduce;
    }
};

namespace detail {

attach_reduce_impl::attach_reduce_impl() {
    implementation_map<reduce>::add(impl_types::ocl, reduce_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
