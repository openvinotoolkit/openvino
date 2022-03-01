// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bucketize_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "bucketize/bucketize_kernel_selector.h"
#include "bucketize/bucketize_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {
struct bucketize_impl : typed_primitive_impl_ocl<bucketize> {
    using parent = typed_primitive_impl_ocl<bucketize>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<bucketize_impl>(*this);
    }

public:
    static primitive_impl* create(const bucketize_node& arg) {
        auto bucketize_params = get_default_params<kernel_selector::bucketize_params>(arg);
        auto bucketize_optional_params = get_default_optional_params<kernel_selector::bucketize_optional_params>(arg.get_program());

        bucketize_params.output_type = arg.get_primitive()->output_type;
        bucketize_params.with_right_bound = arg.get_primitive()->with_right_bound;

        auto& kernel_selector = kernel_selector::bucketize_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(bucketize_params, bucketize_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto bucketize = new bucketize_impl(arg, best_kernels[0]);

        return bucketize;
    }
};

namespace detail {

attach_bucketize_impl::attach_bucketize_impl() {
    implementation_map<bucketize>::add(impl_types::ocl, bucketize_impl::create, {
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
