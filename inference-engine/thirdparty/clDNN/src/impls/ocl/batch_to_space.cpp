// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "batch_to_space/batch_to_space_kernel_selector.h"
#include "batch_to_space/batch_to_space_kernel_ref.h"
#include "cldnn/runtime/error_handler.hpp"
#include "data_inst.h"
#include <vector>

using namespace cldnn;

namespace cldnn {
namespace ocl {
struct batch_to_space_impl : typed_primitive_impl_ocl<batch_to_space> {
    using parent = typed_primitive_impl_ocl<batch_to_space>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<batch_to_space_impl>(*this);
    }

public:
    static primitive_impl* create(const batch_to_space_node& arg) {
        auto batch_to_space_params = get_default_params<kernel_selector::batch_to_space_params>(arg);
        auto batch_to_space_optional_params =
            get_default_optional_params<kernel_selector::batch_to_space_optional_params>(arg.get_program());

        auto primitive = arg.get_primitive();

        batch_to_space_params.block_shape = convert_dim_vector(primitive->block_shape);
        batch_to_space_params.crops_begin = convert_dim_vector(primitive->crops_begin);
        batch_to_space_params.crops_end = convert_dim_vector(primitive->crops_end);

        auto& kernel_selector = kernel_selector::batch_to_space_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(batch_to_space_params, batch_to_space_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto batch_to_space = new batch_to_space_impl(arg, best_kernels[0]);

        return batch_to_space;
    }
};

namespace detail {

attach_batch_to_space_impl::attach_batch_to_space_impl() {
    implementation_map<batch_to_space>::add(impl_types::ocl, batch_to_space_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
