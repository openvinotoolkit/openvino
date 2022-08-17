// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_batch_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "space_to_batch/space_to_batch_kernel_selector.h"
#include "space_to_batch/space_to_batch_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "data_inst.h"
#include <vector>

using namespace cldnn;

namespace cldnn {
namespace ocl {
struct space_to_batch_impl : typed_primitive_impl_ocl<space_to_batch> {
    using parent = typed_primitive_impl_ocl<space_to_batch>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<space_to_batch_impl>(*this);
    }

public:
    static primitive_impl* create(const space_to_batch_node& arg) {
        auto space_to_batch_params = get_default_params<kernel_selector::space_to_batch_params>(arg);
        auto space_to_batch_optional_params =
            get_default_optional_params<kernel_selector::space_to_batch_optional_params>(arg.get_program());

        auto primitive = arg.get_primitive();

        space_to_batch_params.block_shape = convert_dim_vector(primitive->block_shape);
        space_to_batch_params.pads_begin = convert_dim_vector(primitive->pads_begin);
        space_to_batch_params.pads_end = convert_dim_vector(primitive->pads_end);

        auto& kernel_selector = kernel_selector::space_to_batch_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(space_to_batch_params, space_to_batch_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto space_to_batch = new space_to_batch_impl(arg, best_kernels[0]);

        return space_to_batch;
    }
};

namespace detail {

attach_space_to_batch_impl::attach_space_to_batch_impl() {
    implementation_map<space_to_batch>::add(impl_types::ocl, space_to_batch_impl::create, {
        MAKE_TUPLE4(bfwzyx,         f32, f16, u8, i8),
        MAKE_TUPLE4(bfyx,           f32, f16, u8, i8),
        MAKE_TUPLE4(bfzyx,          f32, f16, u8, i8),
        MAKE_TUPLE4(b_fs_zyx_fsv16, f32, f16, u8, i8),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
