// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "tile/tile_kernel_selector.h"
#include "tile/tile_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct tile_impl : typed_primitive_impl_ocl<tile> {
    using parent = typed_primitive_impl_ocl<tile>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<tile_impl>(*this);
    }

public:
    static primitive_impl* create(const tile_node& arg) {
        auto tile_params = get_default_params<kernel_selector::tile_params>(arg);
        auto tile_optional_params =
            get_default_optional_params<kernel_selector::tile_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::tile_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(tile_params, tile_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto tile = new tile_impl(arg, best_kernels[0]);

        return tile;
    }
};

namespace detail {

attach_tile_impl::attach_tile_impl() {
    implementation_map<tile>::add(impl_types::ocl, tile_impl::create, {
        MAKE_TUPLE5(bfyx,   f32, f16, u8, i8, i32),
        MAKE_TUPLE2(bfzyx,  f32, f16),
        MAKE_TUPLE5(bfwzyx, f32, f16, u8, i8, i32),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
