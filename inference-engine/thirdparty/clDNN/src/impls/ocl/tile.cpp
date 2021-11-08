// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "tile/tile_kernel_selector.h"
#include "tile/tile_kernel_ref.h"
#include "cldnn/runtime/error_handler.hpp"

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
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
