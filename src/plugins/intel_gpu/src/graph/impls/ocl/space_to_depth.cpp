// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_depth_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "space_to_depth/space_to_depth_kernel_selector.h"
#include "space_to_depth/space_to_depth_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {
struct space_to_depth_impl : typed_primitive_impl_ocl<space_to_depth> {
    using parent = typed_primitive_impl_ocl<space_to_depth>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<space_to_depth_impl>(*this);
    }

public:
    static primitive_impl* create(const space_to_depth_node& arg) {
        const auto& prim = arg.get_primitive();
        const auto& param_info = kernel_impl_params(arg.get_program(), prim, arg.get_unique_id(),
                                                    arg.get_input_layouts(), arg.get_output_layout(),
                                                    arg.get_fused_primitives(),
                                                    arg.get_fused_activations_funcs(), arg.get_fused_activations_params());
        auto space_to_depth_params = get_default_params<kernel_selector::space_to_depth_params>(param_info);
        auto space_to_depth_optional_params =
                get_default_optional_params<kernel_selector::space_to_depth_optional_params>(arg.get_program());

        space_to_depth_params.depth_mode = (prim->mode == space_to_depth::blocks_first) ?
                                           kernel_selector::SpaceToDepthMode::BLOCKS_FIRST :
                                           kernel_selector::SpaceToDepthMode::DEPTH_FIRST;

        space_to_depth_params.block_size = prim->block_size;

        auto& kernel_selector = kernel_selector::space_to_depth_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(space_to_depth_params, space_to_depth_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto space_to_depth = new space_to_depth_impl(arg, best_kernels[0]);

        return space_to_depth;
    }
};

namespace detail {

attach_space_to_depth_impl::attach_space_to_depth_impl() {
    implementation_map<space_to_depth>::add(impl_types::ocl, space_to_depth_impl::create, {
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
