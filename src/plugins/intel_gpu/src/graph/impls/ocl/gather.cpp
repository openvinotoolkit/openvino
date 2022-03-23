// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gather/gather_kernel_selector.h"
#include "gather/gather_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {
static kernel_selector::gather_axis convert_axis(int64_t axis, size_t rank) {
    if (axis == 0) {
        return kernel_selector::gather_axis::BATCH;
    } else if (axis == 1) {
        return kernel_selector::gather_axis::FEATURE;
    }

    if (rank <= 4) {
        switch (axis) {
            case 2: return kernel_selector::gather_axis::Y;
            case 3: return kernel_selector::gather_axis::X;
            case -1: return kernel_selector::gather_axis::Y;
            case -2: return kernel_selector::gather_axis::FEATURE;
            case -3: return kernel_selector::gather_axis::BATCH;
            default: IE_THROW() << "Unsupported gather axis: " << axis;
        }
    } else if (rank == 5) {
        switch (axis) {
            case 2: return kernel_selector::gather_axis::Z;
            case 3: return kernel_selector::gather_axis::Y;
            case 4: return kernel_selector::gather_axis::X;
            case -1: return kernel_selector::gather_axis::Y;
            case -2: return kernel_selector::gather_axis::Z;
            case -3: return kernel_selector::gather_axis::FEATURE;
            case -4: return kernel_selector::gather_axis::BATCH;
            default: IE_THROW() << "Unsupported gather axis: " << axis;
        }
    } else if (rank == 6) {
        switch (axis) {
            case 2: return kernel_selector::gather_axis::W;
            case 3: return kernel_selector::gather_axis::Z;
            case 4: return kernel_selector::gather_axis::Y;
            case 5: return kernel_selector::gather_axis::X;
            case -1: return kernel_selector::gather_axis::Y;
            case -2: return kernel_selector::gather_axis::Z;
            case -3: return kernel_selector::gather_axis::W;
            case -4: return kernel_selector::gather_axis::FEATURE;
            case -5: return kernel_selector::gather_axis::BATCH;
            default: IE_THROW() << "Unsupported gather axis: " << axis;
        }
    } else {
        IE_THROW() << "Unsupported gather axis: " << axis;
    }
}

struct gather_impl : typed_primitive_impl_ocl<gather> {
    using parent = typed_primitive_impl_ocl<gather>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_impl>(*this);
    }

public:
    static primitive_impl* create(const gather_node& arg) {
        const auto& prim = arg.get_primitive();
        const auto& param_info = kernel_impl_params(arg.get_program(), prim, arg.get_unique_id(),
                                                    arg.get_input_layouts(), arg.get_output_layout(),
                                                    arg.get_fused_primitives(),
                                                    arg.get_fused_activations_funcs(), arg.get_fused_activations_params());

        auto gather_params = get_default_params<kernel_selector::gather_params>(param_info);
        auto gather_optional_params =
            get_default_optional_params<kernel_selector::gather_optional_params>(arg.get_program());

        auto input_layout = arg.get_dependency(0).get_output_layout();
        gather_params.axis = convert_axis(prim->axis, input_layout.get_rank());
        gather_params.batch_dim = size_t(prim->batch_dim);
        gather_params.support_neg_ind = prim->support_neg_ind;

        gather_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::gather_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(gather_params, gather_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto gather = new gather_impl(arg, best_kernels[0]);

        return gather;
    }
};

namespace detail {

attach_gather_impl::attach_gather_impl() {
    implementation_map<gather>::add(impl_types::ocl, gather_impl::create, {
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
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
