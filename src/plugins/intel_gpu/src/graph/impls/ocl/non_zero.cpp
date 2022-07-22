// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "non_zero_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "non_zero/count_nonzero_kernel_ref.h"
#include "non_zero/count_nonzero_kernel_selector.h"
#include "non_zero/gather_nonzero_kernel_ref.h"
#include "non_zero/gather_nonzero_kernel_selector.h"
#include "intel_gpu/runtime/error_handler.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct count_nonzero_impl : typed_primitive_impl_ocl<count_nonzero> {
    using parent = typed_primitive_impl_ocl<count_nonzero>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<count_nonzero_impl>(*this);
    }

    static primitive_impl* create(const count_nonzero_node& arg, const kernel_impl_params& impl_param) {
        auto nonzero_params = get_default_params<kernel_selector::count_nonzero_params>(impl_param);
        auto nonzero_optional_params =
            get_default_optional_params<kernel_selector::count_nonzero_optional_params>(arg.get_program());

        nonzero_params.ov_input_rank = impl_param.get_input_layout().get_shape().size();

        auto& kernel_selector = kernel_selector::count_nonzero_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(nonzero_params, nonzero_optional_params);

        OPENVINO_ASSERT(!best_kernels.empty(), "Cannot find a proper kernel for ", arg.id());

        auto count_nonzero = new count_nonzero_impl(arg, best_kernels[0]);

        return count_nonzero;
    }
};

struct gather_nonzero_impl : typed_primitive_impl_ocl<gather_nonzero> {
    using parent = typed_primitive_impl_ocl<gather_nonzero>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_nonzero_impl>(*this);
    }

public:
    static primitive_impl* create(const gather_nonzero_node& arg, const kernel_impl_params& impl_param) {
        auto nonzero_params = get_default_params<kernel_selector::gather_nonzero_params>(impl_param);
        auto nonzero_optional_params =
            get_default_optional_params<kernel_selector::gather_nonzero_optional_params>(arg.get_program());

        nonzero_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));
        nonzero_params.ov_input_rank = impl_param.get_input_layout().get_shape().size();

        auto& kernel_selector = kernel_selector::gather_nonzero_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(nonzero_params, nonzero_optional_params);

        OPENVINO_ASSERT(!best_kernels.empty(), "Cannot find a proper kernel for ", arg.id());

        auto gather_nonzero = new gather_nonzero_impl(arg, best_kernels[0]);

        return gather_nonzero;
    }
};

namespace detail {

attach_count_nonzero_impl::attach_count_nonzero_impl() {
    auto types = {data_types::u8, data_types::i8, data_types::f16, data_types::f32, data_types::i32};
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    implementation_map<count_nonzero>::add(impl_types::ocl, count_nonzero_impl::create, types, formats);
}

attach_gather_nonzero_impl::attach_gather_nonzero_impl() {
    auto types = {data_types::u8, data_types::i8, data_types::f16, data_types::f32, data_types::i32};
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    implementation_map<gather_nonzero>::add(impl_types::ocl, gather_nonzero_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
