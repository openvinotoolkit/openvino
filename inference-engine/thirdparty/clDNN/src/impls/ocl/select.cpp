// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "select/select_kernel_selector.h"
#include "select/select_kernel_base.h"

namespace cldnn {
namespace ocl {

struct select_impl : typed_primitive_impl_ocl<select> {
    using parent = typed_primitive_impl_ocl<select>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<select_impl>(*this);
    }

public:
    static primitive_impl* create(const select_node& arg) {
        auto select_params = get_default_params<kernel_selector::select_params>(arg);
        auto select_optional_params =
            get_default_optional_params<kernel_selector::select_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.inputs_count(); i++) {
            select_params.inputs.push_back(convert_data_tensor(arg.input(i).get_output_layout()));
        }

        auto& kernel_selector = kernel_selector::select_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(select_params, select_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto select = new select_impl(arg, best_kernels[0]);

        return select;
    }
};

namespace detail {

attach_select_impl::attach_select_impl() {
    implementation_map<select>::add(impl_types::ocl, select_impl::create, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
