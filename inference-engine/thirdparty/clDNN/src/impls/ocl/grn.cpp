// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grn_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "grn/grn_kernel_selector.h"
#include "grn/grn_kernel_base.h"

#include <algorithm>

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct grn_impl : typed_primitive_impl_ocl<grn> {
    using parent = typed_primitive_impl_ocl<grn>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<grn_impl>(*this);
    }

public:
    static primitive_impl* create(const grn_node& arg) {
        auto grn_params = get_default_params<kernel_selector::grn_params>(arg);
        auto grn_optional_params = get_default_optional_params<kernel_selector::grn_optional_params>(arg.get_program());

        grn_params.bias = arg.get_primitive()->bias;

        auto& kernel_selector = kernel_selector::grn_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(grn_params, grn_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto grn = new grn_impl(arg, best_kernels[0]);

        return grn;
    }
};

namespace detail {

attach_grn_impl::attach_grn_impl() {
    implementation_map<grn>::add(impl_types::ocl, grn_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
