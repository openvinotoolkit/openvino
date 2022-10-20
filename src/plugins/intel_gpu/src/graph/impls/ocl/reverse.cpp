// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "primitive_base.hpp"
#include "reverse/reverse_kernel_ref.h"
#include "reverse/reverse_kernel_selector.h"
#include "reverse_inst.h"
#include "impls/implementation_map.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct reverse_impl : typed_primitive_impl_ocl<reverse> {
    using parent = typed_primitive_impl_ocl<reverse>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reverse_impl>(*this);
    }

public:
    static primitive_impl* create(const reverse_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::reverse_params>(impl_param);
        const auto optional_params =
            get_default_optional_params<kernel_selector::reverse_optional_params>(arg.get_program());

        params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));
        params.reverseMode = arg.get_primitive()->mode == reverse_mode::index ? kernel_selector::reverse_mode::index
                                                                              : kernel_selector::reverse_mode::mask;

        const auto& kernel_selector = kernel_selector::reverse_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto reverse = new reverse_impl(arg, best_kernels[0]);

        return reverse;
    }
};

namespace detail {

attach_reverse_impl::attach_reverse_impl() {
    static const auto types =
        {data_types::f16, data_types::f32, data_types::i8, data_types::u8, data_types::i32, data_types::i64};
    static const auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv32_fsv16,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,

        format::bfwzyx,
    };
    std::set<std::tuple<data_types, format::type>> keys;
    for (const auto t : types) {
        for (const auto f : formats) {
            keys.emplace(t, f);
        }
    }
    implementation_map<reverse>::add(impl_types::ocl, reverse_impl::create, keys);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
