// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ctc_loss/ctc_loss_kernel_ref.hpp"
#include "ctc_loss/ctc_loss_kernel_selector.hpp"
#include "ctc_loss_inst.hpp"
#include "impls/implementation_map.hpp"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct ctc_loss_impl : typed_primitive_impl_ocl<ctc_loss> {
    using parent = typed_primitive_impl_ocl<ctc_loss>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<ctc_loss_impl>(*this);
    }

    static primitive_impl* create(const ctc_loss_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::ctc_loss_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::ctc_loss_optional_params>(arg.get_program());

        const auto& primitive = impl_param.typed_desc<ctc_loss>();
        params.preprocess_collapse_repeated = primitive->preprocess_collapse_repeated;
        params.ctc_merge_repeated = primitive->ctc_merge_repeated;
        params.unique = primitive->unique;
        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[i]));
        }

        const auto& kernel_selector = kernel_selector::ctc_loss_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new ctc_loss_impl(arg, best_kernels.front());
    }
};

namespace detail {

attach_ctc_loss_impl::attach_ctc_loss_impl() {
    auto types = {data_types::f16, data_types::f32};

    auto formats = {format::bfyx,
                    format::b_fs_yx_fsv16,
                    format::b_fs_yx_fsv32,
                    format::bs_fs_yx_bsv16_fsv16,
                    format::bs_fs_yx_bsv32_fsv32,
                    format::bs_fs_yx_bsv32_fsv16};

    implementation_map<ctc_loss>::add(impl_types::ocl, ctc_loss_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
