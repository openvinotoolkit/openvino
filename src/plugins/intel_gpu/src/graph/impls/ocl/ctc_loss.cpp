// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "ctc_loss_inst.hpp"
#include "ctc_loss/ctc_loss_kernel_ref.hpp"
#include "ctc_loss/ctc_loss_kernel_selector.hpp"

namespace cldnn {
namespace ocl {

struct ctc_loss_impl : typed_primitive_impl_ocl<ctc_loss> {
    using parent = typed_primitive_impl_ocl<ctc_loss>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::ctc_loss_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::ctc_loss_params, kernel_selector::ctc_loss_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::ctc_loss_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<ctc_loss_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<ctc_loss>();
        auto params = get_default_params<kernel_selector::ctc_loss_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::ctc_loss_optional_params>(impl_param.get_program());

        params.preprocess_collapse_repeated = primitive->preprocess_collapse_repeated;
        params.ctc_merge_repeated = primitive->ctc_merge_repeated;
        params.unique = primitive->unique;
        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }
        return {params, optional_params};
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

    implementation_map<ctc_loss>::add(impl_types::ocl, typed_primitive_impl_ocl<ctc_loss>::create<ctc_loss_impl>, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::ctc_loss_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ctc_loss)
