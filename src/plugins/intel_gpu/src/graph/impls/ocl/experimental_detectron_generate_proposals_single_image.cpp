// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "experimental_detectron_generate_proposals_single_image_inst.hpp"
#include "ed_gpsi/generate_proposals_single_image_kernel_selector.h"
#include "ed_gpsi/generate_proposals_single_image_kernel_ref.h"

namespace cldnn {
namespace ocl {
struct experimental_detectron_generate_proposals_single_image_impl
        : public typed_primitive_impl_ocl<experimental_detectron_generate_proposals_single_image> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_generate_proposals_single_image>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::experimental_detectron_generate_proposals_single_image_kernel_selector;
    using kernel_params_t = kernel_selector::experimental_detectron_generate_proposals_single_image_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::experimental_detectron_generate_proposals_single_image_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<experimental_detectron_generate_proposals_single_image_impl, kernel_params_t>(*this);
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<experimental_detectron_generate_proposals_single_image>();
        auto params = get_default_params<kernel_selector::experimental_detectron_generate_proposals_single_image_params>(impl_param);

        params.min_size = primitive->min_size;
        params.nms_threshold  = primitive->nms_threshold;
        params.pre_nms_count = primitive->pre_nms_count;
        params.post_nms_count = primitive->post_nms_count;

        const size_t num_inputs = primitive->input_size();
        for (size_t i = 1; i < num_inputs; i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[1]));

        return params;
    }
};

namespace detail {
attach_experimental_detectron_generate_proposals_single_image_impl::attach_experimental_detectron_generate_proposals_single_image_impl() {
    auto types = {data_types::f16, data_types::f32};
    auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32
    };

    implementation_map<experimental_detectron_generate_proposals_single_image>::add(
        impl_types::ocl,
        typed_primitive_impl_ocl<experimental_detectron_generate_proposals_single_image>::create<experimental_detectron_generate_proposals_single_image_impl>,
        types, formats);
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::experimental_detectron_generate_proposals_single_image_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::experimental_detectron_generate_proposals_single_image)
