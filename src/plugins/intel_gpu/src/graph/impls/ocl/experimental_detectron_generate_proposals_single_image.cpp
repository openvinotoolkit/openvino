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
    using kernel_params_t = std::pair<kernel_selector::experimental_detectron_generate_proposals_single_image_params,
                                      kernel_selector::experimental_detectron_generate_proposals_single_image_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<experimental_detectron_generate_proposals_single_image_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<experimental_detectron_generate_proposals_single_image>& instance) const override {
        kernel_arguments_data args;
        const auto num_inputs = instance.inputs_memory_count();
        for (size_t i = 0; i < num_inputs; ++i) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        args.outputs.push_back(instance.output_memory_ptr());
        //TODO: Future improvement: To add second output parameter only when it's needed
        args.inputs.push_back(instance.output_roi_scores_memory());

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<experimental_detectron_generate_proposals_single_image>();
        auto params = get_default_params<kernel_selector::experimental_detectron_generate_proposals_single_image_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::experimental_detectron_generate_proposals_single_image_optional_params>(impl_param.get_program());

        params.min_size = primitive->min_size;
        params.nms_threshold  = primitive->nms_threshold;
        params.pre_nms_count = primitive->pre_nms_count;
        params.post_nms_count = primitive->post_nms_count;

        for (size_t i = 1; i < impl_param.input_layouts.size(); i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        return {params, optional_params};
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
