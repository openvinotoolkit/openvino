// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "matrix_nms_inst.h"
#include "matrix_nms/matrix_nms_kernel_ref.h"
#include "matrix_nms/matrix_nms_kernel_selector.h"

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::matrix_nms_params::decay_function from(matrix_nms::decay_function decay) {
    switch (decay) {
    case matrix_nms::decay_function::gaussian:
        return kernel_selector::matrix_nms_params::decay_function::GAUSSIAN;
    default:
    case matrix_nms::decay_function::linear:
        return kernel_selector::matrix_nms_params::decay_function::LINEAR;
    }
}

kernel_selector::matrix_nms_params::sort_result_type from(matrix_nms::sort_result_type type) {
    switch (type) {
    case matrix_nms::sort_result_type::class_id:
        return kernel_selector::matrix_nms_params::sort_result_type::CLASS_ID;
    case matrix_nms::sort_result_type::score:
        return kernel_selector::matrix_nms_params::sort_result_type::SCORE;
    default:
    case matrix_nms::sort_result_type::none:
        return kernel_selector::matrix_nms_params::sort_result_type::NONE;
    }
}
}  // namespace

struct matrix_nms_impl : typed_primitive_impl_ocl<matrix_nms> {
    using parent = typed_primitive_impl_ocl<matrix_nms>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::matrix_nms_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::matrix_nms_params, kernel_selector::matrix_nms_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<matrix_nms_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const matrix_nms_inst& instance) const override {
        kernel_arguments_data args;
        args.inputs = {instance.input_boxes_mem(),
                       instance.input_scores_mem(),
                       instance.input_selected_boxes_mem(),
                       instance.input_valid_outputs_mem()};
        args.outputs = {instance.output_memory_ptr()};

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<matrix_nms>();
        auto params = get_default_params<kernel_selector::matrix_nms_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::matrix_nms_optional_params>(impl_param.get_program());

        const auto& scores_layout = impl_param.get_input_layout(1);
        const auto& second_output_layout = impl_param.get_input_layout(2);
        const auto& third_output_layout = impl_param.get_input_layout(3);

        params.inputs.push_back(convert_data_tensor(scores_layout));
        params.inputs.push_back(convert_data_tensor(second_output_layout));
        params.inputs.push_back(convert_data_tensor(third_output_layout));

        params.sort_type = from(primitive->attribs.sort_type);
        params.sort_result_across_batch = primitive->attribs.sort_result_across_batch;
        params.score_threshold = primitive->attribs.score_threshold;
        params.nms_top_k = primitive->attribs.nms_top_k;
        params.keep_top_k = primitive->attribs.keep_top_k;
        params.background_class = primitive->attribs.background_class;
        params.decay = from(primitive->attribs.decay);
        params.gaussian_sigma = primitive->attribs.gaussian_sigma;
        params.post_threshold = primitive->attribs.post_threshold;
        params.normalized = primitive->attribs.normalized;

        return {params, optional_params};
    }
};

namespace detail {

attach_matrix_nms_impl::attach_matrix_nms_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i32};

    auto formats = {format::bfyx,
                    format::b_fs_yx_fsv16,
                    format::b_fs_yx_fsv32,
                    format::bs_fs_yx_bsv16_fsv16,
                    format::bs_fs_yx_bsv32_fsv16,
                    format::bs_fs_yx_bsv32_fsv32};

    implementation_map<matrix_nms>::add(impl_types::ocl, typed_primitive_impl_ocl<matrix_nms>::create<matrix_nms_impl>, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::matrix_nms_impl)
