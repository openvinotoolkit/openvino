// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "matrix_nms_inst.h"
#include "matrix_nms/matrix_nms_kernel_ref.h"
#include "matrix_nms/matrix_nms_kernel_selector.h"

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::matrix_nms_params::decay_function from(ov::op::v8::MatrixNms::DecayFunction decay) {
    switch (decay) {
    case ov::op::v8::MatrixNms::DecayFunction::GAUSSIAN:
        return kernel_selector::matrix_nms_params::decay_function::GAUSSIAN;
    default:
    case ov::op::v8::MatrixNms::DecayFunction::LINEAR:
        return kernel_selector::matrix_nms_params::decay_function::LINEAR;
    }
}

kernel_selector::matrix_nms_params::sort_result_type from(ov::op::v8::MatrixNms::SortResultType type) {
    switch (type) {
    case ov::op::v8::MatrixNms::SortResultType::CLASSID:
        return kernel_selector::matrix_nms_params::sort_result_type::CLASS_ID;
    case ov::op::v8::MatrixNms::SortResultType::SCORE:
        return kernel_selector::matrix_nms_params::sort_result_type::SCORE;
    default:
    case ov::op::v8::MatrixNms::SortResultType::NONE:
        return kernel_selector::matrix_nms_params::sort_result_type::NONE;
    }
}
}  // namespace

struct matrix_nms_impl : typed_primitive_impl_ocl<matrix_nms> {
    using parent = typed_primitive_impl_ocl<matrix_nms>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::matrix_nms_kernel_selector;
    using kernel_params_t = kernel_selector::matrix_nms_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::matrix_nms_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<matrix_nms_impl, kernel_params_t>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const matrix_nms_inst& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        // Legacy multi-output
        if (instance.desc()->num_outputs == 1) {
            args.outputs.push_back(instance.input_selected_boxes_mem());
            args.outputs.push_back(instance.input_valid_outputs_mem());
        }

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<matrix_nms>();
        auto params = get_default_params<kernel_selector::matrix_nms_params>(impl_param);

        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));

        if (primitive->num_outputs == 3) {
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[1]));
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[2]));
        } else {
            // Legacy multi-output
            params.outputs.push_back(convert_data_tensor(impl_param.get_input_layout(2)));
            params.outputs.push_back(convert_data_tensor(impl_param.get_input_layout(3)));
        }

        params.sort_type = from(primitive->attribs.sort_result_type);
        params.sort_result_across_batch = primitive->attribs.sort_result_across_batch;
        params.score_threshold = primitive->attribs.score_threshold;
        params.nms_top_k = primitive->attribs.nms_top_k;
        params.keep_top_k = primitive->attribs.keep_top_k;
        params.background_class = primitive->attribs.background_class;
        params.decay = from(primitive->attribs.decay_function);
        params.gaussian_sigma = primitive->attribs.gaussian_sigma;
        params.post_threshold = primitive->attribs.post_threshold;
        params.normalized = primitive->attribs.normalized;

        return params;
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
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::matrix_nms)
