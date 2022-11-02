// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "data_inst.h"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "matrix_nms/matrix_nms_kernel_ref.h"
#include "matrix_nms/matrix_nms_kernel_selector.h"
#include "matrix_nms_inst.h"
#include "primitive_base.hpp"

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

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<matrix_nms_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(matrix_nms_inst& instance, int32_t) const override {
        kernel_arguments_data args;
        args.inputs = {instance.input_boxes_mem(),
                       instance.input_scores_mem(),
                       instance.input_selected_boxes_mem(),
                       instance.input_valid_outputs_mem()};
        args.outputs = {instance.output_memory_ptr()};

        return args;
    }

public:
    static primitive_impl* create(const matrix_nms_node& node, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::matrix_nms_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::matrix_nms_optional_params>(node.get_program());

        const auto& scores_layout = impl_param.get_input_layout(1);
        const auto& second_output_layout = impl_param.get_input_layout(2);
        const auto& third_output_layout = impl_param.get_input_layout(3);

        params.inputs.push_back(convert_data_tensor(scores_layout));
        params.inputs.push_back(convert_data_tensor(second_output_layout));
        params.inputs.push_back(convert_data_tensor(third_output_layout));

        const auto& primitive = node.get_primitive();
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

        auto& kernel_selector = kernel_selector::matrix_nms_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(node.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this nodeuments");

        auto matrix_nms_node = new matrix_nms_impl(node, best_kernels[0]);

        return matrix_nms_node;
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

    implementation_map<matrix_nms>::add(impl_types::ocl, matrix_nms_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
