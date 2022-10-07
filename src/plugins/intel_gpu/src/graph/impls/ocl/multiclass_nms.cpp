// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multiclass_nms_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "multiclass_nms/multiclass_nms_kernel_ref.h"
#include "multiclass_nms/multiclass_nms_kernel_selector.h"


namespace cldnn {
namespace ocl {
struct multiclass_nms_impl : public typed_primitive_impl_ocl<multiclass_nms> {
    using parent = typed_primitive_impl_ocl<multiclass_nms>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<multiclass_nms_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<multiclass_nms>& instance, int32_t unused) const override {
        kernel_arguments_data args = parent::get_arguments(instance, unused);
        args.inputs.push_back(instance.output_indices_memory());
        args.inputs.push_back(instance.output_num_memory());
        return args;
    }

public:
    static primitive_impl* create(const multiclass_nms_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::multiclass_nms_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::multiclass_nms_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();

        switch (primitive->sort_result) {
            case cldnn::sort_result_type::classid:
                params.sort_result_type = kernel_selector::SortResultType::CLASSID;
                break;
            case cldnn::sort_result_type::score:
                params.sort_result_type = kernel_selector::SortResultType::SCORE;
                break;
            case cldnn::sort_result_type::none:
                params.sort_result_type = kernel_selector::SortResultType::NONE;
                break;
            default:
                throw std::runtime_error{"Not supported sort result type"};
        }
        params.sort_result_across_batch = primitive->sort_result_across_batch;
        switch (primitive->indices_output_type) {
            case cldnn::data_types::i32: {
                params.indices_output_type = kernel_selector::Datatype::INT32;
                break;
            }
            case cldnn::data_types::i64: {
                params.indices_output_type = kernel_selector::Datatype::INT64;
                break;
            }
            default:
                throw std::runtime_error{"Not supported index element type"};
        }
        params.iou_threshold = primitive->iou_threshold;
        params.score_threshold = primitive->score_threshold;
        params.nms_top_k = primitive->nms_top_k;
        params.keep_top_k = primitive->keep_top_k;
        params.background_class = primitive->background_class;
        params.normalized = primitive->normalized;
        params.nms_eta = primitive->nms_eta;
        params.has_roisnum = arg.has_roisnum();

        params.inputs.push_back(convert_data_tensor(arg.scores().get_output_layout()));

        if (arg.has_roisnum()) {
            params.inputs.push_back(convert_data_tensor(arg.roisnum().get_output_layout()));
        }

        params.inputs.push_back(convert_data_tensor(arg.output_selected_indices().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.output_selected_num().get_output_layout()));

        const auto& kernel_selector = kernel_selector::multiclass_nms_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "best_kernels.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new multiclass_nms_impl(arg, best_kernels[0]);
    }
};

namespace detail {
attach_multiclass_nms_impl::attach_multiclass_nms_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i32, data_types::i64};
    auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
    };

    implementation_map<multiclass_nms>::add(impl_types::ocl, multiclass_nms_impl::create, types, formats);
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
