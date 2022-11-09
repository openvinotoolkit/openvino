// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generate_proposals_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "generate_proposals/generate_proposals_kernel_selector.h"
#include "generate_proposals/generate_proposals_kernel_ref.h"



namespace cldnn {
namespace ocl {
struct generate_proposals_impl
        : public typed_primitive_impl_ocl<generate_proposals> {
    using parent = typed_primitive_impl_ocl<generate_proposals>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<generate_proposals_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<generate_proposals>& instance, int32_t) const override {
        auto args = parent::get_arguments(instance, 0);
        args.inputs.push_back(instance.output_rois_scores_memory());
        args.inputs.push_back(instance.output_rois_nums_memory());
        return args;
    }

public:
    static primitive_impl* create(const generate_proposals_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::generate_proposals_params>(impl_param);
        auto optional_params = get_default_optional_params<
                kernel_selector::generate_proposals_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();

        params.min_size = primitive->min_size;
        params.nms_threshold  = primitive->nms_threshold;
        params.pre_nms_count = primitive->pre_nms_count;
        params.post_nms_count = primitive->post_nms_count;
        params.normalized = primitive->normalized;
        params.nms_eta = primitive->nms_eta;
        params.roi_num_type = primitive->roi_num_type == cldnn::data_types::i32 ?
                kernel_selector::Datatype::INT32 : kernel_selector::Datatype::INT64;

        params.inputs.push_back(convert_data_tensor(arg.anchors().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.deltas().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.scores().get_output_layout()));

        params.inputs.push_back(convert_data_tensor(arg.output_rois_scores_node().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.output_rois_nums_node().get_output_layout()));

        const auto& kernel_selector = kernel_selector::generate_proposals_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "best_kernels.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new generate_proposals_impl(arg, best_kernels[0]);
    }
};

namespace detail {
    attach_generate_proposals_impl::attach_generate_proposals_impl() {
        implementation_map<generate_proposals>::add(impl_types::ocl,
                                                    generate_proposals_impl::create, {
                                                            std::make_tuple(data_types::f16, format::bfyx),
                                                            std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
                                                            std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
                                                            std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
                                                            std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
                                                            std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),

                                                            std::make_tuple(data_types::f32, format::bfyx),
                                                            std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
                                                            std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
                                                            std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
                                                            std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
                                                            std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32)});
    }
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
