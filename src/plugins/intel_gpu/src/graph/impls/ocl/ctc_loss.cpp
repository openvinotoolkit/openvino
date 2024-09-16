// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "kernel_selector_params.h"
#include "jitter.h"
#include "primitive_ocl_base.hpp"

#include "ctc_loss.hpp"
#include "ctc_loss_inst.hpp"
#include "kernel_base.hpp"

namespace cldnn {
namespace ocl {

namespace {

using namespace ov::intel_gpu::ocl;


class CTCLossGenerator : public ov::intel_gpu::ocl::KernelGeneratorBase {
public:
    CTCLossGenerator() : KernelGeneratorBase("ctc_loss_ref") {}
    KernelData get_kernel_data(const program_node& node, const kernel_impl_params& params) const override {
        KernelData kd;
        auto kernel_str = std::make_shared<KernelString>();
        auto entry_point = get_entry_point(node, params);
        kernel_str->entry_point = entry_point;
        kernel_str->jit = "";
        kernel_str->undefs = "";
        kernel_str->options = "";
        kernel_str->batch_compilation = false;
        kernel_str->has_microkernels = false;
        kernel_str->str = build_code(get_name(), get_jit_constants(node, params), entry_point);
        kd.code.kernelString = kernel_str;
        kd.params.workGroups = get_dispatch_data(node, params);
        kd.params.arguments = get_arguments_desc(node, params);

        return kd;
    }

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = make_base_jit_constants(node, params);
        const auto& desc = node.as<ctc_loss>().get_primitive();

        jit_constants.add({
            make_jit_constant("PREPROCESS_COLLAPSE_REPEATED", desc->preprocess_collapse_repeated),
            make_jit_constant("CTC_MERGE_REPEATED", desc->ctc_merge_repeated),
            make_jit_constant("UNIQUE", desc->unique),
        });

        return jit_constants;
    }

    std::string get_entry_point(const program_node& node, const kernel_impl_params& params) const override {
        std::string entry_point = get_name();

        entry_point += "_" + std::to_string(params.hash());
        entry_point += "__sa";

        return entry_point;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;

        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        return args;
    }

    WorkGroupSizes get_dispatch_data(const program_node& node, const kernel_impl_params& params) const override {
        WorkGroupSizes dispatch_data;
        const auto& output = params.output_layouts[0];

        dispatch_data.global = {output.get_shape()[0], 1, 1};
        dispatch_data.local = {1, 1, 1}; /*GetOptimalLocalWorkGroupSizes(dispatch_data.gws, kernel_params.engineInfo)*/;
        return dispatch_data;
    }
};

}  // namespace

struct ctc_loss_impl : typed_primitive_impl_ocl_new<ctc_loss> {
    using parent = typed_primitive_impl_ocl_new<ctc_loss>;
    using parent::parent;

    ctc_loss_impl(const KernelData& kd, const std::string& impl_name) : parent({ kd }, impl_name) {}

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::ctc_loss_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<ctc_loss_impl>(*this);
    }
};

std::unique_ptr<primitive_impl> CTCLoss::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<ctc_loss>());
    CTCLossGenerator gen;
    auto kd = gen.get_kernel_data(node, params);
    return cldnn::make_unique<ctc_loss_impl>(kd, std::string(get_type_info().name) + "::" + gen.get_name());
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::ctc_loss_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ctc_loss)
