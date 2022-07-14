// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/plugin/common_utils.hpp>
#include <intel_gpu/plugin/program.hpp>
#include <intel_gpu/primitives/experimental_detectron_prior_grid_generator.hpp>
#include <ngraph/op/experimental_detectron_prior_grid_generator.hpp>

namespace ov {
namespace intel_gpu {
namespace {
cldnn::tensor mkTensor(const ov::Shape& shape) {
    if (shape.size() == 4)
        return cldnn::tensor{cldnn::batch(shape[0]), cldnn::feature(shape[1]), cldnn::spatial(4, shape[2])};
    else
        return cldnn::tensor{cldnn::spatial(4, shape[0])};
}

static void CreateExperimentalDetectronPriorGridGeneratorOp(
    Program& p,
    const std::shared_ptr<ngraph::op::v6::ExperimentalDetectronPriorGridGenerator>& op) {
    p.ValidateInputs(op, {3});
    cldnn::tensor outTensor = mkTensor(op->get_output_shape(0));
    auto outDataType = DataTypeFromPrecision(op->get_output_element_type(0));
    cldnn::layout outLayout{outDataType, cldnn::format::bfyx, outTensor};
    auto& attrs = op->get_attrs();
    auto& featmap_shape = op->get_input_shape(1);
    auto& image_shape = op->get_input_shape(2);
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    inputPrimitives.resize(1);  // only priors is read
    cldnn::experimental_detectron_prior_grid_generator prim{layer_type_name_ID(op),
                                                            inputPrimitives,
                                                            attrs.flatten,
                                                            static_cast<uint64_t>(attrs.h),
                                                            static_cast<uint64_t>(attrs.w),
                                                            attrs.stride_x,
                                                            attrs.stride_y,
                                                            featmap_shape[2],
                                                            featmap_shape[3],
                                                            image_shape[2],
                                                            image_shape[3],
                                                            op->get_friendly_name()};
    p.AddPrimitive(prim);
    p.AddPrimitiveToProfiler(op);
}
}  // namespace

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronPriorGridGenerator);

}  // namespace intel_gpu
}  // namespace ov
