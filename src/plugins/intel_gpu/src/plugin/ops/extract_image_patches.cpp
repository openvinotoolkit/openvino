// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/extractimagepatches.hpp"

#include "intel_gpu/primitives/extract_image_patches.hpp"

namespace ov::intel_gpu {

static void CreateExtractImagePatchesOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::ExtractImagePatches>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto extractImagePatchesPrim = cldnn::extract_image_patches(layerName,
                                                                inputs[0],
                                                                op->get_sizes(),
                                                                op->get_strides(),
                                                                op->get_rates(),
                                                                op->get_auto_pad(),
                                                                tensor_from_dims(op->get_output_shape(0)));

    p.add_primitive(*op, extractImagePatchesPrim);
}

REGISTER_FACTORY_IMPL(v3, ExtractImagePatches);

}  // namespace ov::intel_gpu
