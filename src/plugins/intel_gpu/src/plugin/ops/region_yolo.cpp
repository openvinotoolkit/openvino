// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/region_yolo.hpp"

#include "intel_gpu/primitives/region_yolo.hpp"

namespace ov::intel_gpu {

static void CreateRegionYoloOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::RegionYolo>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    uint32_t coords = static_cast<uint32_t>(op->get_num_coords());
    uint32_t classes = static_cast<uint32_t>(op->get_num_classes());
    uint32_t num = static_cast<uint32_t>(op->get_num_regions());
    bool do_softmax = op->get_do_softmax();
    std::vector<int64_t> mask = op->get_mask();
    uint32_t mask_size = static_cast<uint32_t>(mask.size());
    int32_t axis = op->get_axis();
    int32_t end_axis = op->get_end_axis();

    auto regionPrim = cldnn::region_yolo(layerName,
                                         inputs[0],
                                         coords,
                                         classes,
                                         num,
                                         mask,
                                         mask_size,
                                         axis,
                                         end_axis,
                                         do_softmax);

    p.add_primitive(*op, regionPrim);
}

REGISTER_FACTORY_IMPL(v0, RegionYolo);

}  // namespace ov::intel_gpu
