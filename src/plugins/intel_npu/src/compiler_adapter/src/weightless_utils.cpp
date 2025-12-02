// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weightless_utils.hpp"

#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"

namespace intel_npu {

bool isInitMetadata(const NetworkMetadata& networkMetadata) {
    if (networkMetadata.inputs.size() == 0) {
        return false;
    }
    return networkMetadata.inputs.at(0).isInitInputWeights;
}

/**
 * @brief Stores the information within the "WeightlessCacheAttribute" as runtime fields that persist upon
 * serialization.
 * @details Constant nodes (weights) may contain as mesdatadata the "WeightlessCacheAttribute", that is information
 * regarding the offset of the weights within the binary file, as well as the original size and precision. This
 * information is required within the "weights separation" flow, therefore this function is here to store it.
 * @note Not calling this function in the weights separation flow would lead to this information being lost upon
 * serialization. The "WeightlessCacheAttribute" information that is populated upon de-serialization would represent
 * metadata corresponding to the serialized stream, not the original weights file. Therefore the compiler would be
 * misinformed and lookups of weights offsets could fail.
 *
 * @param model Both source and target.
 */
void storeWeightlessCacheAttribute(const std::shared_ptr<ov::Model>& model) {
    size_t constantId = 0;
    for (auto&& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            ov::RTMap& runtimeInfoMap = node->get_rt_info();
            const auto& weightlessCacheAttrIt =
                runtimeInfoMap.find(ov::WeightlessCacheAttribute::get_type_info_static());

            const std::string constantIdString = std::to_string(constantId++);
            if (weightlessCacheAttrIt != runtimeInfoMap.end()) {
                auto& weightlessCacheAttr = weightlessCacheAttrIt->second.as<ov::WeightlessCacheAttribute>();
                model->set_rt_info(weightlessCacheAttr.bin_offset, "ws_bin_offset_" + constantIdString);
                model->set_rt_info(weightlessCacheAttr.original_size, "ws_original_size_" + constantIdString);
                model->set_rt_info(weightlessCacheAttr.original_dtype, "ws_original_dtype_" + constantIdString);
            }
        }
    }
}
}  // namespace intel_npu
