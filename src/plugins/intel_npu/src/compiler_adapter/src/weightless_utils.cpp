// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weightless_utils.hpp"

#include "openvino/core/memory_util.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace intel_npu {

bool isInitMetadata(const NetworkMetadata& networkMetadata) {
    if (networkMetadata.inputs.size() == 0) {
        return false;
    }
    return networkMetadata.inputs.at(0).isInitInputWeights;
}

std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> get_all_constants_in_topological_order(
    const std::shared_ptr<const ov::Model>& model) {
    std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> constants;

    // Match the inputs of the "init" model with the Constant nodes of the original model
    for (auto&& node : model->get_ops()) {
        if (!ov::is_type<ov::op::v0::Constant>(node)) {
            continue;
        }

        auto constantNode = std::static_pointer_cast<ov::op::v0::Constant>(node);
        ov::RTMap& runtimeInfoMap = constantNode->get_rt_info();
        const auto& weightlessCacheAttrIt = runtimeInfoMap.find(ov::WeightlessCacheAttribute::get_type_info_static());
        if (weightlessCacheAttrIt != runtimeInfoMap.end()) {
            auto& weightlessCacheAttr = weightlessCacheAttrIt->second.as<ov::WeightlessCacheAttribute>();

            auto& constant = constants[weightlessCacheAttr.bin_offset];
            if (constant != nullptr) {
                // if multiple constants point to the same buffer, ensure that
                // their binary sizes are the same
                OPENVINO_ASSERT(constant->get_byte_size() == constantNode->get_byte_size(),
                                "Found ov::Constant that points to the common buffer but has mismatching byte size. "
                                "This may indicate a bug in OV model compression.");
                continue;
            }
            constant = std::move(constantNode);
        }
    }

    return constants;
}

std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> get_all_constants_memory_mapped(
    const std::string& weightsPath,
    const std::vector<NetworkMetadata>& initNetworkMetadata) {
    std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> constants;

    auto mapped_memory = ov::load_mmap_object(weightsPath);
    for (const auto& initMetadata : initNetworkMetadata) {
        for (const IODescriptor& descriptor : initMetadata.inputs) {
            const auto& opt = ov::util::view_to_number<size_t>(descriptor.nameFromCompiler);
            OPENVINO_ASSERT(opt.has_value(), "Failed parse id for constant: ", descriptor.nameFromCompiler);

            const size_t id = opt.value();
            const size_t byte_size =
                ov::util::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
            OPENVINO_ASSERT(id <= mapped_memory->size() && byte_size <= mapped_memory->size() - id,
                            "Constant offset/size is out of bounds for mapped weights file: offset=",
                            id,
                            ", size=",
                            byte_size,
                            ", file_size=",
                            mapped_memory->size(),
                            ", name=",
                            descriptor.nameFromCompiler);
            auto weight_buffer =
                std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data() + id,
                                                                                      byte_size,
                                                                                      mapped_memory);

            auto [it, inserted] =
                constants.emplace(id,
                                  std::make_shared<ov::op::v0::Constant>(descriptor.precision,
                                                                         descriptor.shapeFromCompiler.to_shape(),
                                                                         weight_buffer));
            if (!inserted) {
                OPENVINO_ASSERT(it->second->get_byte_size() == byte_size,
                                "Duplicate constant offset found with mismatching byte size: offset=",
                                id);
            }
        }
    }

    return constants;
}

}  // namespace intel_npu
