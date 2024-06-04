// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/al/icompiler.hpp"

namespace intel_npu {

std::optional<size_t> NetworkMetadata::findByName(const std::vector<IODescriptor>& descriptors,
                                                  const std::string_view targetName) {
    for (size_t descriptorIndex = 0; descriptorIndex < descriptors.size(); ++descriptorIndex) {
        if (descriptors.at(descriptorIndex).nameFromCompiler == targetName) {
            return descriptorIndex;
        }
    }

    return std::nullopt;
}

void NetworkMetadata::bindRelatedDescriptors() {
    size_t ioIndex = 0;

    for (IODescriptor& input : inputs) {
        if (input.relatedDescriptorIndex.has_value()) {
            ++ioIndex;
            continue;
        }

        if (input.isStateInput) {
            const std::optional<size_t> relatedDescriptorIndex = findByName(outputs, input.nameFromCompiler);

            if (relatedDescriptorIndex.has_value()) {
                input.relatedDescriptorIndex = relatedDescriptorIndex;
                outputs.at(*relatedDescriptorIndex).relatedDescriptorIndex = std::optional(ioIndex);
            }
        } else if (input.isShapeTensor) {
            const std::optional<size_t> relatedDescriptorIndex = findByName(inputs, input.nameFromCompiler);

            if (relatedDescriptorIndex.has_value() && *relatedDescriptorIndex != ioIndex) {
                input.relatedDescriptorIndex = relatedDescriptorIndex;
                inputs.at(*relatedDescriptorIndex).relatedDescriptorIndex = std::optional(ioIndex);
            }
        }

        ++ioIndex;
    }

    ioIndex = 0;

    for (IODescriptor& output : outputs) {
        if (output.relatedDescriptorIndex.has_value()) {
            ++ioIndex;
            continue;
        }

        if (output.isShapeTensor) {
            const std::optional<size_t> relatedDescriptorIndex = findByName(outputs, output.nameFromCompiler);

            if (relatedDescriptorIndex.has_value() && *relatedDescriptorIndex != ioIndex) {
                output.relatedDescriptorIndex = relatedDescriptorIndex;
                outputs.at(*relatedDescriptorIndex).relatedDescriptorIndex = std::optional(ioIndex);
            }
        }

        ++ioIndex;
    }
}

}  // namespace intel_npu
