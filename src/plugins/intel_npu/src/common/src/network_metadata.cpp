// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/network_metadata.hpp"

#include "openvino/core/except.hpp"

namespace intel_npu {

void NetworkMetadata::bindRelatedDescriptors() {
    size_t ioIndex = 0;

    for (IODescriptor& input : inputs) {
        if (input.relatedDescriptorIndex.has_value()) {
            ++ioIndex;
            continue;
        }

        if (input.isStateInput) {
            const auto relatedDescriptorIterator =
                std::find_if(outputs.begin(), outputs.end(), [&](const IODescriptor& output) {
                    return output.isStateOutput && (output.nameFromCompiler == input.nameFromCompiler);
                });

            if (relatedDescriptorIterator != outputs.end()) {
                input.relatedDescriptorIndex = std::distance(outputs.begin(), relatedDescriptorIterator);
                outputs.at(*input.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;

                // A state input and its state output are two views of the same variable, so their shape and
                // precision must match. Otherwise the buffer shared between them, which is sized from the input,
                // would be undersized for the output.
                OPENVINO_ASSERT(relatedDescriptorIterator->shapeFromCompiler == input.shapeFromCompiler &&
                                    relatedDescriptorIterator->precision == input.precision,
                                "State input '",
                                input.nameFromCompiler,
                                "' does not match its state output in shape or precision.");
            }
        } else if (input.isShapeTensor) {
            const auto relatedDescriptorIterator =
                std::find_if(inputs.begin(), inputs.end(), [&](const IODescriptor& candidate) {
                    return !candidate.isShapeTensor && (candidate.nameFromCompiler == input.nameFromCompiler);
                });

            if (relatedDescriptorIterator != inputs.end()) {
                input.relatedDescriptorIndex = std::distance(inputs.begin(), relatedDescriptorIterator);
                inputs.at(*input.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;
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
            const auto relatedDescriptorIterator =
                std::find_if(outputs.begin(), outputs.end(), [&](const IODescriptor& candidate) {
                    return !candidate.isShapeTensor && (candidate.nameFromCompiler == output.nameFromCompiler);
                });

            if (relatedDescriptorIterator != outputs.end()) {
                output.relatedDescriptorIndex = std::distance(outputs.begin(), relatedDescriptorIterator);
                outputs.at(*output.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;
            }
        }

        ++ioIndex;
    }
}

}  // namespace intel_npu
