// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "intel_npu/config/config.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/common.hpp"

namespace intel_npu {

/**
 * @brief A helper structure used for storing metadata corresponding to one input/output entry.
 */
struct IODescriptor {
    /**
     * @brief The name of the input/output assigned by the compiler.
     * @details This value may differ from other name attributes:
     *  - The compiler could have created additional inputs/outputs (e.g. for representing states). These are not
     * found in the original IR model.
     *  - The compiler may append indices to names in the case where duplicate names are found.
     * @note The prefixes introduced by the compiler in order to differentiate the special cases (e.g. states and shape
     * tensors) were removed prior to initializing this field.
     */
    std::string nameFromCompiler;

    ov::element::Type precision;

    ov::PartialShape shapeFromCompiler;

    /**
     * @brief If set to "true", the current object describes a buffer which may be used for altering a state tensor.
     * @details This flag is set if the compiler prefixed the name using a "read value" prefix. The state input and
     * state output descriptors are also tied using the "relatedDescriptorIndex" attribute.
     */
    bool isStateInput = false;

    /**
     * @brief If set to "true", the current object describes a buffer which reflects the value of a state tensor.
     * @details This flag is set if the compiler prefixed the name using an "assign" prefix. The state input and
     * state output descriptors are also tied using the "relatedDescriptorIndex" attribute.
     */
    bool isStateOutput = false;

    /**
     * @brief If set to "true", the buffer of the tensor described here contains as value the shape of the referenced
     * tensor.
     * @details This flag is set if the compiler prefixed the name using a "shape" prefix.
     *
     * The referenced tensor bears the same name ("nameFromCompiler"), but its "isShapeTensor" value is set to
     * "false". The two descriptors are also tied using the "relatedDescriptorIndex" attribute.
     */
    bool isShapeTensor = false;

    /**
     * @brief Points towards a related descriptor.
     * @details The related descriptors are defined by (state input, state output) or (dynamic tensor, shape tensor)
     * pairs.
     */
    std::optional<size_t> relatedDescriptorIndex;

    /**
     * @brief The friendly name of the node extracted from the IR model.
     * @details In some cases, this field is required for constructing a dummy model which uses the same input/output
     * metadata as the original IR model.
     *
     * This field may be empty if the I/O entry is not found in the original IR model (i.e. the entry was added by the
     * compiler).
     */
    std::string nodeFriendlyName;

    /**
     * @brief The names of the output tensors extracted from the IR model.
     * @details In some cases, this field is required for constructing a dummy model which uses the same input/output
     * metadata as the original IR model.
     *
     * This field may be empty if the I/O entry is not found in the original IR model (i.e. the entry was added by the
     * compiler).
     */
    std::unordered_set<std::string> outputTensorNames;

    /**
     * @brief The shape extracted from the IR model.
     * @details The values may differ from the ones found in "shapeFromCompiler" if batching is to be handled by the
     * plugin.
     *
     * This field may be empty if the I/O entry is not found in the original IR model (i.e. the entry was added
     * by the compiler).
     */
    std::optional<ov::PartialShape> shapeFromIRModel = std::nullopt;
};

struct NetworkMetadata final {
    std::string name;

    std::vector<IODescriptor> inputs;
    std::vector<IODescriptor> outputs;
    std::vector<IODescriptor> profilingOutputs;

    size_t numStreams = 1;

    /**
     * @brief Binds the (state input, state output) and (dynamic tensor, shape tensor) pairs using the
     * "relatedDescriptorIndex" attribute.
     * @details For state inputs, the "relatedDescriptorIndex" value is set to the index of the output which bears the
     * same name. The reverse is also applied.
     *
     * For shape tensors, the lookup is performed in the same container (inputs or outputs). The value is once again set
     * to the index of the entry which bears the same name.
     */
    void bindRelatedDescriptors();

};  // namespace intel_npu

}  // namespace intel_npu
