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

#include "intel_npu/al/config/config.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/profiling_info.hpp"

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

private:
    std::optional<size_t> findByName(const std::vector<IODescriptor>& descriptors, const std::string_view targetName);

};  // namespace intel_npu

/**
 * @struct NetworkDescription
 * @brief The object returned by the compiler
 * to provide such information about a network as description of inputs and outputs,
 * name and compiled network in a format executable by device
 */
struct NetworkDescription final {
    NetworkDescription(std::vector<uint8_t>&& compiledNetwork, NetworkMetadata&& metadata)
        : compiledNetwork(std::move(compiledNetwork)),
          metadata(std::move(metadata)) {}
    // Force move semantics to prevent blob copies
    NetworkDescription(const NetworkDescription&) = delete;
    NetworkDescription(NetworkDescription&&) = default;
    NetworkDescription& operator=(const NetworkDescription&) = delete;
    NetworkDescription& operator=(NetworkDescription&&) = default;
    ~NetworkDescription() = default;

    std::vector<uint8_t> compiledNetwork;

    NetworkMetadata metadata;
};

/**
 * @interface ICompiler
 * @brief An interface to be implemented by a concrete compiler to provide
 * methods for preparing a network for execution on a NPU device
 */
class ICompiler : public std::enable_shared_from_this<ICompiler> {
public:
    /**
     * @brief Returns the maximum OpenVino opset version supported by the compiler
     * @return opset version e.g. 11 for opset11
     */
    virtual uint32_t getSupportedOpsetVersion() const = 0;

    /**
     * @brief Transforms a network from the OpenVINO model representation to a format executable
     * by a NPU device
     * @param model a shared pointer to the OpenVINO model to be compiled
     * @param config a reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @return a shared pointer on an object implementing NetworkDescription interface
     */
    virtual NetworkDescription compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const = 0;

    /**
     * @brief Returns information about supported layers of the network passed
     * @param model The model to be queried
     * @param config A reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @returns SupportedOpsMap structure with information about supported layers
     */
    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const = 0;

    /**
     * @brief Parses already compiled network to extract meta information:
     *        inputs and outputs descriptions
     * @param network compiled network represented as a vector of char
     * @param config a reference to NPUConfig containing plugin config options
     *        Note: compilation options will be ignored,
     *        since the network is already compiled
     * @param netName a reference to the string describing network name
     *        to be used for creating network description
     * @return a shared pointer on an object implementing NetworkDescription interface
     */
    virtual NetworkMetadata parse(const std::vector<uint8_t>& network, const Config& config) const = 0;

    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                                    const std::vector<uint8_t>& network,
                                                                    const Config& config) const = 0;

protected:
    virtual ~ICompiler() = default;
};

}  // namespace intel_npu
