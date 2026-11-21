// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "openvino/core/runtime_attribute.hpp"
#include "openvino/frontend/onnx/decoder.hpp"
#include "openvino/frontend/onnx/visibility.hpp"

namespace ov::frontend::onnx {

/// Abstract representation for an input model graph that gives nodes in topologically sorted order
/// It returns decoders for model inputs and outputs (DecoderBaseTensor objects) and for operation nodes
/// (DecoderBaseOperation objects) DecoderBaseOperation objects for operation nodes must be sorted in topological order
/// from producing nodes to consumer nodes when `get_decoder()` is called. DecoderBaseTensor objects for inputs and
/// outputs must be returned first by `get_decoder()` method. Order of DecoderBaseTensor objects for inputs and outputs
/// defines their order in the original model, i.e. model input index and model output index.
/// For example, calling `get_decoder()` during iterating GraphIterator returns
/// DecoderBaseTensor (for input 0), ..., DecoderBaseTensor (for input n-1),
/// DecoderBaseTensor (for output 0), ..., DecoderBaseTensor (for output m-1),
/// DecoderBaseOperation (for op 1), ..., DecoderBaseOperation (for op k),
/// where n - number of inputs in the model, m - number of outputs in the model k - number of operation nodes.
/// NOTE: constants are ignored and no decoder object is returned for constant.
class ONNX_FRONTEND_API GraphIterator : ::ov::RuntimeAttribute {
public:
    using Ptr = std::shared_ptr<GraphIterator>;

    /// \brief Get a number of operation nodes in the graph
    virtual size_t size() const = 0;

    /// \brief Set iterator to the start position
    virtual void reset() = 0;

    /// \brief Move to the next node in the graph
    virtual void next() = 0;

    /// \brief Returns true if iterator goes out of the range of available nodes
    virtual bool is_end() const = 0;

    /// \brief Return a pointer to a decoder of the current node
    virtual std::shared_ptr<DecoderBase> get_decoder() const = 0;

    /// \brief Returns opset version of requested domain, stored in a ModelProto
    /// If there are no domain found returns -1
    virtual int64_t get_opset_version(const std::string& domain) const = 0;

    /// \brief Retrieves metadata associated with the graph.
    virtual std::map<std::string, std::string> get_metadata() const = 0;

    /// \brief Returns the directory path where the model is located.
    virtual std::filesystem::path get_model_dir() const = 0;

    /// \brief Destructor
    virtual ~GraphIterator();
};

/// \brief Where the source model holds an initializer's data, independent of how the data is supplied now.
enum class SourceModelLocation : uint8_t {
    UNKNOWN = 0,   ///< The provider can't tell; the frontend applies its default rules
    NONE = 1,      ///< Not in the source model (created, rewritten or replaced after loading it)
    INTERNAL = 2,  ///< Stored inside the source model
    EXTERNAL = 3,  ///< Stored in an external data file of the source model
};

/// \brief External data location of an initializer as declared by the source model's external_data entries.
struct SourceModelExternalData {
    std::string location;  ///< The 'location' entry, relative to the source model's directory
    uint64_t offset = 0;   ///< The 'offset' entry, in bytes
    uint64_t length = 0;   ///< The 'length' entry, in bytes
};

/// \brief Optional interface a GraphIterator may also implement to report the provenance of initializers.
///
/// The ONNX frontend discovers it with dynamic_cast on the GraphIterator (and on the GraphIterators of subgraphs).
/// It uses the answers to decide which initializers get a WeightlessCacheAttribute: initializers that the source model
/// doesn't provide must not be referenced by weightless compiled blobs. Without this interface every initializer's
/// location is SourceModelLocation::UNKNOWN.
class ONNX_FRONTEND_API TensorOriginProvider {
public:
    /// \brief Returns where the source model holds the data of the named initializer.
    virtual SourceModelLocation get_source_model_location(const std::string& tensor_name) const = 0;

    /// \brief Returns the external data location the source model declares for the named initializer, if it is
    /// SourceModelLocation::EXTERNAL and declares a file. Returned even when the data is supplied in memory now.
    virtual std::optional<SourceModelExternalData> get_source_model_external_data(
        const std::string& tensor_name) const = 0;

    /// \brief Destructor
    virtual ~TensorOriginProvider();
};

}  // namespace ov::frontend::onnx
