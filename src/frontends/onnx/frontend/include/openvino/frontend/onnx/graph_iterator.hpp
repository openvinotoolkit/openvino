// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <map>
#include <mutex>
#include <string>

#include "openvino/core/deprecated.hpp"
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
    /// \deprecated Returning a std::map by value lets an implementation living in another shared library (possibly
    /// linked against a different C runtime) allocate memory that this library frees. Override get_metadata_ref()
    /// instead. This method is still required to be implemented, and it is only called by the default
    /// implementation of get_metadata_ref().
    OPENVINO_DEPRECATED(
        "Returns by value, which is not safe across shared libraries with different C runtimes. "
        "Implement get_metadata_ref() instead.")
    virtual std::map<std::string, std::string> get_metadata() const = 0;

    /// \brief Returns the directory path where the model is located.
    /// \deprecated Returning a std::filesystem::path by value lets an implementation living in another shared library
    /// (possibly linked against a different C runtime) allocate memory that this library frees. Override
    /// get_model_dir_ref() instead. This method is still required to be implemented, and it is only called by the
    /// default implementation of get_model_dir_ref().
    OPENVINO_DEPRECATED(
        "Returns by value, which is not safe across shared libraries with different C runtimes. "
        "Implement get_model_dir_ref() instead.")
    virtual std::filesystem::path get_model_dir() const = 0;

    /// \brief Destructor
    virtual ~GraphIterator();

    // NOTE: the virtual functions below are declared after the destructor on purpose, so that the position of every
    // virtual function declared above stays unchanged in the vtable.

    /// \brief Retrieves metadata associated with the graph.
    /// The map is returned by reference and must stay valid for the lifetime of the iterator. The caller copies it
    /// with its own code, so the nodes of the map are allocated and released by the same module.
    /// The default implementation calls the deprecated get_metadata() once and keeps the result, so iterators that
    /// were written against the previous interface keep working (they still allocate in their own module).
    virtual const std::map<std::string, std::string>& get_metadata_ref() const;

    /// \brief Returns the directory path where the model is located.
    /// The path is returned by reference and must stay valid for the lifetime of the iterator. The caller copies it
    /// with its own code, so the storage of the path is allocated and released by the same module.
    /// The default implementation calls the deprecated get_model_dir() once and keeps the result, so iterators that
    /// were written against the previous interface keep working (they still allocate in their own module).
    virtual const std::filesystem::path& get_model_dir_ref() const;

private:
    // Storage for the default implementations of get_metadata_ref() and get_model_dir_ref().
    mutable std::once_flag m_legacy_metadata_once;
    mutable std::map<std::string, std::string> m_legacy_metadata;
    mutable std::once_flag m_legacy_model_dir_once;
    mutable std::filesystem::path m_legacy_model_dir;
};

}  // namespace ov::frontend::onnx
