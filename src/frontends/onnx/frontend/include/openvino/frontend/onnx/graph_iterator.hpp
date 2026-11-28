// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <map>
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
    /// linked against a different C runtime) allocate memory that this library frees. Additionally implement
    /// GraphIteratorV2::get_metadata_ref() by deriving from GraphIteratorV2: the frontend uses it and never calls this
    /// function for such an iterator. It remains pure virtual so that the layout and the vtable of GraphIterator stay
    /// unchanged for iterators that were built against an earlier version of this header.
    OPENVINO_DEPRECATED("Returns by value, which is not safe across shared libraries with different C runtimes. "
                        "Derive from GraphIteratorV2 and implement get_metadata_ref().")
    virtual std::map<std::string, std::string> get_metadata() const = 0;

    /// \brief Returns the directory path where the model is located.
    /// \deprecated Returning a std::filesystem::path by value lets an implementation living in another shared library
    /// (possibly linked against a different C runtime) allocate memory that this library frees. Implement
    /// GraphIteratorV2::get_model_dir_ref() by deriving from GraphIteratorV2: the frontend uses it and never calls this
    /// function for such an iterator. It remains pure virtual so that the layout and the vtable of GraphIterator stay
    /// unchanged for iterators that were built against an earlier version of this header.
    OPENVINO_DEPRECATED("Returns by value, which is not safe across shared libraries with different C runtimes. "
                        "Derive from GraphIteratorV2 and implement get_model_dir_ref().")
    virtual std::filesystem::path get_model_dir() const = 0;

    /// \brief Destructor
    virtual ~GraphIterator();
};

/// \brief GraphIterator that returns shared data by reference. Derive from it instead of GraphIterator.
///
/// GraphIterator itself must not change (data layout and vtable): iterators built against an earlier header are
/// loaded into the same process as a newer frontend. Instead of adding virtual functions or data members to it, the
/// newer accessors live in this derived class. An iterator opts in by deriving from GraphIteratorV2 instead of
/// GraphIterator and implementing the two reference accessors; the deprecated by-value functions are implemented here
/// (as copies of the referenced values) and do not have to be written again.
///
/// The frontend works with GraphIterator pointers only. It discovers this class with
/// `dynamic_cast<const GraphIteratorV2*>(iterator)`. An iterator that is not a GraphIteratorV2 (for example, one built
/// against an earlier header) is served by the deprecated by-value functions of GraphIterator, exactly as before.
///
/// The references must stay valid for the lifetime of the iterator. The frontend copies the values with its own code,
/// so the memory is allocated and released by the same module. This is not guaranteed for values returned by value
/// from a library that uses a different C runtime.
///
/// Later revisions must be new classes (for example GraphIteratorV3 deriving from GraphIteratorV2) and must not add
/// virtual functions to this one, so that every revision keeps a stable vtable. When the deprecated by-value functions
/// are removed from GraphIterator in a release that is allowed to break the ABI, the accessors below move into
/// GraphIterator and GraphIteratorV2 becomes an alias of it.
class ONNX_FRONTEND_API GraphIteratorV2 : public GraphIterator {
public:
    /// \brief Retrieves metadata associated with the graph.
    virtual const std::map<std::string, std::string>& get_metadata_ref() const = 0;

    /// \brief Returns the directory path where the model is located.
    virtual const std::filesystem::path& get_model_dir_ref() const = 0;

    /// \brief Implements the deprecated by-value function as a copy of get_metadata_ref(). The frontend does not call
    /// it for a GraphIteratorV2.
    OPENVINO_SUPPRESS_DEPRECATED_START
    std::map<std::string, std::string> get_metadata() const override;

    /// \brief Implements the deprecated by-value function as a copy of get_model_dir_ref(). The frontend does not call
    /// it for a GraphIteratorV2.
    std::filesystem::path get_model_dir() const override;
    OPENVINO_SUPPRESS_DEPRECATED_END

    ~GraphIteratorV2() override;
};

}  // namespace ov::frontend::onnx
