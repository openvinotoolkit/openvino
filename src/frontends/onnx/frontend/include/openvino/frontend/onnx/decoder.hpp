// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/onnx/visibility.hpp"

namespace ov {
namespace frontend {
namespace onnx {

struct ONNX_FRONTEND_API TensorMetaInfo {
    ov::PartialShape m_partial_shape;
    ov::element::Type m_element_type;
    const uint8_t* m_tensor_data;
    std::string m_tensor_name;
};

class ONNX_FRONTEND_API DecoderBase : public ov::frontend::DecoderBase {
public:
    ~DecoderBase() override;
};

// DecoderBaseOperation corresponds to operation node to retrieve its attributes and information about input and output
// tensors
class ONNX_FRONTEND_API DecoderBaseOperation : public ov::frontend::onnx::DecoderBase {
public:
    /// \brief Get input tensor name by index
    /// Operation nodes are connected between each other by tensors.
    /// Each tensor must have unique name in a graph.
    /// The tensor name uniqueness is provided by developer during GraphIterator construction.
    /// This method returns tensor name that comes to this operation node by input index idx
    /// If idx is out-of-range, it throws std::exception inherited exception
    virtual std::string get_input_tensor_name(size_t idx) const = 0;

    /// \brief Get input tensor type by index
    /// If idx is out-of-range, it throws std::exception inherited exception
    virtual ov::element::Type get_input_tensor_type(size_t idx) const = 0;

    /// \brief Get output tensor name by index
    /// Operation nodes are connected between each other by tensors.
    /// Each tensor must have unique name in a graph.
    /// The tensor name uniqueness is provided by developer during GraphIterator construction.
    /// This method returns tensor name that outputs by output index idx from this operation
    /// If idx is out-of-range, it throws std::exception inherited exception
    virtual std::string get_output_tensor_name(size_t idx) const = 0;

    /// \brief Get output tensor type by index
    /// If idx is out-of-range, it throws std::exception inherited exception
    virtual ov::element::Type get_output_tensor_type(size_t idx) const = 0;

    /// \brief Get input tensor info
    /// returns TensorInfo by input idx index that corresponds to a tensor
    /// (it can correspond to Constant, Parameter or  intermediate tensor connecting a producer and this current node)
    /// If idx is out-of-range, it throws std::exception inherited exception
    virtual const TensorMetaInfo& get_input_tensor_info(size_t idx) const = 0;

    /// \brief Get output tensor info
    /// returns TensorInfo by output idx index that corresponds to a tensor
    /// (it can correspond to intermediate tensor connecting this current node and a consumer)
    /// If idx is out-of-range, it throws std::exception inherited exception
    virtual const TensorMetaInfo& get_output_tensor_info(size_t idx) const = 0;

    /// \brief Get a number of outputs
    virtual size_t get_output_size() const = 0;

    /// \brief Returns operation's opset version
    virtual uint64_t get_op_set() const = 0;

    /// \brief Returns operation's domain
    virtual const std::string& get_domain() const = 0;

    /// \brief Returns true if node has attribute
    virtual bool has_attribute(const std::string& name) const = 0;

    virtual void experimental_get_internal_structures(const void** node_def) const = 0;

    ~DecoderBaseOperation() override;
};

// DecoderBaseTensor corresponds to tensor node to retrieve information about type, shapem quantization and sparsity
// information
class ONNX_FRONTEND_API DecoderBaseTensor : public ov::frontend::onnx::DecoderBase {
public:
    /// \brief Get tensor info
    virtual const TensorMetaInfo& get_tensor_info() const = 0;

    /// \brief Get input index for tensor
    /// returns index of this input in the list of inputs in the model
    /// it must be from 0 to n-1, where n - number of inputs in the model
    /// if it is not input, returns  -1
    virtual int64_t get_input_idx() const = 0;

    /// \brief Get output index for tensor
    /// returns index of this output in the list of outputs in the model
    /// it must be from 0 to m-1, where m - number of outputs in the model
    /// if it is not input, returns  -1
    virtual int64_t get_output_idx() const = 0;

    ~DecoderBaseTensor() override;
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
