// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/quantization_info.hpp"
#include "openvino/frontend/tensorflow_lite/sparsity_info.hpp"
#include "openvino/frontend/tensorflow_lite/visibility.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

struct TensorMetaInfo {
    std::shared_ptr<QuantizationInfo> m_quantization_info;
    std::shared_ptr<SparsityInfo> m_sparsity_info;
    ov::PartialShape m_partial_shape;
    ov::element::Type m_element_type;
    const uint8_t* m_tensor_data;
    std::string m_tensor_name;
};

class DecoderBase : public ov::frontend::DecoderBase {};

// DecoderBaseOperation corresponds to operation node to retrieve its attributes and information about input and output
// tensors
class DecoderBaseOperation : public ov::frontend::tensorflow_lite::DecoderBase {
public:
    /// \brief Get input tensor name by index
    /// Operation nodes are connected between each other by tensors.
    /// Each tensor must have unique name in a graph
    /// This method returns tensor name that comes to this operation node by input index idx
    virtual std::string get_input_tensor_name(size_t idx) const = 0;

    /// \brief Get input tensor type by index
    virtual ov::element::Type get_input_tensor_type(size_t idx) const = 0;

    /// \brief Get output tensor name by index
    /// Operation nodes are connected between each other by tensors.
    /// Each tensor must have unique name in a graph
    /// This method returns tensor name that outputs by output index idx from this operation
    virtual std::string get_output_tensor_name(size_t idx) const = 0;

    /// \brief Get output tensor type by index
    virtual ov::element::Type get_output_tensor_type(size_t idx) const = 0;

    /// \brief Get input tensor info
    /// returns TensorInfo by input idx index that corresponds to a tensor
    /// (it can be Constant or just connection between this tensor producer and this current node)
    virtual TensorMetaInfo get_input_tensor_info(size_t idx) const = 0;

    /// \brief Get output tensor info
    /// returns TensorInfo by output idx index that corresponds to a tensor
    /// (it can be connection between this tensor consumer and this current node)
    virtual TensorMetaInfo get_output_tensor_info(size_t idx) const = 0;

    /// \brief Get a number of outputs
    virtual size_t get_output_size() const = 0;
};

// DecoderBaseTensor corresponds to tensor node to retrieve information about type, shapem quantization and sparsity
// information
class DecoderBaseTensor : public ov::frontend::tensorflow_lite::DecoderBase {
public:
    /// \brief Get tensor info
    virtual TensorMetaInfo get_tensor_info() const = 0;

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
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
