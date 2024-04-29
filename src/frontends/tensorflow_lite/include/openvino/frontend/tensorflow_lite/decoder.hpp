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
    std::shared_ptr<ov::frontend::tensorflow_lite::QuantizationInfo> m_quantization_info;
    std::shared_ptr<ov::frontend::tensorflow_lite::SparsityInfo> m_sparsity_info;
    ov::PartialShape m_partial_shape;
    ov::element::Type m_element_type;
    const uint8_t* m_tensor_data;
    std::vector<std::string> m_tensor_names;
};

class TENSORFLOW_LITE_API DecoderBase : public ov::frontend::DecoderBase {
public:
    using OpTypeByName = std::unordered_map<std::string, std::string>;
    /// \brief Get attribute value by name
    ///
    /// \param name Attribute name
    /// \return Shared pointer to appropriate value converted to openvino data type if it exists, 'nullptr' otherwise
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    /// \brief Get a number of inputs
    virtual size_t get_input_size() const = 0;

    /// \brief Get a number of outputs
    virtual size_t get_output_size() const = 0;

    /// \brief Get a producer name and its output port index
    ///
    /// \param input_port_idx              Input port index by which data is consumed
    /// \param producer_name               A producer name
    /// \param producer_output_port_name   Output port name if exists
    /// \param producer_output_port_index  Output port index from which data is generated
    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    /// \brief Get operation type
    virtual const std::string& get_op_type() const = 0;

    /// \brief Get node name
    virtual const std::string& get_op_name() const = 0;

    virtual std::string get_output_tensor_name(size_t idx) const = 0;
    virtual ov::element::Type get_output_tensor_type(size_t idx) const = 0;
    virtual std::string get_input_tensor_name(size_t idx) const = 0;

    virtual TensorMetaInfo get_input_tensor_info(size_t idx) const = 0;
    virtual TensorMetaInfo get_output_tensor_info(size_t idx) const = 0;

    /// \brief Destructor
    virtual ~DecoderBase();
};

class TENSORFLOW_LITE_API DecoderBaseOperation : public ov::frontend::tensorflow_lite::DecoderBase {};

class TENSORFLOW_LITE_API DecoderBaseTensor : public ov::frontend::tensorflow_lite::DecoderBase {
public:
    virtual TensorMetaInfo get_tensor_info() const = 0;
    virtual int64_t get_input_idx() const = 0;
    virtual int64_t get_output_idx() const = 0;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
