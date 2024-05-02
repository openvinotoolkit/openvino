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
    /// \brief Get a number of outputs
    virtual size_t get_output_size() const = 0;

    /// \brief Get output tensor name by index
    virtual std::string get_output_tensor_name(size_t idx) const = 0;

    /// \brief Get output tensor type by index
    virtual ov::element::Type get_output_tensor_type(size_t idx) const = 0;

    /// \brief Get input tensor name by index
    virtual std::string get_input_tensor_name(size_t idx) const = 0;

    /// \brief Get input tensor info
    virtual TensorMetaInfo get_input_tensor_info(size_t idx) const = 0;

    /// \brief Get output tensor info
    virtual TensorMetaInfo get_output_tensor_info(size_t idx) const = 0;
};

class TENSORFLOW_LITE_API DecoderBaseOperation : public ov::frontend::tensorflow_lite::DecoderBase {};

class TENSORFLOW_LITE_API DecoderBaseTensor : public ov::frontend::tensorflow_lite::DecoderBase {
public:
    /// \brief Get tensor info
    virtual TensorMetaInfo get_tensor_info() const = 0;

    /// \brief Get input index for tensor
    virtual int64_t get_input_idx() const = 0;

    /// \brief Get output index for tensor
    virtual int64_t get_output_idx() const = 0;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
