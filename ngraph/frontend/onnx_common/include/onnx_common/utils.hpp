// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ngraph/type/element_type.hpp"

namespace ONNX_NAMESPACE
{
    enum TensorProto_DataType;
}

namespace ngraph
{
    namespace onnx_common
    {
        /// \brief Retuns size of an ONNX data type in bytes.
        ///
        /// \param onnx_type Number assigned to an ONNX data type in the TensorProto_DataType enum.
        ///
        size_t get_onnx_data_size(int32_t onnx_type);

        /// \brief Retuns a nGraph data type corresponding to an ONNX type.
        ///
        /// \param onnx_type An element of TensorProto_DataType enum which determines an ONNX type.
        ///
        element::Type_t onnx_to_ng_data_type(const ONNX_NAMESPACE::TensorProto_DataType& onnx_type);

        /// \brief Retuns an ONNX data type corresponding to a nGraph data type.
        ///
        /// \param ng_type An element of element::Type_t enum class which determines a nGraph data
        /// type.
        ///
        ONNX_NAMESPACE::TensorProto_DataType ng_to_onnx_data_type(const element::Type_t& ng_type);

        /// \brief Retuns true if a nGraph data type is mapped to an ONNX data type.
        ///
        /// \param ng_type An element of element::Type_t enum class which determines a nGraph data
        /// type.
        ///
        bool is_supported_ng_type(const element::Type_t& ng_type);

    } // namespace onnx_common
} // namespace ngraph
