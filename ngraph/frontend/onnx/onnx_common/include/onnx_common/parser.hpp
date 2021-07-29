// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <fstream>
#include <string>

namespace ONNX_NAMESPACE
{
    class ModelProto;
}

namespace ngraph
{
    namespace onnx_common
    {
        /// \brief   Parses an ONNX model from a file located on a storage device.
        ///
        /// \param   file_path    Path to the file containing an ONNX model.
        ///
        /// \return  The parsed in-memory representation of the ONNX model
        ONNX_NAMESPACE::ModelProto parse_from_file(const std::string& file_path);

        /// \brief   Parses an ONNX model from a stream (representing for example a file)
        ///
        /// \param   model_stream  Path to the file containing an ONNX model.
        ///
        /// \return  The parsed in-memory representation of the ONNX model
        ONNX_NAMESPACE::ModelProto parse_from_istream(std::istream& model_stream);
    } // namespace onnx_common

} // namespace ngraph
