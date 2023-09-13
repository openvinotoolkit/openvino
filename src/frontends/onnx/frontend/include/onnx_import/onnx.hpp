// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <string>

#include "ngraph/deprecated.hpp"
#include "ngraph/function.hpp"
#include "onnx_importer_visibility.hpp"

/// \brief              Top level nGraph namespace.
namespace ngraph {
/// \brief          ONNX importer features namespace.
///                 Functions in this namespace make it possible to use ONNX models.
namespace onnx_import {
/// \brief      Returns a set of names of supported operators
///             for the given opset version and domain.
///
/// \param[in]  version   An opset version to get the supported operators for.
/// \param[in]  domain    A domain to get the supported operators for.
///
/// \return     The set containing names of supported operators.
NGRAPH_API_DEPRECATED ONNX_IMPORTER_API std::set<std::string> get_supported_operators(std::int64_t version,
                                                                                      const std::string& domain);

/// \brief      Determines whether ONNX operator is supported.
///
/// \param[in]  op_name   The ONNX operator name.
/// \param[in]  version   The ONNX operator set version.
/// \param[in]  domain    The domain the ONNX operator is registered to.
///                       If not set, the default domain "ai.onnx" is used.
///
/// \return     true if operator is supported, false otherwise.
NGRAPH_API_DEPRECATED ONNX_IMPORTER_API bool is_operator_supported(const std::string& op_name,
                                                                   std::int64_t version,
                                                                   const std::string& domain = "ai.onnx");

/// \brief      Imports and converts an serialized ONNX model from the input stream
///             to an nGraph Function representation.
///
/// \note       If stream parsing fails or the ONNX model contains unsupported ops,
///             the function throws an ngraph_error exception.
///
/// \param[in]  stream      The input stream (e.g. file stream, memory stream, etc).
/// \param[in]  model_path  The path to the imported onnx model.
///                         It is required if the imported model uses data saved in external
///                         files.
/// \param[in]  enable_mmap Enable mapping files with external weights instead of reading.
///
/// \return     An nGraph function that represents a single output from the created graph.
NGRAPH_API_DEPRECATED ONNX_IMPORTER_API std::shared_ptr<Function> import_onnx_model(std::istream& stream,
                                                                                    const std::string& model_path = "",
                                                                                    bool enable_mmap = false);

/// \brief     Imports and converts an ONNX model from the input file
///            to an nGraph Function representation.
///
/// \note      If file parsing fails or the ONNX model contains unsupported ops,
///            the function throws an ngraph_error exception.
///
/// \param[in] file_path  The path to a file containing the ONNX model
///                       (relative or absolute).
/// \param[in]  enable_mmap Enable mapping files with external weights instead of reading.
///
/// \return    An nGraph function that represents a single output from the created graph.
NGRAPH_API_DEPRECATED ONNX_IMPORTER_API std::shared_ptr<Function> import_onnx_model(const std::string& file_path,
                                                                                    bool enable_mmap = false);
}  // namespace onnx_import

}  // namespace ngraph
