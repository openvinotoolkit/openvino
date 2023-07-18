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
#include <string>

#include "ngraph/deprecated.hpp"
#include "onnx_import/core/operator_set.hpp"
#include "onnx_importer_visibility.hpp"

namespace ngraph {
namespace onnx_import {
/// \brief      Registers ONNX custom operator.
///             The function performs the registration of external ONNX operator
///             which is not part of ONNX importer.
///
/// \note       The operator must be registered before calling
///             "import_onnx_model" functions.
///
/// \param      name      The ONNX operator name.
/// \param      version   The ONNX operator set version.
/// \param      domain    The domain the ONNX operator is registered to.
/// \param      fn        The function providing the implementation of the operator
///                       which transforms the single ONNX operator to an nGraph sub-graph.
NGRAPH_API_DEPRECATED ONNX_IMPORTER_API void register_operator(const std::string& name,
                                                               std::int64_t version,
                                                               const std::string& domain,
                                                               Operator fn);

/// \brief      Unregisters ONNX custom operator.
///             The function unregisters previously registered operator.
///
/// \param      name      The ONNX operator name.
/// \param      version   The ONNX operator set version.
/// \param      domain    The domain the ONNX operator is registered to.
NGRAPH_API_DEPRECATED ONNX_IMPORTER_API void unregister_operator(const std::string& name,
                                                                 std::int64_t version,
                                                                 const std::string& domain);

}  // namespace onnx_import

}  // namespace ngraph
