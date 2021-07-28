// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>

#include "onnx_import/core/operator_set.hpp"
#include "utils/onnx_importer_visibility.hpp"

namespace ngraph
{
    namespace onnx_import
    {
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
        ONNX_IMPORTER_API
        void register_operator(const std::string& name,
                               std::int64_t version,
                               const std::string& domain,
                               Operator fn);

        /// \brief      Unregisters ONNX custom operator.
        ///             The function unregisters previously registered operator.
        ///
        /// \param      name      The ONNX operator name.
        /// \param      version   The ONNX operator set version.
        /// \param      domain    The domain the ONNX operator is registered to.
        ONNX_IMPORTER_API
        void unregister_operator(const std::string& name,
                                 std::int64_t version,
                                 const std::string& domain);

    } // namespace onnx_import

} // namespace ngraph
