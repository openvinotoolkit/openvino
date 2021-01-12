//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cstdint>
#include <string>

#include "onnx_import/core/operator_set.hpp"
#include "onnx_import/utils/onnx_importer_visibility.hpp"

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
