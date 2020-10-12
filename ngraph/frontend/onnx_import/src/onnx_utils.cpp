//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "onnx_import/onnx_utils.hpp"
#include "onnx_import/ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        void register_operator(const std::string& name,
                               std::int64_t version,
                               const std::string& domain,
                               Operator fn)
        {
            OperatorsBridge::register_operator(name, version, domain, std::move(fn));
        }

        void unregister_operator(const std::string& name,
                                 std::int64_t version,
                                 const std::string& domain)
        {
            OperatorsBridge::unregister_operator(name, version, domain);
        }

    } // namespace onnx_import

} // namespace ngraph
