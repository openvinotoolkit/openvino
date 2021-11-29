// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_import/onnx_utils.hpp"
#include "ops_bridge.hpp"

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
