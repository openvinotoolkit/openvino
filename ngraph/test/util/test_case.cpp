// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>

#include "test_case.hpp"

namespace ngraph
{
    namespace test
    {
        std::shared_ptr<Function> function_from_ir(const std::string& xml_path,
                                                   const std::string& bin_path)
        {
            InferenceEngine::Core c;
            auto network = c.ReadNetwork(xml_path, bin_path);
            return network.getFunction();
        }
    }
}
