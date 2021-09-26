// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ie_core.hpp>
#include "executable.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace ie
        {
            // A Inference Engine executable object produced by compiling an nGraph function.
            class IE_Executable final : public Executable
            {
            public:
                IE_Executable(std::shared_ptr<Function> func, std::string device);
                virtual ~IE_Executable() {}
                bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override final;

            private:
                InferenceEngine::CNNNetwork m_network;
                std::string m_device;
            };
        }
    }
}
