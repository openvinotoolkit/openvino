//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <cmath>
/*#include <runtime/backend_manager.hpp>
#include <runtime/backend.hpp>*/
#include <ngraph/opsets/opset5.hpp>

namespace ngraph
{
    namespace runtime
    {

 /*       template <typename fromPrec, typename toPrec>
        std::vector<std::uint8_t> convertPrecision(std::vector<std::uint8_t> &buffer, const size_t elementsCount, const size_t elementSize);

        std::vector<std::uint8_t> convertOutputPrecision(std::vector<std::uint8_t> &output, const element::Type_t &fromPrecision, const element::Type_t &toPrecision,
                                                         const size_t elementsCount);

        std::vector<std::vector<std::uint8_t>> interpreterFunction(const std::shared_ptr<Function> &function, const std::vector<std::vector<std::uint8_t>> &inputs,
                                                                   element::Type_t convertType);

        namespace reference
        {
            void loop(ngraph::opset5::Loop& loop,
                      const std::vector<std::shared_ptr<HostTensor>>& out,
                      const std::vector<std::shared_ptr<HostTensor>>& args);
        }*/
    }
}
