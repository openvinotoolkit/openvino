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
// See the License for the specific language governing permissions and`
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cmath>
#include <ngraph/opsets/opset5.hpp>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void loop(const std::shared_ptr<Function>& body,
                      const op::util::OutputDescriptionVector& out_descs,
                      const op::util::InputDescriptionVector& input_descs,
                      const opset5::Loop::SpecialBodyPorts& special_ports,
                      const HostTensorVector& out,
                      const HostTensorVector& args);
        }
    }
}
