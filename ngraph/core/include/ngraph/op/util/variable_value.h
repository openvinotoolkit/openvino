//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <ngraph/runtime/host_tensor.hpp>
#include <utility>

namespace ngraph
{
    class NGRAPH_API VariableValue
    {
    public:
        VariableValue() = default;

        VariableValue(HostTensorPtr value, bool reset_state) :
            m_reset(reset_state),
            m_value(std::move(value))
        {

        }

        void set_reset(bool reset) {
            m_reset = reset;
        }

        bool get_reset() const {
            return m_reset;
        }

        const HostTensorPtr& get_value() const {
            return m_value;
        }

        void set_value(const HostTensorPtr& value) {
            m_value = value;
        }
    private:
        bool m_reset = true;
        HostTensorPtr m_value;
    };
    using VariableValuePtr = std::shared_ptr<VariableValue>;
}
