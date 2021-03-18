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

#include <vector>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// Root of nodes that can be sink nodes
        class NGRAPH_API Sink : public Op
        {
        public:
            virtual ~Sink() = 0;
            NGRAPH_RTTI_DECLARATION;

        protected:
            Sink()
                : Op()
            {
            }

            explicit Sink(const OutputVector& arguments)
                : Op(arguments)
            {
            }
        };
    }
    using SinkVector = std::vector<std::shared_ptr<op::Sink>>;
}
