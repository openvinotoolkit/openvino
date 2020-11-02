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

#pragma once

#include "backend_visibility.hpp"
#include "ngraph/pass/pass.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace pass
    {
        class BACKEND_API Opset1Upgrade : public NodePass
        {
        public:
            ///
            /// \brief    Constructor for the Opset1Upgrade transformation pass.
            ///
            /// \details  This transformation pass iterates over all nodes in a graph
            /// and updates version 0 ops to their version 1 equivalents.
            /// All ops in the final graph have op version 1.
            Opset1Upgrade() = default;
            bool run_on_node(std::shared_ptr<ngraph::Node> node) override;
        };
    }
}

NGRAPH_SUPPRESS_DEPRECATED_END
