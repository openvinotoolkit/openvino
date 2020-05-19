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

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class NGRAPH_API Opset0Downgrade : public NodePass
        {
        public:
            ///
            /// \brief    Constructor for the Opv1 downgrade transformation pass.
            ///
            /// \details  This transformation pass iterates over all nodes in a graph
            /// and updates version 1 ops to their version 0 equivalents.
            /// All ops in the final graph have op version 0.
            Opset0Downgrade() = default;
            bool run_on_node(std::shared_ptr<ngraph::Node> node) override;
        };
    }
}
