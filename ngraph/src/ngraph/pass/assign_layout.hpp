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

#include <exception>
#include <sstream>

#include "ngraph/descriptor/output.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        template <typename LT>
        class AssignLayout : public NodePass
        {
        public:
            virtual bool run_on_node(std::shared_ptr<Node> node) override
            {
                try
                {
                    for (size_t i = 0; i < node->get_output_size(); ++i)
                    {
                        auto tv = &node->output(i).get_tensor();
                        if (nullptr == tv->get_tensor_layout())
                        {
                            auto layout = std::make_shared<LT>(*tv);
                            tv->set_tensor_layout(layout);
                        }
                    }
                }
                catch (const std::exception& e)
                {
                    std::stringstream ss;
                    ss << "Error with node " << *node << ": ";
                    ss << e.what();
                    throw std::invalid_argument(ss.str());
                }
                return false;
            }
        };
    }
}
