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
        /// \brief The Validate pass performs sanity checks on attributes and inputs, and
        /// computes output shapes and element types for all computation nodes in a given
        /// computation graph.
        ///
        /// \details The verification and inference is done via invoking each node's specific
        /// implementation of \link ngraph::Node::validate_and_infer_types() \endlink function.
        ///
        /// By default, the \ref ngraph::pass::Manager runs this pass after executing every
        /// optimization pass. This is to ensure that any update to the graph by an optimization
        /// pass does not break the shape and data type requirement on a computation node.
        /// This default validation run can be changed via calling the
        /// \link ngraph::pass::Manager::set_per_pass_validation(bool) \endlink function.
        class NGRAPH_API Validate : public FunctionPass
        {
        public:
            Validate()
                : FunctionPass()
            {
            }
            bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
        };
    }
}
