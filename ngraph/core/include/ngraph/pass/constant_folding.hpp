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

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        /**
         * @brief Constant folding iterates over the function and tries to evaluate nodes
         *        with constant inputs. Such nodes are then replaced with new Constants containing
         *        the result of a folded operation.
         */
        class NGRAPH_API ConstantFolding : public FunctionPass
        {
        public:
            NGRAPH_RTTI_DECLARATION;
            bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

        private:
            void copy_runtime_info_to_target_inputs(const std::shared_ptr<Node>& node,
                                                    const Output<Node>& replacement);
            /// \brief Folds pre-calculated output tensor values to constants in case lower and
            /// upper estimations are equal. Traverses graph backwards starting from the results.
            bool pre_calculated_values_folding(const std::shared_ptr<ngraph::Function>& f);
        };
    } // namespace pass
} // namespace ngraph
