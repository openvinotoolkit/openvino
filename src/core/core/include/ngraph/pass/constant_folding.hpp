// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
