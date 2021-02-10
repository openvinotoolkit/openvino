// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/op/sink.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph
{
    namespace pass
    {
        /**
         * @brief The transformation replaces provided `nodes_to_freeze` nodes with Constants.
         * Constants are created inside the transformation with provided `replacing_values`.
         * `replacing_values` must be provided in the same order as the passed list of nodes.
         *
         * Example:
         * 1. before transformation:
         *  Parameter (shape: 3, 2) -> Split (axis: 1, num_split=2) -> AnyNode_1
         *                                                          \
         *                                                            -> AnyNode_2
         *
         * 2. transformation call, freeze Split layer with values (1, 2, 3), (4, 5, 6)
         * 3. after transformation:
         *  Const_1 (shape: 3, 1; value: (1, 2, 3)) -> AnyNode_1
         *  Const_2 (shape: 3, 1; value: (4, 5, 6)) -> AnyNode_2
         *
         */
        class NGRAPH_API FreezeNodes : public ngraph::pass::FunctionPass
        {
        public:
            using values_for_node = std::vector<std::vector<char>>;
            NGRAPH_RTTI_DECLARATION;
            bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

            /**
             * @brief A constructor of the transformation.
             *
             * @param nodes_to_freeze A list of nodes in the ngraph::function to replace.
             * @param replacing_values A list of values with which the constants will be created
             * that will replace the specified nodes.
             */
            FreezeNodes(const NodeVector& nodes_to_freeze,
                        const std::vector<values_for_node>& replacing_values);

        private:
            NodeVector m_nodes_to_freeze;
            std::vector<values_for_node> m_replacing_values;
        };
    } // namespace pass
} // namespace ngraph
