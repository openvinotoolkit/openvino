// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/pass.hpp>

namespace ngraph
{
    namespace pass
    {
        /**
         * @brief The transformation finds all TensorIterator/Loop layers in the network,
         * processes all back edges that describe a connection between Result and Parameter
         * of the TensorIterator body,and inserts ReadValue layer between Parameter
         * and the next layers after this Parameter, and Assign layer after the layers
         * before the Result layer. Supported platforms: CPU, GNA.
         *
         * The example below describes the changes to the inner part (body, back edges) of the
         * Tensor Iterator layer.
         *  [] - TensorIterator body
         *  () - new layer
         *
         *  before applying the transformation:
         *  back_edge_1 -> [Parameter -> some layers ... -> Result ] -> back_edge_1
         *
         *  after applying the transformation:
         *  back_edge_1 -> [Parameter -> (ReadValue layer) -> some layers ... -> (Assign layer) ]
         *                                                              \
         *                                                               -> Result ] -> back_edge_1
         *
         * It is recommended to use this transformation in conjunction with the Reshape feature to
         * set  sequence dimension to 1 and with the UnrollTensorIterator transformation.
         * For convenience, we have already enabled the unconditional execution of the
         * UnrollTensorIterator transformation when using the LowLatency transformation for
         * CPU, GNA plugins, no action is required here.
         * After applying both of these transformations, the resulting network can be inferred step
         * by step, the states will store between inferences.
         */

        class NGRAPH_DEPRECATED("Use LowLatency2 instead.") NGRAPH_API LowLatency
            : public ngraph::pass::MatcherPass
        {
        public:
            NGRAPH_RTTI_DECLARATION;
            LowLatency();
        };

        /**
         * @brief The transformation finds all TensorIterator/Loop layers in the network,
         * processes all back edges that describe a connection between Result and Parameter
         * of the TensorIterator/Loop bodies,and inserts ReadValue and Assign layers at the
         * input and output corresponding to this back edge.
         * Supported platforms: CPU, GNA.
         *
         * The example below describes the changes made by the transformation
         *  [] - TensorIterator body
         *  () - new layer
         *  BE - back-edge
         *
         *  before applying the transformation:
         *  -> input1[BE_1 -> Parameter -> Layers ... -> Result  -> BE_1 ]output1->
         *
         *  after applying the transformation:
         *  ->(ReadValue)-> input1[BE_1 ->Parameter->Layers ...->Result->BE_1]output1 ->(Assign)
         *                                                                      \
         *                                                                       ->...
         * After applying the transformation, the resulting network can be inferred
         * step by step, the states will store between inferences.
         */
        class NGRAPH_API LowLatency2 : public ngraph::pass::FunctionPass
        {
        public:
            NGRAPH_RTTI_DECLARATION;

            explicit LowLatency2(bool use_const_initializer = true)
                : m_use_const_initializer(use_const_initializer)
            {
            }

            bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

        private:
            bool m_use_const_initializer;
        };
    } // namespace pass
} // namespace ngraph
