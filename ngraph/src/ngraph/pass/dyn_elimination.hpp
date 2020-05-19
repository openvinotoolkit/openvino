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

#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace pass
    {
        /// \brief The DynElimination pass finds dynamic operations in a graph whose
        /// shape relevant inputs have already been resolved to static values, and
        /// replaces those dynamic operations with the equivalent operations using
        /// static inputs and attributes.
        /// \details This pass should be executed after the ConstantFolding pass.
        ///
        /// The ConstantFolding and DynElimination passes are used together to transform
        /// dynamic operations in a computation graph to static operations when the
        /// graph is executed with input data.
        ///
        /// In the example shown below, the original graph is constructed with dynamic
        /// broadcast operation. When the graph is executed with input data, the input
        /// shapes become available, by applying the ConstantFolding and DynElimination
        /// pass, the graph is updated with dynamic broadcast being replaced by a static
        /// broadcast operation.
        /// <table>
        /// <tr>
        ///     <th>Original</th>
        ///     <th>After %ConstantFolding</th>
        ///     <th>After %DynElimination</th>
        /// </tr>
        /// <tr>
        ///      <td> \image html dyn_broadcast_pre_constfld.svg </td>
        ///      <td> \image html dyn_broadcast_post_constfld.svg </td>
        ///      <td> \image html dyn_broadcast_post_dyneliminate.svg </td>
        /// </tr>
        /// </table>
        class NGRAPH_API DynElimination : public GraphRewrite
        {
        public:
            DynElimination();

        private:
            void construct_transpose();
            void construct_dyn_broadcast();
            void construct_dyn_replace_slice();
            void construct_dyn_slice();
            void construct_range();
        };
    }
}
