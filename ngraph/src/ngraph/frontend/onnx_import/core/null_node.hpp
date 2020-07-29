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

#include <memory>

#include "ngraph/node.hpp"
#include "utils/onnx_importer_visibility.hpp"

namespace ngraph
{
    namespace op
    {
        ONNX_IMPORTER_API
        bool is_null(const ngraph::Node* node);
        ONNX_IMPORTER_API
        bool is_null(const std::shared_ptr<ngraph::Node>& node);
        ONNX_IMPORTER_API
        bool is_null(const Output<ngraph::Node>& output);
    }
    namespace onnx_import
    {
        /// \brief Represents a missing optional input or output of an ONNX node
        ///
        /// Some ONNX operators have inputs or outputs that are marked as optional,
        /// which means that a referring node MAY forgo providing values for such inputs
        /// or computing these outputs.
        /// An empty string is used in place of a name of such input or output.
        ///
        /// More:
        /// https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-inputs-and-outputs
        class NullNode : public ngraph::Node
        {
        public:
            static constexpr NodeTypeInfo type_info{"NullNode", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            NullNode() = default;

            virtual std::shared_ptr<Node>
                clone_with_new_inputs(const OutputVector& new_args) const override;
        };
    } // namespace onnx_import
} // namespace ngraph
