// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "onnx_import/utils/onnx_importer_visibility.hpp"

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
    } // namespace op
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
            NullNode()
                : Node(1)
            {
            }

            virtual std::shared_ptr<Node>
                clone_with_new_inputs(const OutputVector& new_args) const override;
        };
    } // namespace onnx_import
} // namespace ngraph
