// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace op
    {
        NGRAPH_API
        bool is_unary_elementwise_arithmetic(const ngraph::Node* node);
        NGRAPH_API
        bool is_binary_elementwise_arithmetic(const ngraph::Node* node);
        NGRAPH_API
        bool is_binary_elementwise_comparison(const ngraph::Node* node);
        NGRAPH_API
        bool is_binary_elementwise_logical(const ngraph::Node* node);

        NGRAPH_API
        bool supports_auto_broadcast(const ngraph::Node* node);

        NGRAPH_API
        bool supports_decompose(const ngraph::Node* node);

        NGRAPH_API
        bool is_op(const ngraph::Node* node);
        NGRAPH_API
        bool is_parameter(const ngraph::Node* node);
        NGRAPH_API
        bool is_output(const ngraph::Node* node);
        NGRAPH_API
        bool is_sink(const ngraph::Node* node);
        NGRAPH_API
        bool is_constant(const ngraph::Node* node);
        NGRAPH_API
        bool is_commutative(const ngraph::Node* node);

        NGRAPH_API
        bool is_unary_elementwise_arithmetic(const std::shared_ptr<ngraph::Node>& node);
        NGRAPH_API
        bool is_binary_elementwise_arithmetic(const std::shared_ptr<ngraph::Node>& node);
        NGRAPH_API
        bool is_binary_elementwise_comparison(const std::shared_ptr<ngraph::Node>& node);
        NGRAPH_API
        bool is_binary_elementwise_logical(const std::shared_ptr<ngraph::Node>& node);

        NGRAPH_API
        bool supports_auto_broadcast(const std::shared_ptr<ngraph::Node>& node);

        NGRAPH_API
        bool supports_decompose(const std::shared_ptr<ngraph::Node>& node);

        NGRAPH_API
        bool is_op(const std::shared_ptr<ngraph::Node>& node);
        NGRAPH_API
        bool is_parameter(const std::shared_ptr<ngraph::Node>& node);
        NGRAPH_API
        bool is_output(const std::shared_ptr<ngraph::Node>& node);
        NGRAPH_API
        bool is_sink(const std::shared_ptr<ngraph::Node>& node);
        NGRAPH_API
        bool is_constant(const std::shared_ptr<ngraph::Node>& node);
        NGRAPH_API
        bool is_commutative(const std::shared_ptr<ngraph::Node>& node);
    } // namespace op
} // namespace ngraph
