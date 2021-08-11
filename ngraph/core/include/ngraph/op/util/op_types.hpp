// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"

namespace ov
{
    namespace op
    {
        NGRAPH_API
        bool is_unary_elementwise_arithmetic(const ov::Node* node);
        NGRAPH_API
        bool is_binary_elementwise_arithmetic(const ov::Node* node);
        NGRAPH_API
        bool is_binary_elementwise_comparison(const ov::Node* node);
        NGRAPH_API
        bool is_binary_elementwise_logical(const ov::Node* node);

        NGRAPH_API
        bool supports_auto_broadcast(const ov::Node* node);

        NGRAPH_API
        bool supports_decompose(const ov::Node* node);

        NGRAPH_API
        bool is_op(const ov::Node* node);
        NGRAPH_API
        bool is_parameter(const ov::Node* node);
        NGRAPH_API
        bool is_output(const ov::Node* node);
        NGRAPH_API
        bool is_sink(const ov::Node* node);
        NGRAPH_API
        bool is_constant(const ov::Node* node);
        NGRAPH_API
        bool is_commutative(const ov::Node* node);

        NGRAPH_API
        bool is_unary_elementwise_arithmetic(const std::shared_ptr<ov::Node>& node);
        NGRAPH_API
        bool is_binary_elementwise_arithmetic(const std::shared_ptr<ov::Node>& node);
        NGRAPH_API
        bool is_binary_elementwise_comparison(const std::shared_ptr<ov::Node>& node);
        NGRAPH_API
        bool is_binary_elementwise_logical(const std::shared_ptr<ov::Node>& node);

        NGRAPH_API
        bool supports_auto_broadcast(const std::shared_ptr<ov::Node>& node);

        NGRAPH_API
        bool supports_decompose(const std::shared_ptr<ov::Node>& node);

        NGRAPH_API
        bool is_op(const std::shared_ptr<ov::Node>& node);
        NGRAPH_API
        bool is_parameter(const std::shared_ptr<ov::Node>& node);
        NGRAPH_API
        bool is_output(const std::shared_ptr<ov::Node>& node);
        NGRAPH_API
        bool is_sink(const std::shared_ptr<ov::Node>& node);
        NGRAPH_API
        bool is_constant(const std::shared_ptr<ov::Node>& node);
        NGRAPH_API
        bool is_commutative(const std::shared_ptr<ov::Node>& node);
    } // namespace op
} // namespace ov
