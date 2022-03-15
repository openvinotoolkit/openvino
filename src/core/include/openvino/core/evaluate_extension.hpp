// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/extension.hpp"
#include "openvino/core/node.hpp"

namespace ov {

/**
 * @brief The base interface for OpenVINO operation extensions
 */
class OPENVINO_API EvaluateExtension : public Extension {
public:
    using Ptr = std::shared_ptr<EvaluateExtension>;
    /**
     * @brief Returns the type info of supported operation
     *
     * @return ov::DiscreteTypeInfo
     */
    virtual const ov::DiscreteTypeInfo& get_type_info() const = 0;
    virtual std::vector<ov::Node::SupportedConfig> support_evaluate(
        const std::shared_ptr<const ov::Node>& node) const = 0;
    virtual bool evaluate(const std::shared_ptr<const ov::Node>& node,
                          ov::TensorVector& output_values,
                          const ov::TensorVector& input_values) const = 0;
    virtual bool evaluate(const std::shared_ptr<const ov::Node>& node,
                          ov::TensorVector& output_values,
                          const ov::TensorVector& input_values,
                          const ov::EvaluationContext& evaluationContext) const {
        return evaluate(node, output_values, input_values);
    }

    /**
     * @brief Destructor
     */
    ~EvaluateExtension() override;
};

}  // namespace ov
