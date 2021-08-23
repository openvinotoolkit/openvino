// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/node.hpp"
#include "ngraph/output_vector.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/extension.hpp"

namespace ov {

class OPENVINO_API OpExtension : public Extension {
    virtual ngraph::OutputVector create(const ngraph::OutputVector& inputs, ngraph::AttributeVisitor& visitor) = 0;
};

template <class T>
class OPENVINO_API DefaultOpExtension : public OpExtension {
    ngraph::OutputVector create(const ngraph::OutputVector& inputs, ngraph::AttributeVisitor& visitor) override {
        std::shared_ptr<ngraph::Node> node = std::make_shared<T>();

        node->set_arguments(inputs);
        if (node->visit_attributes(visitor)) {
            node->constructor_validate_and_infer_types();
        }
        return node->outputs();
    }
};

}  // namespace ov
