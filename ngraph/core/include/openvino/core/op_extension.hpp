// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"

namespace ov {

class OPENVINO_API BaseOpExtension : public Extension {
    virtual ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) = 0;
};

template <class T>
class OPENVINO_EXTENSION_API OpExtension : public BaseOpExtension {
    ngraph::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) override {
        std::shared_ptr<ngraph::Node> node = std::make_shared<T>();

        node->set_arguments(inputs);
        if (node->visit_attributes(visitor)) {
            node->constructor_validate_and_infer_types();
        }
        return node->outputs();
    }
};

}  // namespace ov
