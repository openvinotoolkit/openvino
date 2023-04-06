// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "helper_ops/internal_operation.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class FIFOQueue : public InternalOperation {
public:
    OPENVINO_OP("FIFOQueue", "ov::frontend::tensorflow", InternalOperation);

    FIFOQueue(const std::vector<ov::element::Type>& component_types,
              const std::vector<ov::PartialShape>& shapes,
              const int64_t capacity,
              const std::string& container,
              const std::string& shared_name,
              const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : InternalOperation(decoder, OutputVector{}, 1),
          m_component_types(component_types),
          m_shapes(shapes),
          m_container(container),
          m_shared_name(shared_name) {
        validate_and_infer_types();
    }

    std::vector<ov::element::Type> get_component_types() const {
        return m_component_types;
    }

    std::vector<ov::PartialShape> get_component_shapes() const {
        return m_shapes;
    }

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::dynamic, ov::PartialShape::dynamic());
    }

private:
    std::vector<ov::element::Type> m_component_types;
    std::vector<ov::PartialShape> m_shapes;
    std::string m_container;
    std::string m_shared_name;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
