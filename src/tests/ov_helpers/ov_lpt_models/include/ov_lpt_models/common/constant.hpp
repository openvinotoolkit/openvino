// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class Constant {
public:
    Constant();
    Constant(const float value);
    Constant(const std::vector<float>& values);
    Constant(const std::vector<float>& values, const ov::element::Type outPrecision);
    Constant(const std::vector<float>& values, const ov::element::Type outPrecision, const ov::Shape& shape);
    bool empty() const noexcept;

    std::vector<float> values;
    ov::element::Type outPrecision;
    ov::Shape shape;
    bool shapeIsDefined;
private:
    bool isEmpty;
};

inline std::ostream& operator<<(std::ostream& out, const Constant& constant) {
    auto toStream = [](const std::vector<float>& values) -> std::string {
        std::stringstream os;
        os << "{";
        for (size_t i = 0; i < values.size(); ++i) {
            const float& value = values[i];
            if (i > 0) {
                os << value;
            } else {
                os << ", " << value;
            }
        }
        os << "}";
        return os.str();
    };

    return out << "_" << toStream(constant.values) << "_" << constant.outPrecision << "_" << constant.shape;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
