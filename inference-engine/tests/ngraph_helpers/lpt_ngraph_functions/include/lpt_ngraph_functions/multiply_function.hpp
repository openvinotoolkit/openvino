// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

#include "lpt_ngraph_functions/common/constant.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MultiplyBranch {
public:
    Shape inputShape;
    ngraph::builder::subgraph::Constant constant;
    ngraph::element::Type precisionBeforeDequantization;
    ngraph::builder::subgraph::DequantizationOperations dequantization;
};

inline std::ostream& operator<<(std::ostream& out, const MultiplyBranch& branch) {
    return out << "_" << branch.constant << "_" << branch.precisionBeforeDequantization << "_" << branch.dequantization;
}

class MultiplyValues {
public:
    MultiplyBranch branch1;
    MultiplyBranch branch2;
    bool isDequantization;
};

inline std::ostream& operator<<(std::ostream& out, const MultiplyValues& values) {
    return out << "_" << values.branch1 << "_" << values.branch2 << (values.isDequantization ? "_isDequantization" : "");
}

class MultiplyFunction {
public:
    static std::shared_ptr<ngraph::Function> get(
            const element::Type precision,
            const ngraph::Shape& inputShape,
            const MultiplyValues& actualValues);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool broadcast,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData1,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData2);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
