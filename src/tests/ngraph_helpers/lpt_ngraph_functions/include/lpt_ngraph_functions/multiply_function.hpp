// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

#include "elementwise_function.hpp"
#include "lpt_ngraph_functions/common/constant.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MultiplyBranch {
public:
    PartialShape inputShape;
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

class MultiplyFunction : public ElementwiseFunction {
public:
    static std::shared_ptr<ngraph::Function> get(
            const element::Type precision,
            const MultiplyValues& actualValues);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const bool broadcast1,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fq1,
        const bool broadcast2,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fq2,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqAfter,
        const bool secondInputIsConstant = false);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
