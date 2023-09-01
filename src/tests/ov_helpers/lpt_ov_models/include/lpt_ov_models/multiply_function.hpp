// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>


#include "elementwise_function.hpp"
#include "lpt_ov_models/common/constant.hpp"
#include "lpt_ov_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class MultiplyBranch {
public:
    PartialShape inputShape;
    ov::builder::subgraph::Constant constant;
    ov::element::Type precisionBeforeDequantization;
    ov::builder::subgraph::DequantizationOperations dequantization;
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
    static std::shared_ptr<ov::Model> get(
            const element::Type precision,
            const MultiplyValues& actualValues);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const bool broadcast1,
        const ov::builder::subgraph::FakeQuantizeOnData& fq1,
        const bool broadcast2,
        const ov::builder::subgraph::FakeQuantizeOnData& fq2,
        const ov::builder::subgraph::FakeQuantizeOnData& fqAfter,
        const bool secondInputIsConstant = false);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
