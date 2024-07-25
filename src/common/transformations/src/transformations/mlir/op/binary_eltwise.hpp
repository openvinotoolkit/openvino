// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../conversion_context.hpp"

namespace ov {
namespace mlir {

class BinaryEltwisePatternBase : public MarkPattern {
public:
    using Builder = std::function<Operation*(OpBuilder&, ::mlir::Location, ValueRange, ValueRange)>;

    OPENVINO_RTTI("BinaryEltwisePatternBase", "0");
    BinaryEltwisePatternBase(NodeTypeInfo wrapped_type, Builder op_builder, const std::set<element::Type>& element_types = {});
};


template <typename OVOp, typename LinalgOp>
class BinaryEltwisePattern : public BinaryEltwisePatternBase {
public:
    // Allow conversion for given `element_types` only, except case when `element_types` is empty which means no restrictions on types, everything is allowed.
    BinaryEltwisePattern (const std::set<element::Type>& element_types = {}) :
        BinaryEltwisePatternBase(
            OVOp::get_type_info_static(),
            [](OpBuilder& builder, ::mlir::Location loc, ValueRange ins, ValueRange outs) -> Operation* {
                return builder.create<LinalgOp>(loc, ins, outs);
            },
            element_types)
    {}

    BinaryEltwisePattern (const element::Type& element_type) :
        BinaryEltwisePattern(std::set<element::Type>{element_type})
    {}
};


}  // namespace mlir
}  // namespace ov

