// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../graph_converter.hpp"
// #include <openvino/op/add.hpp>
// #include <openvino/op/divide.hpp>
// #include <openvino/op/multiply.hpp>
// #include <openvino/op/subtract.hpp>

namespace ov {
namespace mlir {

class ReluPattern : public MarkPattern {
public:
    OPENVINO_RTTI("ReluPattern", "0");
    ReluPattern();
};

class ConcatPattern : public MarkPattern {
public:
    OPENVINO_RTTI("ConcatPattern", "0");
    ConcatPattern();
};

class FloorPattern : public MarkPattern {
public:
    OPENVINO_RTTI("FloorPattern", "0");
    FloorPattern();
};

class GatherPattern : public MarkPattern {
public:
    OPENVINO_RTTI("GatherPattern", "0");
    GatherPattern();
};

class MatMulPattern : public MarkPattern {
public:
    OPENVINO_RTTI("MatMulPattern", "0");
    MatMulPattern();
};

template <typename OVOp>
class ReducePattern : public MarkPattern {
public:
    OPENVINO_RTTI("ReducePattern", "0");
    ReducePattern();
};

class SDPAPattern : public MarkPattern {
public:
    OPENVINO_RTTI("SDPAPattern", "0");
    SDPAPattern();
};

class ShapeOfPattern : public MarkPattern {
public:
    OPENVINO_RTTI("ShapeOfPattern", "0");
    ShapeOfPattern();
};

class SlicePattern : public MarkPattern {
public:
    OPENVINO_RTTI("SlicePattern", "0");
    SlicePattern();
};

class SqueezePattern : public MarkPattern {
public:
    OPENVINO_RTTI("SqueezePattern", "0");
    SqueezePattern();
};

class TransposePattern : public MarkPattern {
public:
    OPENVINO_RTTI("TransposePattern", "0");
    TransposePattern();
};

class UnsqueezePattern : public MarkPattern {
public:
    OPENVINO_RTTI("UnsqueezePattern", "0");
    UnsqueezePattern();
};

class BinaryEltwisePatternBase : public MarkPattern {
public:
    OPENVINO_RTTI("BinaryEltwisePatternBase", "0");
    BinaryEltwisePatternBase(NodeTypeInfo wrapped_type, GraphConverter::Convertor convertor,
                             const std::set<element::Type>& element_types = {});
};

template <typename OVOp, typename LinalgOp>
class BinaryEltwisePattern : public BinaryEltwisePatternBase {
public:
    BinaryEltwisePattern(const std::set<element::Type>& element_types = {});

    BinaryEltwisePattern(const element::Type& element_type)
        : BinaryEltwisePattern(std::set<element::Type>{element_type}) {}
};

}  // namespace mlir
}  // namespace ov
