// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../graph_converter.hpp"
// #include <openvino/op/add.hpp>
// #include <openvino/op/divide.hpp>
// #include <openvino/op/multiply.hpp>
// #include <openvino/op/subtract.hpp>

namespace ov::intel_gpu::mlir {

class ReluPattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReluPattern");
    ReluPattern();
};

class ConcatPattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConcatPattern");
    ConcatPattern();
};

class FloorPattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("FloorPattern");
    FloorPattern();
};

class GatherPattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("GatherPattern");
    GatherPattern();
};

class MatMulPattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("MatMulPattern");
    MatMulPattern();
};

template <typename OVOp>
class ReducePattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReducePattern");
    ReducePattern();
};

class RMSPattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("RMSPattern");
    RMSPattern();
};

class ReshapePattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReshapePattern");
    ReshapePattern();
};

class SDPAPattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPAPattern");
    SDPAPattern();
};

class ShapeOfPattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("ShapeOfPattern");
    ShapeOfPattern();
};

class SlicePattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("SlicePattern");
    SlicePattern();
};

class SqueezePattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("SqueezePattern");
    SqueezePattern();
};

class TransposePattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposePattern");
    TransposePattern();
};

class UnsqueezePattern : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("UnsqueezePattern");
    UnsqueezePattern();
};

class BinaryEltwisePatternBase : public MarkPattern {
public:
    OPENVINO_MATCHER_PASS_RTTI("BinaryEltwisePatternBase");
    BinaryEltwisePatternBase(NodeTypeInfo wrapped_type, const GraphConverter::Convertor& convertor, const std::set<element::Type>& element_types = {});
};

template <typename OVOp, typename LinalgOp>
class BinaryEltwisePattern : public BinaryEltwisePatternBase {
public:
    BinaryEltwisePattern(const std::set<element::Type>& element_types = {});

    BinaryEltwisePattern(const element::Type& element_type) : BinaryEltwisePattern(std::set<element::Type>{element_type}) {}
};

template <typename OVOp, typename LinalgOp>
class UnaryEltwisePattern : public MarkPattern {
public:
    UnaryEltwisePattern();
};

}  // namespace ov::intel_gpu::mlir
