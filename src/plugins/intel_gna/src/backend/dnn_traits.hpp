// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// dnn_traits.hpp : c++ trait approach to  define dnn objects
//

#pragma once

#include "dnn_types.h"

template<intel_dnn_operation_t layer>
struct DnnTrait {};

template<>
struct DnnTrait<kDnnDiagonalOp> {
    using Type = intel_affine_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.affine;
    }
};

template<>
struct DnnTrait<kDnnPiecewiselinearOp> {
    using Type = intel_piecewiselinear_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.pwl;
    }
};

template<>
struct DnnTrait<kDnnAffineOp> {
    using Type = intel_affine_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.affine;
    }
};

template<>
struct DnnTrait<kDnnConvolutional1dOp> {
    using Type = intel_convolutionalD_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.conv1D;
    }
};

template<>
struct DnnTrait<kDnnMaxPoolOp> {
    using Type = intel_maxpool_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.maxpool;
    }
};

template<>
struct DnnTrait<kDnnRecurrentOp> {
    using Type = intel_recurrent_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.recurrent;
    }
};

template<>
struct DnnTrait<kDnnInterleaveOp> {
    using Type = intel_interleave_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.interleave;
    }
};

template<>
struct DnnTrait<kDnnDeinterleaveOp> {
    using Type = intel_deinterleave_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.deinterleave;
    }
};

template<>
struct DnnTrait<kDnnCopyOp> {
    using Type = intel_copy_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.copy;
    }
};

template<>
struct DnnTrait<kDnnNullOp> {
    using Type = void;
    static Type *getLayer(intel_dnn_component_t &component) {
        return nullptr;
    }
};
