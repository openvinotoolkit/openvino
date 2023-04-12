// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "backend/dnn_types.hpp"
#include "gna2-model-api.h"

constexpr uint32_t InOpIdx = 0;
constexpr uint32_t OutOpIdx = 1;
constexpr uint32_t FilterOpIdx = 2;
constexpr uint32_t BiasOpIdx = 3;
constexpr uint32_t PwlOpIdx = 4;
constexpr uint32_t WeightScaleFactorOpIdx = 5;

constexpr uint32_t ConvStrideParamIdx = 0;
constexpr uint32_t BiasModeCnnParamIdx = 1;
constexpr uint32_t BiasModeFCAffineParamIdx = 0;
constexpr uint32_t CopyShapeParamIdx = 0;
constexpr uint32_t DelayParamIdx = 0;
constexpr uint32_t PoolModeParamIdx = 2;
constexpr uint32_t PoolWinParamIdx = 3;
constexpr uint32_t PoolStrideParamIdx = 4;
constexpr uint32_t ZeroPaddingParamIdx = 5;
constexpr uint32_t MaximumScoreParamIdx = 0;

#define GNA_MAX_OP_PARAM 10
typedef void (*GnaUserFree)(void*);

Gna2DataType Gna2DataTypeFromBytes(uint32_t num_bytes_per_input);

void* gnaUserAllocatorAlignedPage(uint32_t size);

void* gnaUserAllocator(uint32_t size);

void gnaUserFree(void* ptr);

Gna2Tensor HelperGna2TensorInit1D(uint32_t x, Gna2DataType dataType, void* data);

Gna2Tensor HelperGna2TensorInit2D(uint32_t x, uint32_t y, Gna2DataType dataType, void* data);

Gna2Tensor HelperGna2TensorInit3D(uint32_t x, uint32_t y, uint32_t z, Gna2DataType dataType, void* data);

Gna2Tensor* createGna2Tensor1D(uint32_t x, uint32_t byteSize, void* data);

Gna2Tensor* createGna2TensorPwl(uint32_t x, void* data);

Gna2Tensor* createGna2BiasTensor1D(uint32_t x, uint32_t byteSize, void* data);

Gna2Tensor* createGna2Tensor(OvGnaTensor tensor, void* data);

Gna2Tensor* createGna2Tensor2D(uint32_t x, uint32_t y, uint32_t byteSize, void* data);

Gna2Tensor* createGna2Tensor3D(uint32_t x, uint32_t y, uint32_t z, uint32_t byteSize, void* data);

uint32_t* create_uint32_parameter(uint32_t value);

Gna2Shape* create_shape1D_parameter(uint32_t x);

Gna2Shape* create_shape2D_parameter(uint32_t x, uint32_t y);

void freeGna2Operation(Gna2Operation& operation);

void HelperGna2OperationInit(Gna2Operation* operation, Gna2OperationType type);

void HelperGna2OperationSetOperand(Gna2Operation* operation,
                                   Gna2UserAllocator userAllocator,
                                   GnaUserFree userFree,
                                   uint32_t index,
                                   Gna2Tensor* inputs);

void HelperGna2OperationSetParameter(Gna2Operation* operation,
                                     Gna2UserAllocator userAllocator,
                                     GnaUserFree userFree,
                                     uint32_t index,
                                     void* param);

void HelperGna2OperationInitElementWiseAffine(Gna2Operation* operation,
                                              Gna2UserAllocator userAllocator,
                                              GnaUserFree userFree,
                                              Gna2Tensor* inputs,
                                              Gna2Tensor* outputs,
                                              Gna2Tensor* weights,
                                              Gna2Tensor* biases,
                                              Gna2Tensor* activation);

void HelperGna2OperationInitFullyConnectedAffine(Gna2Operation* operation,
                                                 Gna2UserAllocator userAllocator,
                                                 GnaUserFree userFree,
                                                 Gna2Tensor* inputs,
                                                 Gna2Tensor* outputs,
                                                 Gna2Tensor* weights,
                                                 Gna2Tensor* biases,
                                                 Gna2Tensor* activation);

void HelperGna2OperationInitRecurrent(Gna2Operation* operation,
                                      Gna2UserAllocator userAllocator,
                                      GnaUserFree userFree,
                                      Gna2Tensor* inputs,
                                      Gna2Tensor* outputs,
                                      Gna2Tensor* weights,
                                      Gna2Tensor* biases,
                                      Gna2Tensor* activation,
                                      uint32_t* delay);

void HelperGna2OperationInitConvolution(Gna2Operation* operation,
                                        Gna2UserAllocator userAllocator,
                                        GnaUserFree userFree,
                                        Gna2Tensor* inputs,
                                        Gna2Tensor* outputs,
                                        Gna2Tensor* filters,
                                        Gna2Tensor* biases,
                                        Gna2Tensor* activation,
                                        Gna2Shape* convolutionStride,
                                        enum Gna2BiasMode* biasMode,
                                        Gna2Shape* zeroPadding);

void HelperGna2OperationInitCopy(Gna2Operation* operation,
                                 Gna2UserAllocator userAllocator,
                                 GnaUserFree userFree,
                                 Gna2Tensor* inputs,
                                 Gna2Tensor* outputs,
                                 Gna2Shape* copyShape);

void HelperGna2OperationInitInterleave(Gna2Operation* operation,
                                       Gna2UserAllocator userAllocator,
                                       GnaUserFree userFree,
                                       Gna2Tensor* inputs,
                                       Gna2Tensor* outputs);

void HelperGna2OperationInitDeInterleave(Gna2Operation* operation,
                                         Gna2UserAllocator userAllocator,
                                         GnaUserFree userFree,
                                         Gna2Tensor* inputs,
                                         Gna2Tensor* outputs);
