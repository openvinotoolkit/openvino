// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if GNA_LIB_VER == 2

#if defined __INTEL_COMPILER || defined _MSC_VER
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#include <gna2-model-api.h>
#include "gna2_model_helper.hpp"
#include "gna_plugin_log.hpp"

Gna2DataType Gna2DataTypeFromBytes(uint32_t num_bytes_per_input) {
    if (num_bytes_per_input == 1)
        return Gna2DataTypeInt8;
    if (num_bytes_per_input == 2)
        return Gna2DataTypeInt16;
    if (num_bytes_per_input == 4)
        return Gna2DataTypeInt32;
    if (num_bytes_per_input == 0)
        return Gna2DataTypeNone;
    THROW_GNA_EXCEPTION << "Not supported num_bytes_per_input: " << num_bytes_per_input;
}

void* gnaUserAllocatorAlignedPage(uint32_t size) {
    return _mm_malloc(size, 4096);
}

void* gnaUserAllocator(uint32_t size) {
    return _mm_malloc(size, 64);
}

void gnaUserFree(void* ptr) {
    _mm_free(ptr);
}

Gna2Tensor HelperGna2TensorInit1D(uint32_t x, Gna2DataType dataType, void* data) {
    Gna2Tensor t{};
    t.Type = dataType;
    t.Data = data;
    t.Shape = { 1, {x} };
    return t;
}

Gna2Tensor HelperGna2TensorInit2D(uint32_t x, uint32_t y, Gna2DataType dataType, void* data) {
    auto t = HelperGna2TensorInit1D(x, dataType, data);
    t.Shape = { 2, {x, y} };
    return t;
}

Gna2Tensor HelperGna2TensorInit3D(uint32_t x, uint32_t y, uint32_t z, Gna2DataType dataType, void* data) {
    auto t = HelperGna2TensorInit1D(x, dataType, data);
    t.Shape = { 3, {x, y, z} };
    return t;
}

Gna2Tensor * createGna2Tensor1D(uint32_t x, uint32_t byteSize, void* data) {
    const auto input = reinterpret_cast<Gna2Tensor*>(gnaUserAllocator(sizeof(Gna2Tensor)));
    IE_ASSERT(input != nullptr);
    *input = HelperGna2TensorInit1D(x, Gna2DataTypeFromBytes(byteSize), data);
    return input;
}

Gna2Tensor * createGna2TensorPwl(uint32_t x, void* data) {
    auto ret = createGna2Tensor1D(x, 1, data);
    ret->Type = Gna2DataTypePwlSegment;
    if (data == nullptr)
        ret->Mode = Gna2TensorModeDisabled;
    return ret;
}

Gna2Tensor * createGna2BiasTensor1D(uint32_t x, uint32_t byteSize, void* data) {
    const auto input = reinterpret_cast<Gna2Tensor*>(gnaUserAllocator(sizeof(Gna2Tensor)));
    IE_ASSERT(input != nullptr);
    if (byteSize == 8) {
        *input = HelperGna2TensorInit1D(x, Gna2DataTypeCompoundBias, data);
    } else {
        *input = HelperGna2TensorInit1D(x, Gna2DataTypeFromBytes(byteSize), data);
    }
    return input;
}

Gna2Tensor * createGna2Tensor2D(uint32_t x, uint32_t y, uint32_t byteSize, void* data) {
    const auto input = reinterpret_cast<Gna2Tensor*>(gnaUserAllocator(sizeof(Gna2Tensor)));
    IE_ASSERT(input != nullptr);
    *input = HelperGna2TensorInit2D(x, y, Gna2DataTypeFromBytes(byteSize), data);
    return input;
}

Gna2Tensor * createGna2Tensor3D(uint32_t x, uint32_t y, uint32_t z, uint32_t byteSize, void* data) {
    const auto input = reinterpret_cast<Gna2Tensor*>(gnaUserAllocator(sizeof(Gna2Tensor)));
    IE_ASSERT(input != nullptr);
    *input = HelperGna2TensorInit3D(x, y, z, Gna2DataTypeFromBytes(byteSize), data);
    return input;
}

uint32_t* create_uint32_parameter(uint32_t value) {
    const auto param = reinterpret_cast<uint32_t*>(gnaUserAllocator(sizeof(uint32_t)));
    IE_ASSERT(param != nullptr);
    *param = value;
    return param;
}

Gna2Shape* create_shape1D_parameter(uint32_t x) {
    const auto shp = reinterpret_cast<Gna2Shape*>(gnaUserAllocator(sizeof(Gna2Shape)));
    IE_ASSERT(shp != nullptr);
    shp->NumberOfDimensions = 1;
    shp->Dimensions[0] = x;
    return shp;
}

Gna2Shape* create_shape2D_parameter(uint32_t x, uint32_t y) {
    const auto shp = create_shape1D_parameter(x);
    shp->NumberOfDimensions++;
    shp->Dimensions[1] = y;
    return shp;
}

void freeGna2Operation(Gna2Operation& operation) {
    if (operation.Operands != nullptr) {
        for (auto i = 0; i < operation.NumberOfOperands; i++) {
            if (operation.Operands[i] != nullptr) {
                gnaUserFree(const_cast<Gna2Tensor*>(operation.Operands[i]));
                operation.Operands[i] = nullptr;
            }
        }
        gnaUserFree(operation.Operands);
        operation.Operands = nullptr;
        operation.NumberOfOperands = 0;
    }
    if (operation.Parameters != nullptr) {
        for (auto i = 0; i < operation.NumberOfParameters; i++) {
            if (operation.Parameters[i] != nullptr) {
                gnaUserFree(operation.Parameters[i]);
                operation.Parameters[i] = nullptr;
            }
        }
        gnaUserFree(operation.Parameters);
        operation.Parameters = nullptr;
        operation.NumberOfParameters = 0;
    }
    operation.Type = Gna2OperationTypeNone;
}

void HelperGna2OperationInit(Gna2Operation * operation, Gna2OperationType type) {
    operation->Type = type;
    operation->NumberOfOperands = 0;
    operation->NumberOfParameters = 0;
}

void HelperGna2OperationSetOperand(Gna2Operation * operation,
    Gna2UserAllocator userAllocator, GnaUserFree userFree,
    uint32_t index, Gna2Tensor * inputs) {
    if (index >= GNA_MAX_OP_PARAM) {
        THROW_GNA_EXCEPTION << "HelperGna2OperationSetOperand: index >= GNA_MAX_OP_PARAM";
    }
    if (operation->NumberOfOperands <= index) {
        const auto o = reinterpret_cast<Gna2Tensor const **>(userAllocator(sizeof(Gna2Tensor*) * (index + 1)));
        for (unsigned i = 0; i < operation->NumberOfOperands; i++) {
            o[i] = operation->Operands[i];
        }
        for (auto i = operation->NumberOfOperands; i <= index; i++) {
            o[i] = nullptr;
        }
        operation->NumberOfOperands = index + 1;
        userFree(operation->Operands);
        operation->Operands = o;
    }
    operation->Operands[index] = inputs;
}

void HelperGna2OperationSetParameter(Gna2Operation * operation,
    Gna2UserAllocator userAllocator, GnaUserFree userFree,
    uint32_t index, void * param) {
    if (index >= GNA_MAX_OP_PARAM) {
        THROW_GNA_EXCEPTION << "HelperGna2OperationSetParameter: (index >= GNA_MAX_OP_PARAM) index=" << index <<" GNA_MAX_OP_PARAM=" << GNA_MAX_OP_PARAM;
    }
    if (operation->NumberOfParameters <= index) {
        const auto p = reinterpret_cast<void **>(userAllocator(sizeof(void *) * (index + 1)));
        for (unsigned i = 0; i < operation->NumberOfParameters; i++) {
            p[i] = operation->Parameters[i];
        }
        for (unsigned i = operation->NumberOfParameters; i <= index; i++) {
            p[i] = nullptr;
        }
        operation->NumberOfParameters = index + 1;
        userFree(operation->Parameters);
        operation->Parameters = p;
    }
    operation->Parameters[index] = param;
}

void HelperGna2OperationInitElementWiseAffine(Gna2Operation * operation,
    Gna2UserAllocator userAllocator, GnaUserFree userFree,
    Gna2Tensor * inputs, Gna2Tensor * outputs,
    Gna2Tensor * weights, Gna2Tensor * biases,
    Gna2Tensor * activation) {
    HelperGna2OperationInit(operation, Gna2OperationTypeElementWiseAffine);
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, InOpIdx, inputs);
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, OutOpIdx, outputs);
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, FilterOpIdx, weights);
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, BiasOpIdx, biases);
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, PwlOpIdx, activation);
}

void HelperGna2OperationInitFullyConnectedAffine(Gna2Operation * operation,
    Gna2UserAllocator userAllocator, GnaUserFree userFree,
    Gna2Tensor * inputs, Gna2Tensor * outputs,
    Gna2Tensor * weights, Gna2Tensor * biases,
    Gna2Tensor * activation) {
    HelperGna2OperationInitElementWiseAffine(operation,
        userAllocator, userFree,
        inputs, outputs, weights, biases, activation);
    operation->Type = Gna2OperationTypeFullyConnectedAffine;
    // TODO: GNA2: remove when GNA2 library does not expect optional operands/parameters to be provided
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, WeightScaleFactorOpIdx, nullptr);
    HelperGna2OperationSetParameter(operation, userAllocator, userFree, BiasModeFCAffineParamIdx, nullptr);
}

void HelperGna2OperationInitRecurrent(Gna2Operation * operation,
    Gna2UserAllocator userAllocator, GnaUserFree userFree,
    Gna2Tensor * inputs, Gna2Tensor * outputs,
    Gna2Tensor * weights, Gna2Tensor * biases,
    Gna2Tensor * activation,
    uint32_t * delay) {
    HelperGna2OperationInitElementWiseAffine(operation, userAllocator, userFree, inputs, outputs, weights, biases, activation);
    operation->Type = Gna2OperationTypeRecurrent;
    HelperGna2OperationSetParameter(operation, userAllocator, userFree, DelayParamIdx, delay);
}

void HelperGna2OperationInitConvolution(Gna2Operation * operation,
    Gna2UserAllocator userAllocator, GnaUserFree userFree,
    Gna2Tensor * inputs, Gna2Tensor * outputs,
    Gna2Tensor * filters, Gna2Tensor * biases,
    Gna2Tensor * activation,
    Gna2Shape * convolutionStride,
    Gna2BiasMode * biasMode) {
    HelperGna2OperationInitElementWiseAffine(operation, userAllocator, userFree, inputs, outputs, filters, biases, activation);
    operation->Type = Gna2OperationTypeConvolution;
    HelperGna2OperationSetParameter(operation, userAllocator, userFree, ConvStrideParamIdx, convolutionStride);
    HelperGna2OperationSetParameter(operation, userAllocator, userFree, BiasModeCnnParamIdx, biasMode);
}

void HelperGna2OperationInitCopy(Gna2Operation * operation,
    Gna2UserAllocator userAllocator, GnaUserFree userFree,
    Gna2Tensor * inputs, Gna2Tensor * outputs,
    Gna2Shape * copyShape) {
    HelperGna2OperationInit(operation, Gna2OperationTypeCopy);
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, InOpIdx, inputs);
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, OutOpIdx, outputs);
    HelperGna2OperationSetParameter(operation, userAllocator, userFree, CopyShapeParamIdx, copyShape);
}

void HelperGna2OperationInitInterleave(Gna2Operation * operation,
    Gna2UserAllocator userAllocator, GnaUserFree userFree,
    Gna2Tensor * inputs, Gna2Tensor * outputs) {
    HelperGna2OperationInit(operation, Gna2OperationTypeTransposition);
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, InOpIdx, inputs);
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, OutOpIdx, outputs);
}

void HelperGna2OperationInitDeInterleave(Gna2Operation * operation,
    Gna2UserAllocator userAllocator, GnaUserFree userFree,
    Gna2Tensor * inputs, Gna2Tensor * outputs) {
    HelperGna2OperationInit(operation, Gna2OperationTypeTransposition);
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, InOpIdx, inputs);
    HelperGna2OperationSetOperand(operation, userAllocator, userFree, OutOpIdx, outputs);
}

#endif
