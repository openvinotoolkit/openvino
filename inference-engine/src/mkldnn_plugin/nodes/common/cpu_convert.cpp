// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_convert.h"
#include "cpu_memcpy.h"
#include "utils/bfloat16.hpp"
#include <type_traits>
#include <ie_parallel.hpp>

using namespace InferenceEngine;

template<typename srcType, typename dstType>
void convert(const void *srcPtr, void *dstPtr, const size_t size) {
    if (std::is_same<srcType, dstType>::value) {
        cpu_memcpy(dstPtr, srcPtr, size*sizeof(dstType));
    } else {
        const srcType *srcData = reinterpret_cast<const srcType *>(srcPtr);
        dstType *dstData = reinterpret_cast<dstType *>(dstPtr);

        parallel_for(size, [&](size_t i) {
            dstData[i] = static_cast<dstType>(srcData[i]);
        });
    }
}

template <typename srcType>
void convertFrom(const void *srcPtr, void *dstPtr, Precision dstPrc, const size_t size) {
    switch (dstPrc) {
        case Precision::U8:
            convert<srcType, PrecisionTrait<Precision::U8>::value_type>(srcPtr, dstPtr, size);
            break;
        case Precision::I8:
            convert<srcType, PrecisionTrait<Precision::I8>::value_type>(srcPtr, dstPtr, size);
            break;
        case Precision::U16:
            convert<srcType, PrecisionTrait<Precision::U16>::value_type>(srcPtr, dstPtr, size);
            break;
        case Precision::I16:
            convert<srcType, PrecisionTrait<Precision::I16>::value_type>(srcPtr, dstPtr, size);
            break;
        case Precision::I32:
            convert<srcType, PrecisionTrait<Precision::I32>::value_type>(srcPtr, dstPtr, size);
            break;
        case Precision::U64:
            convert<srcType, PrecisionTrait<Precision::U64>::value_type>(srcPtr, dstPtr, size);
            break;
        case Precision::I64:
            convert<srcType, PrecisionTrait<Precision::I64>::value_type>(srcPtr, dstPtr, size);
            break;
        case Precision::FP32:
            convert<srcType, PrecisionTrait<Precision::FP32>::value_type>(srcPtr, dstPtr, size);
            break;
        case Precision::BF16:
            convert<srcType, MKLDNNPlugin::bfloat16_t>(srcPtr, dstPtr, size);
            break;
        case Precision::BOOL:
            convert<srcType, PrecisionTrait<Precision::BOOL>::value_type>(srcPtr, dstPtr, size);
            break;
        default:
            THROW_IE_EXCEPTION << "cpu_convert can't convert to: " << dstPrc << " precision";
    }
}

void cpu_convert(const void *srcPtr, void *dstPtr, Precision srcPrc, Precision dstPrc, const size_t size) {
    if (srcPtr == nullptr || dstPtr == nullptr)
        THROW_IE_EXCEPTION << "cpu_convert has null data pointer";

    if (srcPrc == dstPrc) {
        cpu_memcpy(dstPtr, srcPtr, size*dstPrc.size());
        return;
    }

    switch (srcPrc) {
        case Precision::U8:
            convertFrom<PrecisionTrait<Precision::U8>::value_type>(srcPtr, dstPtr, dstPrc, size);
            break;
        case Precision::I8:
            convertFrom<PrecisionTrait<Precision::I8>::value_type>(srcPtr, dstPtr, dstPrc, size);
            break;
        case Precision::U16:
            convertFrom<PrecisionTrait<Precision::U16>::value_type>(srcPtr, dstPtr, dstPrc, size);
            break;
        case Precision::I16:
            convertFrom<PrecisionTrait<Precision::I16>::value_type>(srcPtr, dstPtr, dstPrc, size);
            break;
        case Precision::I32:
            convertFrom<PrecisionTrait<Precision::I32>::value_type>(srcPtr, dstPtr, dstPrc, size);
            break;
        case Precision::U64:
            convertFrom<PrecisionTrait<Precision::U64>::value_type>(srcPtr, dstPtr, dstPrc, size);
            break;
        case Precision::I64:
            convertFrom<PrecisionTrait<Precision::I64>::value_type>(srcPtr, dstPtr, dstPrc, size);
            break;
        case Precision::FP32:
            convertFrom<PrecisionTrait<Precision::FP32>::value_type>(srcPtr, dstPtr, dstPrc, size);
            break;
        case Precision::BF16:
            convertFrom<MKLDNNPlugin::bfloat16_t>(srcPtr, dstPtr, dstPrc, size);
            break;
        case Precision::BOOL:
            convertFrom<PrecisionTrait<Precision::BOOL>::value_type>(srcPtr, dstPtr, dstPrc, size);
            break;
        default:
            THROW_IE_EXCEPTION << "cpu_convert can't convert from: " << srcPrc << " precision";
    }
}
