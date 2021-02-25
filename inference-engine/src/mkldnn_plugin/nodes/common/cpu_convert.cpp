// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_convert.h"
#include "cpu_memcpy.h"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>
#include <type_traits>
#include <tuple>
#include <ie_parallel.hpp>
#include <ngraph/type/float16.hpp>

using namespace InferenceEngine;

namespace {

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

template <Precision::ePrecision p>
struct PrecisionInfo {
    using value_type = typename PrecisionTrait<p>::value_type;
};

template <>
struct PrecisionInfo<Precision::BF16> {
    using value_type = MKLDNNPlugin::bfloat16_t;
};

template <>
struct PrecisionInfo<Precision::FP16> {
    using value_type = ngraph::float16;
};

struct ConvertContext {
    const void *srcPtr;
    void *dstPtr;
    size_t size;
    bool converted;
};

template<typename T>
struct ConvertPrecision {
    using src_t = typename std::tuple_element<0, T>::type;
    using dst_t = typename std::tuple_element<1, T>::type;

    void operator()(ConvertContext & ctx) {
        convert<src_t, dst_t>(ctx.srcPtr, ctx.dstPtr, ctx.size);
        ctx.converted = true;
    }
};

}   // namespace

#define MKLDNN_CVT(ST, DT) OV_CASE2(Precision::ST, Precision::DT, PrecisionInfo<Precision::ST>::value_type, PrecisionInfo<Precision::DT>::value_type)

void cpu_convert(const void *srcPtr, void *dstPtr, Precision srcPrc, Precision dstPrc, const size_t size) {
    using namespace MKLDNNPlugin;

    if (srcPtr == nullptr || dstPtr == nullptr)
        THROW_IE_EXCEPTION << "cpu_convert has null data pointer";

    if (srcPrc == dstPrc) {
        cpu_memcpy(dstPtr, srcPtr, size*dstPrc.size());
        return;
    }

    ConvertContext ctx = { srcPtr, dstPtr, size, false };

    OV_SWITCH(MKLDNNPlugin, ConvertPrecision, ctx, std::tie(srcPrc, dstPrc),
    MKLDNN_CVT(U8, I8),    MKLDNN_CVT(U8, U16),    MKLDNN_CVT(U8, I16),
    MKLDNN_CVT(U8, I32),   MKLDNN_CVT(U8, U64),    MKLDNN_CVT(U8, I64),
    MKLDNN_CVT(U8, FP32),  MKLDNN_CVT(U8, BF16),   MKLDNN_CVT(U8, BOOL),
    MKLDNN_CVT(I8, U8),    MKLDNN_CVT(I8, U16),    MKLDNN_CVT(I8, I16),
    MKLDNN_CVT(I8, I32),   MKLDNN_CVT(I8, U64),    MKLDNN_CVT(I8, I64),
    MKLDNN_CVT(I8, FP32),  MKLDNN_CVT(I8, BF16),   MKLDNN_CVT(I8, BOOL),
    MKLDNN_CVT(U16, U8),   MKLDNN_CVT(U16, I8),    MKLDNN_CVT(U16, I16),
    MKLDNN_CVT(U16, I32),  MKLDNN_CVT(U16, U64),   MKLDNN_CVT(U16, I64),
    MKLDNN_CVT(U16, FP32), MKLDNN_CVT(U16, BF16),  MKLDNN_CVT(U16, BOOL),
    MKLDNN_CVT(I16, U8),   MKLDNN_CVT(I16, I8),    MKLDNN_CVT(I16, U16),
    MKLDNN_CVT(I16, I32),  MKLDNN_CVT(I16, U64),   MKLDNN_CVT(I16, I64),
    MKLDNN_CVT(I16, FP32), MKLDNN_CVT(I16, BF16),  MKLDNN_CVT(I16, BOOL),
    MKLDNN_CVT(I32, U8),   MKLDNN_CVT(I32, I8),    MKLDNN_CVT(I32, U16),
    MKLDNN_CVT(I32, I16),  MKLDNN_CVT(I32, U64),   MKLDNN_CVT(I32, I64),
    MKLDNN_CVT(I32, FP32), MKLDNN_CVT(I32, BF16),  MKLDNN_CVT(I32, BOOL),
    MKLDNN_CVT(U64, U8),   MKLDNN_CVT(U64, I8),    MKLDNN_CVT(U64, U16),
    MKLDNN_CVT(U64, I16),  MKLDNN_CVT(U64, I32),   MKLDNN_CVT(U64, I64),
    MKLDNN_CVT(U64, FP32), MKLDNN_CVT(U64, BF16),  MKLDNN_CVT(U64, BOOL),
    MKLDNN_CVT(I64, U8),   MKLDNN_CVT(I64, I8),    MKLDNN_CVT(I64, U16),
    MKLDNN_CVT(I64, I16),  MKLDNN_CVT(I64, I32),   MKLDNN_CVT(I64, U64),
    MKLDNN_CVT(I64, FP32), MKLDNN_CVT(I64, BF16),  MKLDNN_CVT(I64, BOOL),
    MKLDNN_CVT(FP32, U8),  MKLDNN_CVT(FP32, I8),   MKLDNN_CVT(FP32, U16),
    MKLDNN_CVT(FP32, I16), MKLDNN_CVT(FP32, I32),  MKLDNN_CVT(FP32, U64),
    MKLDNN_CVT(FP32, I64), MKLDNN_CVT(FP32, BF16), MKLDNN_CVT(FP32, BOOL),
    MKLDNN_CVT(BF16, U8),  MKLDNN_CVT(BF16, I8),   MKLDNN_CVT(BF16, U16),
    MKLDNN_CVT(BF16, I16), MKLDNN_CVT(BF16, I32),  MKLDNN_CVT(BF16, U64),
    MKLDNN_CVT(BF16, I64), MKLDNN_CVT(BF16, FP32), MKLDNN_CVT(BF16, BOOL),
    MKLDNN_CVT(BOOL, U8),  MKLDNN_CVT(BOOL, I8),   MKLDNN_CVT(BOOL, U16),
    MKLDNN_CVT(BOOL, I16), MKLDNN_CVT(BOOL, I32),  MKLDNN_CVT(BOOL, U64),
    MKLDNN_CVT(BOOL, I64), MKLDNN_CVT(BOOL, FP32), MKLDNN_CVT(BOOL, BF16),
    MKLDNN_CVT(U8, FP16),  MKLDNN_CVT(I8, FP16),   MKLDNN_CVT(U16, FP16),
    MKLDNN_CVT(I16, FP16), MKLDNN_CVT(I32, FP16),  MKLDNN_CVT(U64, FP16),
    MKLDNN_CVT(I64, FP16), MKLDNN_CVT(FP32, FP16), MKLDNN_CVT(BOOL, FP16),
    MKLDNN_CVT(FP16, U8),  MKLDNN_CVT(FP16, I8),   MKLDNN_CVT(FP16, U16),
    MKLDNN_CVT(FP16, I16), MKLDNN_CVT(FP16, I32),  MKLDNN_CVT(FP16, U64),
    MKLDNN_CVT(FP16, I64), MKLDNN_CVT(FP16, FP32), MKLDNN_CVT(FP16, BOOL));

    if (!ctx.converted)
        THROW_IE_EXCEPTION << "cpu_convert can't convert from: " << srcPrc << " precision to: " << dstPrc;
}

#undef MKLDNN_CVT
