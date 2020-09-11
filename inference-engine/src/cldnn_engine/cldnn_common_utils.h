// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layouts.h>
#include <details/ie_exception.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include <api/layout.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {

#ifndef NDEBUG
#define THROW_CLDNN_EXCEPTION(desc)\
do { \
InferenceEngineException ex(__FILE__, __LINE__);\
std::cout << desc << "\n---\nException detected at " << __FILE__ << ":" << \
__LINE__ << " (" << __FUNCTION__ << ")\n---\n" << std::endl; THROW_IE_EXCEPTION << desc; } while (0);
#else
#define THROW_CLDNN_EXCEPTION(desc) THROW_IE_EXCEPTION << desc;
#endif  // NDEBUG
#define TensorValue(val) static_cast<cldnn::tensor::value_type>(val)

const auto CldnnTensorFromIEDims = [](const InferenceEngine::SizeVector& dims, int def = 1) {
    switch (dims.size()) {
    case 0: return cldnn::tensor(cldnn::batch(def), cldnn::feature(def), cldnn::spatial(def, def));
    case 1: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(def), cldnn::spatial(def, def));
    case 2: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, def));
    case 3: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, dims[2]));
    case 4: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[3], dims[2]));
    case 5: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[4], dims[3], dims[2]));
    case 6: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[5], dims[4], dims[3], dims[2]));
    default: THROW_CLDNN_EXCEPTION("Invalid dimensions size(" << dims.size() << ") for clDNN tensor");
    }
};

inline cldnn::data_types DataTypeFromPrecision(InferenceEngine::Precision p) {
    switch (p) {
    case Precision::I16:
    case Precision::FP32:
        return cldnn::data_types::f32;
    case Precision::FP16:
        return cldnn::data_types::f16;
    case Precision::U8:
        return cldnn::data_types::u8;
    case Precision::I8:
        return cldnn::data_types::i8;
    case Precision::I32:
        return cldnn::data_types::i32;
    case Precision::I64:
        return cldnn::data_types::i64;
    case Precision::BIN:
        return cldnn::data_types::bin;
    case Precision::BOOL:
        return cldnn::data_types::i8;
    default:
        THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "The plugin does not support " << p.name() << " precision";
        break;
    }
}

inline cldnn::format FormatFromLayout(InferenceEngine::Layout l) {
    switch (l) {
        // TODO: change 6d case once new layout added in IE
    case InferenceEngine::Layout::BLOCKED:
        return cldnn::format::bfwzyx;
    case InferenceEngine::Layout::NCDHW:
        return cldnn::format::bfzyx;
    case InferenceEngine::Layout::NCHW:
    case InferenceEngine::Layout::NC:
    case InferenceEngine::Layout::CHW:
    case InferenceEngine::Layout::C:
        return cldnn::format::bfyx;
    case InferenceEngine::Layout::SCALAR:
    case InferenceEngine::Layout::NHWC:
        return cldnn::format::byxf;
    default:
        THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "The plugin does not support " << l << " layout";
        break;
    }
}

inline cldnn::format FormatFromTensorDesc(InferenceEngine::TensorDesc desc) {
    switch (desc.getLayout()) {
    case InferenceEngine::Layout::BLOCKED: {
        if (desc.getDims().size() == 6)
            return cldnn::format::bfwzyx;
        else if (desc.getDims().size() == 5)
            return cldnn::format::bfzyx;
        else if (desc.getDims().size() <= 4)
            return cldnn::format::bfyx;
    }
    case InferenceEngine::Layout::NCDHW:
        return cldnn::format::bfzyx;
    case InferenceEngine::Layout::NCHW:
    case InferenceEngine::Layout::NC:
    case InferenceEngine::Layout::CHW:
    case InferenceEngine::Layout::C:
        return cldnn::format::bfyx;
    case InferenceEngine::Layout::SCALAR:
    case InferenceEngine::Layout::NHWC:
        return cldnn::format::byxf;
    default:
        THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "The plugin does not support " << desc.getLayout() << " layout";
        break;
    }
}

inline cldnn::format ImageFormatFromLayout(InferenceEngine::Layout l) {
    switch (l) {
    // currently, nv12 is the only supported image layout
    case InferenceEngine::Layout::BLOCKED:
    case InferenceEngine::Layout::NCDHW:
    case InferenceEngine::Layout::NCHW:
    case InferenceEngine::Layout::NC:
    case InferenceEngine::Layout::CHW:
    case InferenceEngine::Layout::C:
    case InferenceEngine::Layout::NHWC:
        return cldnn::format::nv12;
    default:
        THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "The plugin does not support " << l << " image layout";
        break;
    }
}


inline cldnn::format defaultFormatForDims(size_t dimensions) {
    switch (dimensions) {
    case 1:
    case 2:
    case 3:
    case 4:
        return cldnn::format::bfyx;
    case 5:
        return cldnn::format::bfzyx;
    case 6:
        return cldnn::format::bfwzyx;
    default:
        THROW_CLDNN_EXCEPTION("Unsupported number of dimensions: " << dimensions);
    }

    return cldnn::format::bfyx;  // Should not get here
}

}  // namespace CLDNNPlugin
