// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/data_utils.hpp"

#include "blob_factory.hpp"
#include "debug.h"  // to allow putting vector into exception string stream
#include "ie_blob.h"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/tensor.hpp"
#include "precomp.hpp"

using namespace InferenceEngine::details;

namespace ov {
namespace test {
namespace utils {

OPENVINO_SUPPRESS_DEPRECATED_START

bool isDenseBlob(const InferenceEngine::Blob::Ptr& blob) {
    auto blk_desc = blob->getTensorDesc().getBlockingDesc();
    auto dims = blk_desc.getBlockDims();
    auto strs = blk_desc.getStrides();

    IE_ASSERT(dims.size() == strs.size()) << " isDenseBlob: inconsistent tensor descriptor";

    auto size = dims.size();
    if (size == 0)
        return true;
    if (size == 1)
        return strs[0] == 1;

    for (auto i = size - 1; i > 0; i--) {
        if (strs[i - 1] != strs[i - 1] * dims[i])
            return false;
    }

    return true;
}

template <typename T>
void copy_7D(void* src_raw_ptr,
             std::vector<size_t>& src_str,
             void* dst_raw_ptr,
             std::vector<size_t>& dst_str,
             std::vector<size_t>& dims) {
    auto src_ptr = static_cast<T*>(src_raw_ptr);
    auto dst_ptr = static_cast<T*>(dst_raw_ptr);

    for (size_t d0 = 0; d0 < dims[0]; d0++) {
        auto src_ptr_0 = src_ptr + src_str[0] * d0;
        auto dst_ptr_0 = dst_ptr + dst_str[0] * d0;
        for (size_t d1 = 0; d1 < dims[1]; d1++) {
            auto src_ptr_1 = src_ptr_0 + src_str[1] * d1;
            auto dst_ptr_1 = dst_ptr_0 + dst_str[1] * d1;
            for (size_t d2 = 0; d2 < dims[2]; d2++) {
                auto src_ptr_2 = src_ptr_1 + src_str[2] * d2;
                auto dst_ptr_2 = dst_ptr_1 + dst_str[2] * d2;
                for (size_t d3 = 0; d3 < dims[3]; d3++) {
                    auto src_ptr_3 = src_ptr_2 + src_str[3] * d3;
                    auto dst_ptr_3 = dst_ptr_2 + dst_str[3] * d3;
                    for (size_t d4 = 0; d4 < dims[4]; d4++) {
                        auto src_ptr_4 = src_ptr_3 + src_str[4] * d4;
                        auto dst_ptr_4 = dst_ptr_3 + dst_str[4] * d4;
                        for (size_t d5 = 0; d5 < dims[5]; d5++) {
                            auto src_ptr_5 = src_ptr_4 + src_str[5] * d5;
                            auto dst_ptr_5 = dst_ptr_4 + dst_str[5] * d5;
                            for (size_t d6 = 0; d6 < dims[6]; d6++) {
                                auto src_ptr_6 = src_ptr_5 + src_str[6] * d6;
                                auto dst_ptr_6 = dst_ptr_5 + dst_str[6] * d6;
                                *dst_ptr_6 = *src_ptr_6;
                            }
                        }
                    }
                }
            }
        }
    }
}

void fill_data_with_broadcast(InferenceEngine::Blob::Ptr& blob, InferenceEngine::Blob::Ptr& values) {
    using InferenceEngine::SizeVector;
    constexpr size_t MAX_N_DIMS = 7;  // Suppose it's enough

    IE_ASSERT(blob->getTensorDesc().getPrecision() == values->getTensorDesc().getPrecision());

    auto values_dims = values->getTensorDesc().getDims();
    auto blob_dims = blob->getTensorDesc().getDims();
    auto n_dims = blob_dims.size();
    IE_ASSERT(values_dims.size() <= n_dims);
    IE_ASSERT(n_dims <= MAX_N_DIMS);

    ov::Shape src_dims(MAX_N_DIMS, 1);
    std::copy(values_dims.rbegin(), values_dims.rend(), src_dims.rbegin());

    ov::Shape dst_dims(MAX_N_DIMS, 1);
    std::copy(blob_dims.rbegin(), blob_dims.rend(), dst_dims.rbegin());

    bool compatible = true;
    for (int i = 0; i < MAX_N_DIMS; i++) {
        if (src_dims[i] != dst_dims[i] && src_dims[i] != 1)
            compatible = false;
    }

    IE_ASSERT(compatible);

    auto fill_strides_like_plain = [](ov::Shape dims) {
        ov::Shape str(dims.size());
        if (str.empty())
            return str;
        else
            str.back() = 1;

        // stride[i] = stride[i+1]*d[i+1]
        std::transform(dims.rbegin(), dims.rend() - 1, str.rbegin(), str.rbegin() + 1, [](size_t d, size_t s) {
            return d * s;
        });

        // zeroing broadcast dimension equal 1
        std::transform(str.begin(), str.end(), dims.begin(), str.begin(), [](size_t s, size_t d) {
            return d == 1 ? 0 : s;
        });

        return str;
    };

    SizeVector src_strides = fill_strides_like_plain(src_dims);
    SizeVector dst_strides = fill_strides_like_plain(dst_dims);

    auto get_data = [](InferenceEngine::Blob::Ptr& blob) {
        auto mem_blob = dynamic_cast<InferenceEngine::MemoryBlob*>(blob.get());
        auto mem = mem_blob->rwmap();
        return mem.as<float*>();
    };

    auto dst_ptr = get_data(blob);
    auto src_ptr = get_data(values);

    switch (blob->getTensorDesc().getPrecision()) {
    case InferenceEngine::Precision::U64:
    case InferenceEngine::Precision::I64:
        copy_7D<uint64_t>(src_ptr, src_strides, dst_ptr, dst_strides, dst_dims);
        break;
    case InferenceEngine::Precision::FP32:
    case InferenceEngine::Precision::I32:
        copy_7D<uint32_t>(src_ptr, src_strides, dst_ptr, dst_strides, dst_dims);
        break;
    case InferenceEngine::Precision::I16:
    case InferenceEngine::Precision::U16:
    case InferenceEngine::Precision::FP16:
    case InferenceEngine::Precision::BF16:
        copy_7D<uint16_t>(src_ptr, src_strides, dst_ptr, dst_strides, dst_dims);
        break;
    case InferenceEngine::Precision::U8:
    case InferenceEngine::Precision::I8:
        copy_7D<uint8_t>(src_ptr, src_strides, dst_ptr, dst_strides, dst_dims);
        break;
    default:
        IE_THROW() << "Unsupported precision by fill_data_with_broadcast function";
    }
}

template <InferenceEngine::Precision::ePrecision SRC_E, InferenceEngine::Precision::ePrecision DST_E>
void copy_with_convert(InferenceEngine::Blob::Ptr& src_blob, InferenceEngine::Blob::Ptr& dst_blob) {
    using SRC_TYPE = typename InferenceEngine::PrecisionTrait<SRC_E>::value_type;
    using DST_TYPE = typename InferenceEngine::PrecisionTrait<DST_E>::value_type;

    auto src_lock_m = src_blob->as<InferenceEngine::MemoryBlob>()->rwmap();
    auto src_ptr = src_lock_m.as<SRC_TYPE*>();
    auto src_size = src_blob->size();

    auto dst_lock_m = dst_blob->as<InferenceEngine::MemoryBlob>()->rwmap();
    auto dst_ptr = dst_lock_m.as<DST_TYPE*>();

    std::copy(src_ptr, src_ptr + src_size, dst_ptr);
}

InferenceEngine::Blob::Ptr make_with_precision_convert(InferenceEngine::Blob::Ptr& blob,
                                                       InferenceEngine::Precision prc) {
    IE_ASSERT(isDenseBlob(blob));
    auto td = blob->getTensorDesc();
    td.setPrecision(prc);

    auto new_blob = make_blob_with_precision(td);
    new_blob->allocate();

#define CASE(_PRC)                                                                                             \
    case InferenceEngine::Precision::_PRC:                                                                     \
        copy_with_convert<InferenceEngine::Precision::FP32, InferenceEngine::Precision::_PRC>(blob, new_blob); \
        break
    switch (prc) {
        CASE(FP32);
        CASE(I64);
        CASE(U64);
        CASE(I32);
        CASE(U32);
        CASE(I16);
        CASE(U16);
        CASE(I8);
        CASE(U8);
    default:
        IE_THROW() << "Unsupported precision case";
    }
#undef CASE

    return new_blob;
}

void fill_data_with_broadcast(InferenceEngine::Blob::Ptr& blob, size_t axis, std::vector<float> values) {
    InferenceEngine::SizeVector value_dims(blob->getTensorDesc().getDims().size() - axis, 1);
    value_dims.front() = values.size();
    auto prc = blob->getTensorDesc().getPrecision();
    auto layout = InferenceEngine::TensorDesc::getLayoutByDims(value_dims);
    InferenceEngine::TensorDesc value_tdesc(prc, value_dims, layout);

    InferenceEngine::Blob::Ptr values_blob;
    if (prc == InferenceEngine::Precision::FP32) {
        values_blob = make_blob_with_precision(value_tdesc, values.data());
    } else {
        values_blob = make_blob_with_precision(value_tdesc, values.data());
        values_blob = make_with_precision_convert(values_blob, prc);
    }

    fill_data_with_broadcast(blob, values_blob);
}

InferenceEngine::Blob::Ptr make_reshape_view(const InferenceEngine::Blob::Ptr& blob,
                                             InferenceEngine::SizeVector new_shape) {
    using InferenceEngine::TensorDesc;
    auto new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    IE_ASSERT(new_size == blob->size());

    auto orig_mem_blob = dynamic_cast<InferenceEngine::MemoryBlob*>(blob.get());
    auto orig_mem = orig_mem_blob->rwmap();
    auto orig_ptr = orig_mem.as<float*>();

    auto new_tdesc =
        TensorDesc(blob->getTensorDesc().getPrecision(), new_shape, TensorDesc::getLayoutByDims(new_shape));
    auto new_blob = make_blob_with_precision(new_tdesc, orig_ptr);
    return new_blob;
}

size_t byte_size(const InferenceEngine::TensorDesc& tdesc) {
    auto prc = tdesc.getPrecision();
    auto dims = tdesc.getDims();
    return prc.size() * std::accumulate(std::begin(dims), std::end(dims), (size_t)1, std::multiplies<size_t>());
}
OPENVINO_SUPPRESS_DEPRECATED_END

template <ov::element::Type_t type>
inline void fill_data_roi_impl(ov::runtime::Tensor& tensor,
                               const uint32_t range,
                               const int height,
                               const int width,
                               const float omega,
                               const bool is_roi_max_mode,
                               const int seed = 1) {
    using T = typename ov::fundamental_type_for<type>;
    auto* data = static_cast<T*>(tensor.data());
    std::default_random_engine random(seed);
    std::uniform_int_distribution<int32_t> distribution(0, range);

    const int max_y = (is_roi_max_mode) ? (height - 1) : 1;
    const int max_x = (is_roi_max_mode) ? (width - 1) : 1;

    float center_h = (max_y) / 2.0f;
    float center_w = (max_x) / 2.0f;

    for (size_t i = 0; i < tensor.get_size(); i += 5) {
        data[i] = static_cast<T>(distribution(random));
        const float x0 = (center_w + width * 0.3f * sin(static_cast<float>(i + 1) * omega));
        const float x1 = (center_w + width * 0.3f * sin(static_cast<float>(i + 3) * omega));
        data[i + 1] = static_cast<T>(is_roi_max_mode ? std::floor(x0) : x0);
        data[i + 3] = static_cast<T>(is_roi_max_mode ? std::floor(x1) : x1);
        if (data[i + 3] < data[i + 1]) {
            std::swap(data[i + 1], data[i + 3]);
        }
        if (data[i + 1] < 0)
            data[i + 1] = 0;
        if (data[i + 3] > max_x)
            data[i + 3] = static_cast<T>(max_x);

        const float y0 = (center_h + height * 0.3f * sin(static_cast<float>(i + 2) * omega));
        const float y1 = (center_h + height * 0.3f * sin(static_cast<float>(i + 4) * omega));
        data[i + 2] = static_cast<T>(is_roi_max_mode ? std::floor(y0) : y0);
        data[i + 4] = static_cast<T>(is_roi_max_mode ? std::floor(y1) : y1);
        if (data[i + 4] < data[i + 2]) {
            std::swap(data[i + 2], data[i + 4]);
        }
        if (data[i + 2] < 0)
            data[i + 2] = 0;
        if (data[i + 4] > max_y)
            data[i + 4] = static_cast<T>(max_y);
    }
}

void fill_data_roi(ov::runtime::Tensor& tensor,
                   const uint32_t range,
                   const int height,
                   const int width,
                   const float omega,
                   const bool is_roi_max_mode,
                   const int seed) {
#define CASE(X)                                                                            \
    case X:                                                                                \
        fill_data_roi_impl<X>(tensor, range, height, width, omega, is_roi_max_mode, seed); \
        break;

    auto element_type = tensor.get_element_type();
    switch (element_type) {
        CASE(ov::element::f64)
        CASE(ov::element::f32)
        CASE(ov::element::f16)
        CASE(ov::element::bf16)
        CASE(ov::element::u1)
        CASE(ov::element::u4)
        CASE(ov::element::u8)
        CASE(ov::element::u32)
        CASE(ov::element::u16)
        CASE(ov::element::u64)
        CASE(ov::element::i4)
        CASE(ov::element::i8)
        CASE(ov::element::i16)
        CASE(ov::element::i32)
        CASE(ov::element::i64)
    default:
        OPENVINO_THROW("Wrong precision specified: ", element_type);
    }
#undef CASE
}

void fill_data_with_broadcast(ov::Tensor& tensor, ov::Tensor& values) {
    constexpr size_t MAX_N_DIMS = 7;  // Suppose it's enough

    OPENVINO_ASSERT(tensor.get_element_type() == values.get_element_type());

    auto values_dims = values.get_shape();
    auto tensor_dims = tensor.get_shape();
    auto n_dims = tensor_dims.size();
    OPENVINO_ASSERT(values_dims.size() <= n_dims);
    OPENVINO_ASSERT(n_dims <= MAX_N_DIMS);

    ov::Shape src_dims(MAX_N_DIMS, 1);
    std::copy(values_dims.rbegin(), values_dims.rend(), src_dims.rbegin());

    ov::Shape dst_dims(MAX_N_DIMS, 1);
    std::copy(tensor_dims.rbegin(), tensor_dims.rend(), dst_dims.rbegin());

    bool compatible = true;
    for (int i = 0; i < MAX_N_DIMS; i++) {
        if (src_dims[i] != dst_dims[i] && src_dims[i] != 1)
            compatible = false;
    }

    OPENVINO_ASSERT(compatible);

    auto fill_strides_like_plain = [](ov::Shape dims) {
        ov::Shape str(dims.size());
        if (str.empty())
            return str;
        else
            str.back() = 1;

        // stride[i] = stride[i+1]*d[i+1]
        std::transform(dims.rbegin(), dims.rend() - 1, str.rbegin(), str.rbegin() + 1, [](size_t d, size_t s) {
            return d * s;
        });

        // zeroing broadcast dimension equal 1
        std::transform(str.begin(), str.end(), dims.begin(), str.begin(), [](size_t s, size_t d) {
            return d == 1 ? 0 : s;
        });

        return str;
    };

    ov::Shape src_strides = fill_strides_like_plain(src_dims);
    ov::Shape dst_strides = fill_strides_like_plain(dst_dims);

    auto dst_ptr = tensor.data();
    auto src_ptr = values.data();

    using namespace ov::element;
    switch (tensor.get_element_type()) {
    case u64:
    case i64:
        copy_7D<uint64_t>(src_ptr, src_strides, dst_ptr, dst_strides, dst_dims);
        break;
    case f32:
    case i32:
        copy_7D<uint32_t>(src_ptr, src_strides, dst_ptr, dst_strides, dst_dims);
        break;
    case i16:
    case u16:
    case f16:
    case bf16:
        copy_7D<uint16_t>(src_ptr, src_strides, dst_ptr, dst_strides, dst_dims);
        break;
    case u8:
    case i8:
        copy_7D<uint8_t>(src_ptr, src_strides, dst_ptr, dst_strides, dst_dims);
        break;
    default:
        OPENVINO_THROW("Unsupported precision by fill_data_with_broadcast function");
    }
}

template <ov::element::Type_t SRC_E, ov::element::Type_t DST_E, typename std::enable_if<SRC_E != DST_E, int>::type = 0>
void copy_tensor_with_convert(const ov::Tensor& src_tensor, ov::Tensor& dst_tensor) {
    using SRC_TYPE = typename ov::fundamental_type_for<SRC_E>;
    using DST_TYPE = typename ov::fundamental_type_for<DST_E>;

    OPENVINO_ASSERT(src_tensor.get_size() == dst_tensor.get_size());

    auto src_ptr = src_tensor.data<SRC_TYPE>();
    auto src_size = src_tensor.get_size();

    auto dst_ptr = dst_tensor.data<DST_TYPE>();

    auto converter = [](SRC_TYPE value) {
        return static_cast<DST_TYPE>(value);
    };

    std::transform(src_ptr, src_ptr + src_size, dst_ptr, converter);
}

template <ov::element::Type_t SRC_E, ov::element::Type_t DST_E, typename std::enable_if<SRC_E == DST_E, int>::type = 0>
void copy_tensor_with_convert(const ov::Tensor& src_tensor, ov::Tensor& dst_tensor) {
    src_tensor.copy_to(dst_tensor);
}

ov::Tensor make_tensor_with_precision_convert(const ov::Tensor& tensor, ov::element::Type prc) {
    ov::Tensor new_tensor(prc, tensor.get_shape());
    auto src_prc = tensor.get_element_type();

#define CASE0(SRC_PRC, DST_PRC)                                                                   \
    case ov::element::DST_PRC:                                                                    \
        copy_tensor_with_convert<ov::element::SRC_PRC, ov::element::DST_PRC>(tensor, new_tensor); \
        break;

#define CASE(SRC_PRC)                                                            \
    case ov::element::SRC_PRC:                                                   \
        switch (prc) {                                                           \
            CASE0(SRC_PRC, bf16)                                                 \
            CASE0(SRC_PRC, f16)                                                  \
            CASE0(SRC_PRC, f32)                                                  \
            CASE0(SRC_PRC, f64)                                                  \
            CASE0(SRC_PRC, i8)                                                   \
            CASE0(SRC_PRC, i16)                                                  \
            CASE0(SRC_PRC, i32)                                                  \
            CASE0(SRC_PRC, i64)                                                  \
            CASE0(SRC_PRC, u8)                                                   \
            CASE0(SRC_PRC, u16)                                                  \
            CASE0(SRC_PRC, u32)                                                  \
            CASE0(SRC_PRC, u64)                                                  \
        default:                                                                 \
            OPENVINO_THROW("Unsupported precision case: ", prc.c_type_string()); \
        }                                                                        \
        break;

    switch (src_prc) {
        CASE(f64);
        CASE(f32);
        CASE(f16);
        CASE(bf16);
        CASE(i64);
        CASE(u64);
        CASE(i32);
        CASE(u32);
        CASE(i16);
        CASE(u16);
        CASE(i8);
        CASE(u8);
    default:
        OPENVINO_THROW("Unsupported precision case: ", src_prc.c_type_string());
    }
#undef CASE0
#undef CASE

    return new_tensor;
}

void fill_data_with_broadcast(ov::Tensor& tensor, size_t axis, std::vector<float> values) {
    ov::Shape value_dims(tensor.get_shape().size() - axis, 1);
    value_dims.front() = values.size();
    auto prc = tensor.get_element_type();

    ov::Tensor values_tensor;
    values_tensor = ov::Tensor(ov::element::f32, value_dims, values.data());

    if (prc != ov::element::f32) {
        values_tensor = make_tensor_with_precision_convert(values_tensor, prc);
    }

    fill_data_with_broadcast(tensor, values_tensor);
}

template <ov::element::Type_t DT>
void fill_tensor_random(ov::Tensor& tensor,
                        const uint32_t range,
                        const int32_t start_from,
                        const int32_t k,
                        const int seed) {
    using T = typename ov::element_type_traits<DT>::value_type;
    auto* rawBlobDataPtr = static_cast<T*>(tensor.data());
    if (DT == ov::element::u4 || DT == ov::element::i4 || DT == ov::element::u1) {
        fill_data_random(rawBlobDataPtr, tensor.get_byte_size(), range, start_from, k, seed);
    } else {
        fill_data_random(rawBlobDataPtr, tensor.get_size(), range, start_from, k, seed);
    }
}

template <ov::element::Type_t DT>
void fill_tensor_random_float(ov::Tensor& tensor,
                              const double range,
                              const double start_from,
                              const int32_t k,
                              const int seed) {
    using T = typename ov::element_type_traits<DT>::value_type;
    std::default_random_engine random(seed);
    // 1/k is the resolution of the floating point numbers
    std::uniform_real_distribution<double> distribution(k * start_from, k * (start_from + range));

    auto* rawBlobDataPtr = static_cast<T*>(tensor.data());
    for (size_t i = 0; i < tensor.get_size(); i++) {
        auto value = static_cast<float>(distribution(random));
        value /= static_cast<float>(k);
        if (DT == ov::element::Type_t::f16) {
            rawBlobDataPtr[i] = static_cast<T>(ov::float16(value).to_bits());
        } else if (DT == ov::element::Type_t::bf16) {
            rawBlobDataPtr[i] = static_cast<T>(ov::bfloat16(value).to_bits());
        } else {
            rawBlobDataPtr[i] = static_cast<T>(value);
        }
    }
}

void fill_tensor_random(ov::Tensor& tensor,
                        const double range,
                        const double start_from,
                        const int32_t k,
                        const int seed) {
    auto element_type = tensor.get_element_type();

#define CASE(X)                                                                                                 \
    case X:                                                                                                     \
        fill_tensor_random<X>(tensor, static_cast<uint32_t>(range), static_cast<int32_t>(start_from), k, seed); \
        break;
#define CASE_FLOAT(X)                                                    \
    case X:                                                              \
        fill_tensor_random_float<X>(tensor, range, start_from, k, seed); \
        break;

    switch (element_type) {
        CASE_FLOAT(ov::element::f64)
        CASE_FLOAT(ov::element::f32)
        CASE_FLOAT(ov::element::f16)
        CASE_FLOAT(ov::element::bf16)
        CASE(ov::element::u1)
        CASE(ov::element::u4)
        CASE(ov::element::u8)
        CASE(ov::element::u32)
        CASE(ov::element::u16)
        CASE(ov::element::u64)
        CASE(ov::element::i4)
        CASE(ov::element::i8)
        CASE(ov::element::i16)
        CASE(ov::element::i32)
        CASE(ov::element::i64)
        CASE(ov::element::boolean)
    default:
        OPENVINO_THROW("Wrong precision specified: ", element_type);
    }
#undef CASE
#undef CASE_FLOAT
}

}  // namespace utils
}  // namespace test
}  // namespace ov
