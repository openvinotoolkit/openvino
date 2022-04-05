// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include <debug.h>  // to allow putting vector into exception string stream

#include <ie_blob.h>
#include <blob_factory.hpp>

using namespace InferenceEngine::details;

namespace CommonTestUtils {

bool isDenseBlob(const InferenceEngine::Blob::Ptr& blob) {
    auto blk_desc = blob->getTensorDesc().getBlockingDesc();
    auto dims = blk_desc.getBlockDims();
    auto strs = blk_desc.getStrides();

    IE_ASSERT(dims.size() == strs.size()) << " isDenseBlob: inconsistent tensor descriptor";

    auto size = dims.size();
    if (size == 0) return true;
    if (size == 1) return strs[0] == 1;

    for (auto i = size - 1; i > 0; i--) {
        if (strs[i - 1] != strs[i - 1] * dims[i])
            return false;
    }

    return true;
}

template<typename T>
void copy_7D(void *src_raw_ptr, std::vector<size_t> &src_str, void *dst_raw_ptr, std::vector<size_t> &dst_str, std::vector<size_t> &dims) {
    auto src_ptr = static_cast<T*>(src_raw_ptr);
    auto dst_ptr = static_cast<T*>(dst_raw_ptr);

    for (size_t d0 = 0; d0 < dims[0]; d0++) { auto src_ptr_0 = src_ptr   + src_str[0]*d0; auto dst_ptr_0 = dst_ptr +   dst_str[0]*d0;
    for (size_t d1 = 0; d1 < dims[1]; d1++) { auto src_ptr_1 = src_ptr_0 + src_str[1]*d1; auto dst_ptr_1 = dst_ptr_0 + dst_str[1]*d1;
    for (size_t d2 = 0; d2 < dims[2]; d2++) { auto src_ptr_2 = src_ptr_1 + src_str[2]*d2; auto dst_ptr_2 = dst_ptr_1 + dst_str[2]*d2;
    for (size_t d3 = 0; d3 < dims[3]; d3++) { auto src_ptr_3 = src_ptr_2 + src_str[3]*d3; auto dst_ptr_3 = dst_ptr_2 + dst_str[3]*d3;
    for (size_t d4 = 0; d4 < dims[4]; d4++) { auto src_ptr_4 = src_ptr_3 + src_str[4]*d4; auto dst_ptr_4 = dst_ptr_3 + dst_str[4]*d4;
    for (size_t d5 = 0; d5 < dims[5]; d5++) { auto src_ptr_5 = src_ptr_4 + src_str[5]*d5; auto dst_ptr_5 = dst_ptr_4 + dst_str[5]*d5;
    for (size_t d6 = 0; d6 < dims[6]; d6++) { auto src_ptr_6 = src_ptr_5 + src_str[6]*d6; auto dst_ptr_6 = dst_ptr_5 + dst_str[6]*d6;
        *dst_ptr_6 = *src_ptr_6;
    }}}}}}}
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

    SizeVector src_dims(MAX_N_DIMS, 1);
    std::copy(values_dims.rbegin(), values_dims.rend(), src_dims.rbegin());

    SizeVector dst_dims(MAX_N_DIMS, 1);
    std::copy(blob_dims.rbegin(), blob_dims.rend(), dst_dims.rbegin());

    bool compatible = true;
    for (int i = 0; i < MAX_N_DIMS; i++) {
        if (src_dims[i] != dst_dims[i] && src_dims[i] != 1)
            compatible = false;
    }

    IE_ASSERT(compatible);

    auto fill_strides_like_plain = [] (SizeVector dims) {
        SizeVector str(dims.size());
        if (str.empty())
            return str;
        else
            str.back() = 1;

        // stride[i] = stride[i+1]*d[i+1]
        std::transform(dims.rbegin(), dims.rend() - 1, str.rbegin(), str.rbegin() + 1,
                       [] (size_t d, size_t s) { return d * s; });

        // zeroing broadcast dimension equal 1
        std::transform(str.begin(), str.end(), dims.begin(), str.begin(),
                       [] (size_t s, size_t d) { return d == 1 ? 0 : s; });

        return str;
    };

    SizeVector src_strides = fill_strides_like_plain(src_dims);
    SizeVector dst_strides = fill_strides_like_plain(dst_dims);

    auto get_data = [] (InferenceEngine::Blob::Ptr &blob) {
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

template<InferenceEngine::Precision::ePrecision SRC_E, InferenceEngine::Precision::ePrecision DST_E>
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

InferenceEngine::Blob::Ptr make_with_precision_convert(InferenceEngine::Blob::Ptr& blob, InferenceEngine::Precision prc) {
    IE_ASSERT(isDenseBlob(blob));
    auto td = blob->getTensorDesc();
    td.setPrecision(prc);

    auto new_blob = make_blob_with_precision(td);
    new_blob->allocate();

#define CASE(_PRC) case InferenceEngine::Precision::_PRC: \
        copy_with_convert<InferenceEngine::Precision::FP32, InferenceEngine::Precision::_PRC> (blob, new_blob); break
    switch (prc) {
        CASE(FP32); CASE(I64); CASE(U64); CASE(I32); CASE(U32); CASE(I16); CASE(U16); CASE(I8); CASE(U8);
        default: IE_THROW() << "Unsupported precision case";
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

InferenceEngine::Blob::Ptr make_reshape_view(const InferenceEngine::Blob::Ptr &blob, InferenceEngine::SizeVector new_shape) {
    using InferenceEngine::TensorDesc;
    auto new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    IE_ASSERT(new_size == blob->size());

    auto orig_mem_blob = dynamic_cast<InferenceEngine::MemoryBlob*>(blob.get());
    auto orig_mem = orig_mem_blob->rwmap();
    auto orig_ptr = orig_mem.as<float*>();

    auto new_tdesc = TensorDesc(blob->getTensorDesc().getPrecision(), new_shape, TensorDesc::getLayoutByDims(new_shape));
    auto new_blob = make_blob_with_precision(new_tdesc, orig_ptr);
    return new_blob;
}

size_t byte_size(const InferenceEngine::TensorDesc &tdesc) {
    auto prc = tdesc.getPrecision();
    auto dims = tdesc.getDims();
    return prc.size() * std::accumulate(std::begin(dims), std::end(dims), (size_t)1, std::multiplies<size_t>());
}

/**
 * repeated filling tensor with data.
 *
 * @tparam PRC
 * @param data
 * @param size
 * @param values
 */
template<InferenceEngine::Precision::ePrecision PRC = InferenceEngine::Precision::FP32>
static void fill_data_const(void *data, size_t size, const std::vector<float> &values) {
    auto t_data = static_cast<typename InferenceEngine::PrecisionTrait<PRC>::value_type *>(data);
    auto val_size = values.size();
    for (size_t i = 0, j = 0; i < size; i++) {
        t_data[i] = values[j++];
        if (j == val_size) j = 0;
    }
}

void fill_data_const(InferenceEngine::Blob::Ptr& blob, const std::vector<float> &val) {
    auto prc = blob->getTensorDesc().getPrecision();
    auto raw_data_ptr = blob->buffer().as<void*>();
    auto raw_data_size = blob->size();

    using InferenceEngine::Precision;
    switch (prc) {
        case Precision::FP32:
            fill_data_const<Precision::FP32>(raw_data_ptr, raw_data_size, val);
            break;
        case Precision::I32:
            fill_data_const<Precision::I32>(raw_data_ptr, raw_data_size, val);
            break;
        case Precision::U8:
            fill_data_const<Precision::U8>(raw_data_ptr, raw_data_size, val);
            break;
        case Precision::I8:
            fill_data_const<Precision::I8>(raw_data_ptr, raw_data_size, val);
            break;
        case Precision::U16:
            fill_data_const<Precision::U16>(raw_data_ptr, raw_data_size, val);
            break;
        case Precision::I16:
            fill_data_const<Precision::I16>(raw_data_ptr, raw_data_size, val);
            break;
        default:
            IE_THROW() << "Unsupported precision by fill_data_const() function";
    }
}

void fill_data_const(InferenceEngine::Blob::Ptr& blob, float val) {
    fill_data_const(blob, std::vector<float> {val});
}
}  // namespace CommonTestUtils
