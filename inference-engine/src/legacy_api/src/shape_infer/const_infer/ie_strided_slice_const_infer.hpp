// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <legacy/ie_layers.h>
#include <ie_memcpy.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_const_infer_impl.hpp"
#include "ie_parallel.hpp"
#include "ie_precision.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

class StridedSliceHelper {
public:
    StridedSliceHelper(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params) {
        LayerParams lp {};
        CNNLayer layer(lp);
        layer.params = params;

        if (inData.size() > 4)
            THROW_IE_EXCEPTION << "StridedSlice constant inference error: Incorrect number of input edges!";

        src_dims = inData[STRIDEDSLICE_DATA]->getTensorDesc().getDims();

        bounds_size = 0;
        if (inData.size() > 1) {
            begin_dims = inData[STRIDEDSLICE_BEGIN]->getTensorDesc().getDims();
            if (inData[STRIDEDSLICE_BEGIN]->getTensorDesc().getPrecision() != Precision::I32)
                THROW_IE_EXCEPTION << "StridedSlice constant inference error: Incorrect 'begin' input precision. Only "
                                      "I32 is supported! Current precision: "
                                   << inData[STRIDEDSLICE_BEGIN]->getTensorDesc().getPrecision();
            if (begin_dims.size() > 1)
                THROW_IE_EXCEPTION << "StridedSlice constant inference error: Begin vector should be 1 dimension, got: "
                                   << begin_dims.size() << " dimensions";
            bounds_size = begin_dims[0];
        }

        if (inData.size() > 2) {
            end_dims = inData[STRIDEDSLICE_END]->getTensorDesc().getDims();
            if (inData[STRIDEDSLICE_END]->getTensorDesc().getPrecision() != Precision::I32)
                THROW_IE_EXCEPTION << "StridedSlice constant inference error: Incorrect 'end' input precision. Only "
                                      "I32 is supported! Current precision: "
                                   << inData[STRIDEDSLICE_END]->getTensorDesc().getPrecision();
            if (end_dims.size() > 1)
                THROW_IE_EXCEPTION << "StridedSlice constant inference error: End vector should be 1 dimension, got: "
                                   << end_dims.size() << " dimensions";
            if (begin_dims[0] != end_dims[0])
                THROW_IE_EXCEPTION
                    << "StridedSlice constant inference error: Begin vector size should be equal end vector size";
        }

        if (inData.size() > 3) {
            stride_dims = inData[STRIDEDSLICE_STRIDE]->getTensorDesc().getDims();
            if (inData[STRIDEDSLICE_STRIDE]->getTensorDesc().getPrecision() != Precision::I32)
                THROW_IE_EXCEPTION << "StridedSlice constant inference error: Incorrect 'strides' input precision. "
                                      "Only I32 is supported! Current precision: "
                                   << inData[STRIDEDSLICE_STRIDE]->getTensorDesc().getPrecision();
            if (stride_dims.size() > 1)
                THROW_IE_EXCEPTION << "StridedSlice constant inference error: End vector should be 1 dimension, got: "
                                   << stride_dims.size() << " dimensions";
            if (begin_dims[0] != stride_dims[0])
                THROW_IE_EXCEPTION
                    << "StridedSlice constant inference error: Stride vector size should be equal begin vector size";
        }

        std::string::size_type i;
        std::string begin_mask_str = layer.GetParamAsString("begin_mask", "");
        for (i = 0; i < begin_mask_str.size(); ++i) {
            if (begin_mask_str[i] == '1')
                begin_mask.push_back(1);
            else if (begin_mask_str[i] == '0')
                begin_mask.push_back(0);
        }
        for (; i < src_dims.size(); ++i) begin_mask.push_back(1);

        std::string end_mask_str = layer.GetParamAsString("end_mask", "");
        for (i = 0; i < end_mask_str.size(); ++i) {
            if (end_mask_str[i] == '1')
                end_mask.push_back(1);
            else if (end_mask_str[i] == '0')
                end_mask.push_back(0);
        }
        for (; i < src_dims.size(); ++i) end_mask.push_back(1);

        std::string ellipsis_mask_str = layer.GetParamAsString("ellipsis_mask", "");
        size_t ellipsis_mask_counter = 0;
        for (i = 0; i < ellipsis_mask_str.size(); ++i) {
            if (ellipsis_mask_str[i] == '1') {
                ellipsis_mask_counter++;
                ellipsis_mask.push_back(1);
            } else if (ellipsis_mask_str[i] == '0') {
                ellipsis_mask.push_back(0);
            }
        }
        if (ellipsis_mask_counter > 1)
            THROW_IE_EXCEPTION << " 'Ellipsis_mask' must be a power of two (only one ellipsis)!";
        for (; i < src_dims.size(); ++i) ellipsis_mask.push_back(0);

        std::string new_axis_mask_str = layer.GetParamAsString("new_axis_mask", "");
        for (i = 0; i < new_axis_mask_str.size(); ++i) {
            if (new_axis_mask_str[i] == '1')
                new_axis_mask.push_back(1);
            else if (new_axis_mask_str[i] == '0')
                new_axis_mask.push_back(0);
        }
        for (; i < src_dims.size(); ++i) new_axis_mask.push_back(0);

        std::string shrink_axis_mask_str = layer.GetParamAsString("shrink_axis_mask", "");
        for (i = 0; i < shrink_axis_mask_str.size(); ++i) {
            if (shrink_axis_mask_str[i] == '1')
                shrink_axis_mask.push_back(1);
            else if (shrink_axis_mask_str[i] == '0')
                shrink_axis_mask.push_back(0);
        }
        for (; i < src_dims.size(); ++i) shrink_axis_mask.push_back(0);

        int new_axis = 0;
        for (auto& na : new_axis_mask) new_axis += na;

        shrink_axis = 0;
        for (auto& sa : shrink_axis_mask) shrink_axis += sa;
        max_dims = src_dims.size() + new_axis;

        //  ellipsis_mask must be a power of two (only one ellipsis), so to take a first position
        ellipsis_pos1 = ellipsis_pos2 = max_dims;
        for (i = 0; i < ellipsis_mask.size(); i++) {
            if (ellipsis_mask[i] > 0) {
                ellipsis_pos1 = i;
                break;
            }
        }
        bounds_size -= ellipsis_pos1;
        if (bounds_size > 0 && (max_dims - bounds_size) > ellipsis_pos1) ellipsis_pos2 = max_dims - bounds_size;

        begin_dms.assign(max_dims, 0);
        end_dms.assign(max_dims, -1);
        stride_dms.assign(max_dims, 1);

        srcStrides = inData[STRIDEDSLICE_DATA]->getTensorDesc().getBlockingDesc().getStrides();

        int *begin = nullptr, *end = nullptr, *stride = nullptr;
        if (begin_dims.size())
            begin = inData[STRIDEDSLICE_BEGIN]->cbuffer().as<int*>() +
                    inData[STRIDEDSLICE_BEGIN]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        if (end_dims.size())
            end = inData[STRIDEDSLICE_END]->cbuffer().as<int*>() +
                  inData[STRIDEDSLICE_END]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        if (stride_dims.size())
            stride = inData[STRIDEDSLICE_STRIDE]->cbuffer().as<int*>() +
                     inData[STRIDEDSLICE_STRIDE]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        int j, k, bj, ej, sj;
        for (i = 0, j = 0, k = 0, bj = 0, ej = 0, sj = 0; i < max_dims; i++) {
            if (i >= ellipsis_pos1 && i < ellipsis_pos2) {
                if (new_axis_mask.size() > i && new_axis_mask[i] == 1)
                    end_dms[i] = 0;
                else
                    end_dms[i] = end_dms[i] >= 0 ? end_dms[i] : src_dims[j++] + end_dms[i];

                out_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) /
                                                         static_cast<float>(abs(stride_dms[i])))));
                our_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) /
                                                         static_cast<float>(abs(stride_dms[i])))));
                k = ellipsis_pos1;
            } else {
                stride_dms[i] = (stride != nullptr && stride_dims[0] > sj && stride[sj] != 0) ? stride[sj++] : 1;

                if (begin_mask.size() > j && begin_mask[j] == 0)
                    begin_dms[i] = stride_dms[i] > 0 ? 0 : -1;
                else
                    begin_dms[i] = (begin != nullptr && begin_dims[0] > bj) ? begin[bj] : (stride_dms[i] > 0 ? 0 : -1);
                bj++;
                begin_dms[i] = begin_dms[i] >= 0 ? begin_dms[i] : src_dims[j] + begin_dms[i];
                //  Clipping 'begin'
                details::clipping(&begin_dms[i], 0, src_dims[j]);

                if (end_mask.size() > j && end_mask[j] == 0) {
                    end_dms[i] = stride_dms[i] > 0 ? -1 : 0;
                } else {
                    int end_dms_tmp = (end != nullptr && end_dims[0] > ej)
                                          ? (stride_dms[i] > 0 ? end[ej] - 1 : end[ej] + 1)
                                          : end_dms[i];
                    end_dms[i] = (end != nullptr && end_dims[0] > ej) ? end_dms_tmp : (stride_dms[i] > 0 ? -1 : 0);
                }
                ej++;
                end_dms[i] = end_dms[i] >= 0 ? end_dms[i] : src_dims[j] + end_dms[i];
                //  Clipping 'end'
                details::clipping(&end_dms[i], 0, src_dims[j]);

                if (new_axis_mask.size() > i && new_axis_mask[i] == 1)
                    end_dms[i] = 0;
                else
                    j++;

                if (shrink_axis_mask.size() > k && shrink_axis_mask[k] == 1) {
                    end_dms[i] = begin_dms[i];
                    if (max_dims == 1) {
                        out_dims.push_back(
                            static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) /
                                                  static_cast<float>(abs(stride_dms[i])))));
                    }
                } else {
                    out_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) /
                                                             static_cast<float>(abs(stride_dms[i])))));
                }

                our_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) /
                                                         static_cast<float>(abs(stride_dms[i])))));
                k++;
            }
        }
    }

    SizeVector getOutputShape() {
        return out_dims;
    }

    template <class src_t, class dst_t>
    void exec_strided_slice(const std::vector<Blob::CPtr>& inData, std::vector<Blob::Ptr>& outData) {
        const src_t* src_data = inData[STRIDEDSLICE_DATA]->cbuffer().as<const src_t*>() +
                                inData[STRIDEDSLICE_DATA]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        dst_t* dst_data =
            outData[0]->cbuffer().as<dst_t*>() + outData[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        if (src_dims.size() == max_dims && shrink_axis == 0 && stride_dms[stride_dms.size() - 1] == 1 &&
            stride_dms.size() > 1)
            strided_slice_vp(src_data, dst_data);
        else if (src_dims.size() == max_dims && shrink_axis == 0)
            strided_slice_p(src_data, dst_data);
        else
            strided_slice(src_data, dst_data, our_dims);
    }

    void infer(const std::vector<Blob::CPtr>& inData, std::vector<Blob::Ptr>& outData) {
        dst_dims = outData[0]->getTensorDesc().getDims();
        size_t range = out_dims.size() < dst_dims.size() ? out_dims.size() : dst_dims.size();
        for (int i = 0; i < range; i++) {
            if (out_dims[i] != dst_dims[i])
                THROW_IE_EXCEPTION << "StridedSlice constant inference error: parameter mismatch";
        }
        dstStrides = outData[0]->getTensorDesc().getBlockingDesc().getStrides();
        if (dst_dims.size() == 1 && dst_dims[0] == 1) dstStrides.push_back(1);
        if (outData.size() != 1)
            THROW_IE_EXCEPTION << "StridedSlice constant inference error: Incorrect number of output edges!";

        auto compare =
            getPrecisionMask(inData[0]->getTensorDesc().getPrecision(), outData[0]->getTensorDesc().getPrecision());
        switch (compare) {
        case getPrecisionMask(Precision::FP32, Precision::FP32):
            exec_strided_slice<PrecisionTrait<Precision::FP32>::value_type,
                               PrecisionTrait<Precision::FP32>::value_type>(inData, outData);
            break;
        case getPrecisionMask(Precision::I32, Precision::I32):
            exec_strided_slice<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::I32>::value_type>(
                inData, outData);
            break;
        case getPrecisionMask(Precision::I32, Precision::I64):
            exec_strided_slice<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::I64>::value_type>(
                inData, outData);
            break;
        case getPrecisionMask(Precision::I32, Precision::U64):
            exec_strided_slice<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::U64>::value_type>(
                inData, outData);
            break;
        default:
            THROW_IE_EXCEPTION << "StridedSlice constant inference error: Unsupported precision configuration:"
                               << " input precision: " << inData[0]->getTensorDesc().getPrecision()
                               << " output precision: " << outData[0]->getTensorDesc().getPrecision();
        }
    }

private:
    template <class src_t, class dst_t>
    void strided_slice(const src_t* src_data, dst_t* dst_data, std::vector<size_t>& dims) {
        size_t i;
        int j;
        size_t work_amount_dst = (dstStrides.empty() && dst_dims.empty()) ? 1 : dstStrides[0] * dst_dims[0];
        SizeVector counters(max_dims, 0);

        for (size_t iwork = 0; iwork < work_amount_dst; ++iwork) {
            int src_idx = 0;
            for (i = 0, j = 0; i < max_dims; ++i) {
                src_idx += (begin_dms[i] + counters[i] * stride_dms[i]) * srcStrides[j];
                if (!(new_axis_mask.size() > i && new_axis_mask[i] == 1)) j++;
            }

            dst_data[iwork] = src_data[src_idx];

            for (j = max_dims - 1; j >= 0; j--) {
                counters[j]++;
                if (counters[j] < dims[j])
                    break;
                else
                    counters[j] = 0;
            }
        }
    }

    template <class src_t, class dst_t>
    void strided_slice_vp(const src_t* src_data, dst_t* dst_data) {
        //  Vectorized copy
        size_t dims_size_1 = dst_dims.size() - 1;
        size_t dataLength = dst_dims[dims_size_1];
        size_t work_amount_dst = dstStrides[0] * dst_dims[0] / dst_dims[dims_size_1];

        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            SizeVector counters(dims_size_1, 0);
            splitter(work_amount_dst, nthr, ithr, start, end);
            int src_idx = begin_dms[dims_size_1];
            for (int j = dims_size_1 - 1, i = start; j >= 0; j--) {
                counters[j] = i % dst_dims[j];
                src_idx += (begin_dms[j] + counters[j] * stride_dms[j]) * srcStrides[j];
                i /= dst_dims[j];
            }

            for (size_t iwork = start, dst_idx = start * dataLength, i = 1; iwork < end;
                 ++iwork, dst_idx += dataLength) {
                memcpy(&dst_data[dst_idx], &src_data[src_idx], sizeof(float) * dataLength);
                for (int j = dims_size_1 - 1; j >= 0; j--) {
                    counters[j]++;
                    if (counters[j] < dst_dims[j]) {
                        src_idx += stride_dms[j] * srcStrides[j];
                        break;
                    } else {
                        counters[j] = i = 0;
                    }
                }
                if (!i) {
                    for (src_idx = begin_dms[dims_size_1]; i < dims_size_1; ++i)
                        src_idx += (begin_dms[i] + counters[i] * stride_dms[i]) * srcStrides[i];
                }
            }
        });
    }

    template <class src_t, class dst_t>
    void strided_slice_p(const src_t* src_data, dst_t* dst_data) {
        size_t dims_size = dst_dims.size();
        size_t work_amount_dst = dstStrides[0] * dst_dims[0];

        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            SizeVector counters(dims_size, 0);
            splitter(work_amount_dst, nthr, ithr, start, end);
            int src_idx = 0;
            for (int j = dims_size - 1, i = start; j >= 0; j--) {
                counters[j] = i % dst_dims[j];
                src_idx += (begin_dms[j] + counters[j] * stride_dms[j]) * srcStrides[j];
                i /= dst_dims[j];
            }

            for (size_t iwork = start, dst_idx = start, i = 1; iwork < end; ++iwork, dst_idx++) {
                dst_data[dst_idx] = src_data[src_idx];
                for (int j = dims_size - 1; j >= 0; j--) {
                    counters[j]++;
                    if (counters[j] < dst_dims[j]) {
                        src_idx += stride_dms[j] * srcStrides[j];
                        break;
                    } else {
                        counters[j] = i = 0;
                    }
                }
                if (!i) {
                    for (src_idx = 0; i < dims_size; ++i)
                        src_idx += (begin_dms[i] + counters[i] * stride_dms[i]) * srcStrides[i];
                }
            }
        });
    }

private:
    const size_t STRIDEDSLICE_DATA = 0;
    const size_t STRIDEDSLICE_BEGIN = 1;
    const size_t STRIDEDSLICE_END = 2;
    const size_t STRIDEDSLICE_STRIDE = 3;

    SizeVector begin_dims;
    SizeVector end_dims;
    SizeVector stride_dims;

    SizeVector begin_mask;
    SizeVector end_mask;
    SizeVector ellipsis_mask;
    SizeVector new_axis_mask;
    SizeVector shrink_axis_mask;
    int shrink_axis;

    SizeVector src_dims;
    SizeVector dst_dims;
    std::vector<int> begin_dms;
    std::vector<int> end_dms;
    std::vector<int> stride_dms;
    SizeVector srcStrides;
    SizeVector dstStrides;
    size_t bounds_size;
    size_t max_dims;
    size_t ellipsis_pos1, ellipsis_pos2;

    InferenceEngine::SizeVector out_dims;
    InferenceEngine::SizeVector our_dims;
};

/**
 *@brief Implementation of Const inference for Tile layer
 */
class StridedSliceConstInfer : public ConstInferImpl {
public:
    explicit StridedSliceConstInfer(const std::string& type): ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override {
        LayerParams lp {};
        StridedSliceLayer layer(lp);
        layer.params = params;
        layer.type = _type;
        _validator->parseParams(&layer);

        StridedSliceHelper helper(inData, params);
        helper.infer(inData, outData);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
