// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

inline void clipping(int *idx, const int min, const int max) {
    (*idx) = ((*idx) > min) ? (*idx) : min;
    (*idx) = ((*idx) < max) ? (*idx) : (max - 1);
    return;
}

class StridedSliceImpl: public ExtLayerBase {
public:
    explicit StridedSliceImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() > 4 || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            src_dims = layer->insData[STRIDEDSLICE_DATA].lock()->getTensorDesc().getDims();

            bounds_size = 0;
            begin_dims = {};
            if (layer->insData.size() > 1) {
                begin_dims = layer->insData[STRIDEDSLICE_BEGIN].lock()->getTensorDesc().getDims();
                if (begin_dims.size() > 1)
                    THROW_IE_EXCEPTION << layer->name << " Begin vector should be 1 dimension";
                bounds_size = begin_dims[0];
            }

            if (layer->insData.size() > 2) {
                end_dims = layer->insData[STRIDEDSLICE_END].lock()->getTensorDesc().getDims();
                if (end_dims.size() > 1)
                    THROW_IE_EXCEPTION << layer->name << " End vector should be 1 dimension";
                if (begin_dims[0] != end_dims[0])
                    THROW_IE_EXCEPTION << layer->name << " Begin vector size should be equal end vectror size";
            }

            if (layer->insData.size() > 3) {
                stride_dims = layer->insData[STRIDEDSLICE_STRIDE].lock()->getTensorDesc().getDims();
                if (stride_dims.size() > 1)
                    THROW_IE_EXCEPTION << layer->name << " End vector should be 1 dimension";
                if (begin_dims[0] != stride_dims[0])
                    THROW_IE_EXCEPTION << layer->name << " Stride vector size should be equal begin vectror size";
            }
            dst_dims = layer->outData[0]->getTensorDesc().getDims();

            std::string::size_type i;
            std::string begin_mask_str = layer->GetParamAsString("begin_mask", "");
            for (i = 0; i < begin_mask_str.size(); ++i) {
                if (begin_mask_str[i] == '1') begin_mask.push_back(1);
                else if (begin_mask_str[i] == '0') begin_mask.push_back(0);
            }
            for (; i < src_dims.size(); ++i) begin_mask.push_back(1);

            std::string end_mask_str = layer->GetParamAsString("end_mask", "");
            for (i = 0; i < end_mask_str.size(); ++i) {
                if (end_mask_str[i] == '1') end_mask.push_back(1);
                else if (end_mask_str[i] == '0') end_mask.push_back(0);
            }
            for (; i < src_dims.size(); ++i) end_mask.push_back(1);

            std::string ellipsis_mask_str = layer->GetParamAsString("ellipsis_mask", "");
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
                THROW_IE_EXCEPTION << layer->name << " 'Ellipsis_mask' must be a power of two (only one ellipsis)!";
            for (; i < src_dims.size(); ++i) ellipsis_mask.push_back(0);

            std::string new_axis_mask_str = layer->GetParamAsString("new_axis_mask", "");
            for (i = 0; i < new_axis_mask_str.size(); ++i) {
                if (new_axis_mask_str[i] == '1') new_axis_mask.push_back(1);
                else if (new_axis_mask_str[i] == '0') new_axis_mask.push_back(0);
            }
            for (; i < src_dims.size(); ++i) new_axis_mask.push_back(0);

            std::string shrink_axis_mask_str = layer->GetParamAsString("shrink_axis_mask", "");
            for (i = 0; i < shrink_axis_mask_str.size(); ++i) {
                if (shrink_axis_mask_str[i] == '1') shrink_axis_mask.push_back(1);
                else if (shrink_axis_mask_str[i] == '0') shrink_axis_mask.push_back(0);
            }
            for (; i < src_dims.size(); ++i) shrink_axis_mask.push_back(0);


            int new_axis = 0;
            for (auto& na : new_axis_mask)
                new_axis += na;

            shrink_axis = 0;
            for (auto& sa : shrink_axis_mask)
                shrink_axis += sa;
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
            if (bounds_size > 0 && (max_dims - bounds_size) > ellipsis_pos1)
                ellipsis_pos2 = max_dims - bounds_size;

            begin_dms.assign(max_dims, 0);
            end_dms.assign(max_dims, -1);
            stride_dms.assign(max_dims, 1);

            srcStrides = layer->insData[STRIDEDSLICE_DATA].lock()->getTensorDesc().getBlockingDesc().getStrides();
            dstStrides = layer->outData[0]->getTensorDesc().getBlockingDesc().getStrides();
            Precision dataPrecision = layer->insData[STRIDEDSLICE_DATA].lock()->getTensorDesc().getPrecision();
            if (layer->insData.size() == 1) {
                addConfig(layer, { DataConfigurator(ConfLayout::PLN, dataPrecision) }, { DataConfigurator(ConfLayout::PLN, dataPrecision) });
            } else if (layer->insData.size() == 2) {
                addConfig(layer, { DataConfigurator(ConfLayout::PLN, dataPrecision), DataConfigurator(ConfLayout::PLN, Precision::I32) },
                    { DataConfigurator(ConfLayout::PLN, dataPrecision) });
            } else if (layer->insData.size() == 3) {
                addConfig(layer, { DataConfigurator(ConfLayout::PLN, dataPrecision), DataConfigurator(ConfLayout::PLN, Precision::I32),
                    DataConfigurator(ConfLayout::PLN, Precision::I32) }, { DataConfigurator(ConfLayout::PLN, dataPrecision) });
            } else {
                addConfig(layer, { DataConfigurator(ConfLayout::PLN, dataPrecision), DataConfigurator(ConfLayout::PLN, Precision::I32),
                    DataConfigurator(ConfLayout::PLN, Precision::I32), DataConfigurator(ConfLayout::PLN, Precision::I32) },
                    { DataConfigurator(ConfLayout::PLN, dataPrecision) });
            }
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        int *begin = nullptr, *end = nullptr, *stride = nullptr;
        if (begin_dims.size())
            begin = inputs[STRIDEDSLICE_BEGIN]->cbuffer().as<int *>() + inputs[STRIDEDSLICE_BEGIN]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        if (end_dims.size())
            end = inputs[STRIDEDSLICE_END]->cbuffer().as<int *>() + inputs[STRIDEDSLICE_END]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        if (stride_dims.size())
            stride = inputs[STRIDEDSLICE_STRIDE]->cbuffer().as<int *>() + inputs[STRIDEDSLICE_STRIDE]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        InferenceEngine::SizeVector src_dims = inputs[STRIDEDSLICE_DATA]->getTensorDesc().getDims();
        InferenceEngine::SizeVector srcStrides = inputs[STRIDEDSLICE_DATA]->getTensorDesc().getBlockingDesc().getStrides();
        InferenceEngine::SizeVector dst_dims = outputs[0]->getTensorDesc().getDims();
        InferenceEngine::SizeVector dstStrides = outputs[0]->getTensorDesc().getBlockingDesc().getStrides();

        size_t i, j, k, bj, ej, sj;
        InferenceEngine::SizeVector our_dims;
        InferenceEngine::SizeVector out_dims;
        for (i = 0, j = 0, k = 0, bj = 0, ej = 0, sj = 0; static_cast<int>(i) < max_dims; i++) {
            if (static_cast<int>(i) >= ellipsis_pos1 &&
                    static_cast<int>(i) < ellipsis_pos2) {
                if (new_axis_mask.size() > i && new_axis_mask[i] == 1)
                    end_dms[i] = 0;
                else
                    end_dms[i] = end_dms[i] >= 0 ? end_dms[i] : src_dims[j++] + end_dms[i];

                out_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) / static_cast<float>(abs(stride_dms[i])))));
                our_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) / static_cast<float>(abs(stride_dms[i])))));
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
                clipping(&begin_dms[i], 0, src_dims[j]);

                if (end_mask.size() > j && end_mask[j] == 0) {
                    end_dms[i] = stride_dms[i] > 0 ? -1 : 0;
                } else {
                    int end_dms_tmp = (end != nullptr && end_dims[0] > ej) ? (stride_dms[i] > 0 ? end[ej] - 1 : end[ej] + 1)
                                                                     : end_dms[i];
                    end_dms[i] = (end != nullptr && end_dims[0] > ej) ? end_dms_tmp : (stride_dms[i] > 0 ? -1 : 0);
                }
                ej++;
                end_dms[i] = end_dms[i] >= 0 ? end_dms[i] : src_dims[j] + end_dms[i];
                //  Clipping 'end'
                clipping(&end_dms[i], 0, src_dims[j]);

                if (new_axis_mask.size() > i && new_axis_mask[i] == 1)
                    end_dms[i] = 0;
                else
                    j++;

                if (shrink_axis_mask.size() > k && shrink_axis_mask[k] == 1)
                    end_dms[i] = begin_dms[i];
                else
                    out_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) /
                                                             static_cast<float>(abs(stride_dms[i])))));

                our_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) /
                                                         static_cast<float>(abs(stride_dms[i])))));
                k++;
            }
        }

        for (i = 0; i < (std::min)(out_dims.size(), dst_dims.size()); i++) {
            if (out_dims[i] != dst_dims[i])
                return PARAMETER_MISMATCH;
        }

        const size_t inputsPrecSize = inputs[STRIDEDSLICE_DATA]->getTensorDesc().getPrecision().size();
        if (static_cast<int>(src_dims.size()) == max_dims && shrink_axis == 0 &&
                stride_dms[stride_dms.size()-1] == 1 && stride_dms.size() > 1) {
            if (inputsPrecSize != outputs[0]->getTensorDesc().getPrecision().size()) {
                if (resp) {
                    std::string errorMsg = "StridedSlice layer doesn't support 'Data' input precision: "
                        + std::string(inputs[STRIDEDSLICE_DATA]->getTensorDesc().getPrecision().name());
                        errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
            strided_slice_vp(inputs[STRIDEDSLICE_DATA], outputs[0]);
        } else if (static_cast<int>(src_dims.size()) == max_dims && shrink_axis == 0) {
            switch (inputsPrecSize) {
                case 1: { strided_slice_p<uint8_t>(inputs[STRIDEDSLICE_DATA], outputs[0]); break; }
                case 2: { strided_slice_p<uint16_t>(inputs[STRIDEDSLICE_DATA], outputs[0]); break; }
                case 4: { strided_slice_p<uint32_t>(inputs[STRIDEDSLICE_DATA], outputs[0]); break; }
                case 8: { strided_slice_p<uint64_t>(inputs[STRIDEDSLICE_DATA], outputs[0]); break; }
                default: {
                    if (resp) {
                        std::string errorMsg = "StridedSlice layer doesn't support 'Data' input precision: "
                            + std::string(inputs[STRIDEDSLICE_DATA]->getTensorDesc().getPrecision().name());
                            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                    }
                    return GENERAL_ERROR;
                }
            }
        } else {
            switch (inputsPrecSize) {
                case 1: { strided_slice<uint8_t>(inputs[STRIDEDSLICE_DATA], outputs[0], our_dims); break; }
                case 2: { strided_slice<uint16_t>(inputs[STRIDEDSLICE_DATA], outputs[0], our_dims); break; }
                case 4: { strided_slice<uint32_t>(inputs[STRIDEDSLICE_DATA], outputs[0], our_dims); break; }
                case 8: { strided_slice<uint64_t>(inputs[STRIDEDSLICE_DATA], outputs[0], our_dims); break; }
                default: {
                    if (resp) {
                        std::string errorMsg = "StridedSlice layer doesn't support 'Data' input precision: "
                            + std::string(inputs[STRIDEDSLICE_DATA]->getTensorDesc().getPrecision().name());
                            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                    }
                    return GENERAL_ERROR;
                }
            }
        }

        return OK;
    }

private:
    const size_t STRIDEDSLICE_DATA = 0;
    const size_t STRIDEDSLICE_BEGIN = 1;
    const size_t STRIDEDSLICE_END = 2;
    const size_t STRIDEDSLICE_STRIDE = 3;

    template <typename T>
    void strided_slice(Blob::Ptr&, Blob::Ptr& dst_data, std::vector<size_t> &dims);
    void strided_slice_vp(Blob::Ptr&, Blob::Ptr& dst_data);
    template <typename T>
    void strided_slice_p(Blob::Ptr&, Blob::Ptr& dst_data);

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
    int bounds_size;
    int max_dims;
    int ellipsis_pos1, ellipsis_pos2;
};

template <typename T>
void StridedSliceImpl::strided_slice(Blob::Ptr& input, Blob::Ptr& output, std::vector<size_t> &dims) {
    auto* src_data = input->cbuffer().as<const T*>() + input->getTensorDesc().getBlockingDesc().getOffsetPadding();
    auto* dst_data = output->buffer().as<T*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();
    auto dst_size = output->byteSize();
    memset(dst_data, 0, dst_size);

    size_t work_amount_dst = dstStrides[0] * dst_dims[0];
    parallel_nt(0, [&](const int ithr, const int nthr) {
        int j;
        size_t i, start = 0, end = 0;
        SizeVector counters(max_dims, 0);
        splitter(work_amount_dst, nthr, ithr, start, end);
        for (j = max_dims - 1, i = start; j >= 0; j--) {
            counters[j] = i % dims[j];
            i /= dims[j];
        }
        for (size_t iwork = start; iwork < end; ++iwork) {
            int src_idx = 0;
            for (i = 0, j = 0; static_cast<int>(i) < max_dims; ++i) {
                if (!(new_axis_mask.size() > i && new_axis_mask[i] == 1))
                    src_idx += (begin_dms[i] + counters[i] * stride_dms[i]) * srcStrides[j++];
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
    });
}

void StridedSliceImpl::strided_slice_vp(Blob::Ptr& input, Blob::Ptr& output) {
    size_t dataSize = input->getTensorDesc().getPrecision().size();
    const uint8_t* src_data = input->cbuffer().as<const uint8_t*>() + input->getTensorDesc().getBlockingDesc().getOffsetPadding() * dataSize;
    uint8_t* dst_data = output->buffer().as<uint8_t*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding() * dataSize;
    auto dst_size = output->byteSize();
    memset(dst_data, 0, dst_size);

    //  Vectorized copy
    size_t dims_size_1 = dst_dims.size() - 1;
    size_t len = dst_dims[dims_size_1] * dataSize;
    size_t work_amount_dst = dstStrides[0] * dst_dims[0] / dst_dims[dims_size_1];

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dims_size_1, 0);
        splitter(work_amount_dst, nthr, ithr, start, end);
        size_t src_idx = begin_dms[dims_size_1];
        for (int j = dims_size_1 - 1, i = start; j >= 0; j--) {
            counters[j] = i % dst_dims[j];
            src_idx += (begin_dms[j] + counters[j] * stride_dms[j]) * srcStrides[j];
            i /= dst_dims[j];
        }

        for (size_t iwork = start, dst_idx = start * len, i = 1; iwork < end; ++iwork, dst_idx += len) {
            cpu_memcpy(&dst_data[dst_idx], &src_data[src_idx * dataSize], len);
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

template <typename T>
void StridedSliceImpl::strided_slice_p(Blob::Ptr& input, Blob::Ptr& output) {
    auto* src_data = input->cbuffer().as<const T*>() + input->getTensorDesc().getBlockingDesc().getOffsetPadding();
    auto* dst_data = output->buffer().as<T*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();
    auto dst_size = output->byteSize();
    memset(dst_data, 0, dst_size);

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

REG_FACTORY_FOR(StridedSliceImpl, StridedSlice);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
