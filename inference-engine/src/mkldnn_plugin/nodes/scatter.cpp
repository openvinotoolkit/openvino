// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <limits>
#include "ie_parallel.hpp"
#include "common/simple_copy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ScatterImpl: public ExtLayerBase {
public:
    explicit ScatterImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 3 || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output tensors!";


            Precision inIdxPrecision = layer->insData[SCATTER_INDEXES].lock()->getTensorDesc().getPrecision();
            if (inIdxPrecision != Precision::FP32 && inIdxPrecision != Precision::I32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input 'Indexes' precision. Only FP32 or I32 are supported!";

            Precision inDataPrecision = layer->insData[SCATTER_DATA].lock()->getTensorDesc().getPrecision();
            if (inDataPrecision != layer->insData[SCATTER_UPDATES].lock()->getTensorDesc().getPrecision())
                THROW_IE_EXCEPTION << layer->name << " Precision should be equal for input tensors 'Data' and 'Updates'";

            //  Remove redundant dimensions
            const SizeVector& data_dims = layer->insData[SCATTER_DATA].lock()->getTensorDesc().getDims();
            if (data_dims.size() == 0 ||
                (data_dims.size() == 1 && data_dims[0] == 1) ||
                layer->insData[SCATTER_DATA].lock()->getTensorDesc().getLayout() == Layout::SCALAR)
                    THROW_IE_EXCEPTION << layer->name << " 'Data' tensor rank should be >= 1";

            axis = layer->GetParamAsInt("axis", 0);

            IE_ASSERT(-static_cast<int>(data_dims.size()) <= axis && axis < static_cast<int>(data_dims.size()))
                << layer->name << " Incorrect input parameters dimensions and axis number!";

            if (axis < 0)
                axis += data_dims.size();

            SizeVector dst_dims = layer->outData[0]->getTensorDesc().getDims();
            if (data_dims != dst_dims)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output dimensions!";

            SizeVector idx_dims = layer->insData[SCATTER_INDEXES].lock()->getTensorDesc().getDims();
            if (idx_dims.size() == 0 ||
                (idx_dims.size() == 1 && idx_dims[0] == 1) ||
                layer->insData[SCATTER_INDEXES].lock()->getTensorDesc().getLayout() == Layout::SCALAR)
                THROW_IE_EXCEPTION << layer->name << " 'Indexes' tensor rank should be >= 1";

            SizeVector upd_dims = layer->insData[SCATTER_UPDATES].lock()->getTensorDesc().getDims();
            if (layer->insData[SCATTER_UPDATES].lock()->getTensorDesc().getLayout() == Layout::SCALAR)
                THROW_IE_EXCEPTION << layer->name << " 'Indexes' tensor rank should be >= 1";

            if (idx_dims != upd_dims)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of 'indexes' and 'updates' tensors dimension";

            for (size_t i = 0; i < idx_dims.size(); i++) {
                if (i == static_cast<size_t>(axis)) continue;
                if (idx_dims[i] > data_dims[i])
                    THROW_IE_EXCEPTION << layer->name << " Incorrect number of data and indexes dimensions!";
            }

            LayerConfig config;
            DataConfig dataConfig, indexesConfig, updatesConfig;
            Precision dataPrecision = layer->outData[0]->getTensorDesc().getPrecision();
            dataConfig.desc = TensorDesc(dataPrecision, data_dims,
                                         layer->insData[SCATTER_DATA].lock()->getTensorDesc().getLayout());
            dataConfig.constant = false;
            dataConfig.inPlace = 0;
            config.inConfs.push_back(dataConfig);
            indexesConfig.desc = TensorDesc(inIdxPrecision, idx_dims,
                                            layer->insData[SCATTER_INDEXES].lock()->getTensorDesc().getLayout());
            config.inConfs.push_back(indexesConfig);
            updatesConfig.desc = TensorDesc(dataPrecision, upd_dims,
                                            layer->insData[SCATTER_UPDATES].lock()->getTensorDesc().getLayout());
            config.inConfs.push_back(updatesConfig);

            DataConfig outConfig;
            outConfig.desc = TensorDesc(dataPrecision, dst_dims, layer->outData[0]->getTensorDesc().getLayout());
            outConfig.constant = false;
            outConfig.inPlace = 0;
            config.outConfs.push_back(outConfig);
            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[SCATTER_INDEXES]->getTensorDesc().getPrecision()) {
            case Precision::FP32:
                scatter<float>(inputs[SCATTER_DATA], inputs[SCATTER_INDEXES], inputs[SCATTER_UPDATES], outputs[0]);
                break;
            case Precision::I32:
                scatter<int32_t>(inputs[SCATTER_DATA], inputs[SCATTER_INDEXES], inputs[SCATTER_UPDATES], outputs[0]);
                break;
            default:
                return GENERAL_ERROR;
        }

        return OK;
    }

private:
    template <typename index_t>
    void scatter(Blob::Ptr data, Blob::Ptr indexes, Blob::Ptr updates, Blob::Ptr output) {
        const uint8_t *src_data = data->cbuffer().as<const uint8_t *>() + data->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const index_t *src_index = indexes->cbuffer().as<const index_t *>() + indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const uint8_t *src_updates = updates->cbuffer().as<const uint8_t *>() + updates->getTensorDesc().getBlockingDesc().getOffsetPadding();
        uint8_t *dst_data = output->cbuffer().as<uint8_t*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();
        size_t data_size = data->getTensorDesc().getPrecision().size();

        InferenceEngine::SizeVector index_dims = indexes->getTensorDesc().getDims();
        InferenceEngine::SizeVector data_dims = data->getTensorDesc().getDims();
        InferenceEngine::SizeVector dataStrides = data->getTensorDesc().getBlockingDesc().getStrides();

        if (src_data != dst_data) {
            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t start = 0, end = 0;
                splitter(output->size(), nthr, ithr, start, end);
                size_t size = (end - start) * data_size;
                start *= data_size;
                simple_copy(dst_data + start, size, src_data + start, size);
            });
        }

        parallel_nt(0, [&](const int ithr, const int nthr) {
            int j;
            size_t i, dst_idx = 0, start = 0, end = 0;
            SizeVector counters(index_dims.size(), 0);
            splitter(indexes->size(), nthr, ithr, start, end);
            for (j = index_dims.size() - 1, i = start; j >= 0; j--) {
                counters[j] = i % index_dims[j];
                i /= index_dims[j];
            }

            for (i = 0; i < static_cast<size_t>(axis); ++i)
                dst_idx += counters[i] * dataStrides[i];
            for (i++; i < data_dims.size(); ++i)
                dst_idx += counters[i] * dataStrides[i];

            for (size_t iwork = start; iwork < end; iwork++) {
                unsigned int idx = static_cast<unsigned int>(src_index[iwork]);
                if (idx < data_dims[axis])
                    simple_copy(dst_data + data_size * (dst_idx + idx * dataStrides[axis]), data_size,
                                src_updates + iwork * data_size, data_size);

                for (j = index_dims.size() - 1; j >= 0; j--) {
                    counters[j]++;
                    if (counters[j] < index_dims[j]) {
                        if (j != static_cast<size_t>(axis))
                            dst_idx += dataStrides[j];
                        break;
                    } else {
                        counters[j] = 0;
                        for (dst_idx = 0, i = 0; i < static_cast<size_t>(axis); ++i)
                            dst_idx += counters[i] * dataStrides[i];
                        for (i++; i < data_dims.size(); ++i)
                            dst_idx += counters[i] * dataStrides[i];
                    }
                }
            }
        });
    }

    int axis = 0;
    const size_t SCATTER_DATA = 0;
    const size_t SCATTER_INDEXES = 1;
    const size_t SCATTER_UPDATES = 2;
};

REG_FACTORY_FOR(ScatterImpl, ScatterUpdate);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
