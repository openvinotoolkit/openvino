// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <set>
#include <cassert>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class SpaceToBatchImpl: public ExtLayerBase {
public:
    explicit SpaceToBatchImpl(const CNNLayer* layer) {
        try {
            auto spaceToBatchLayer = dynamic_cast<const SpaceToBatchLayer*>(layer);
            if (!spaceToBatchLayer)
                THROW_IE_EXCEPTION << "'" << layer->name << "' layer is not instance of SpaceToBatchLayer class";

            if (spaceToBatchLayer->insData.size() != 4 || spaceToBatchLayer->outData.size() != 1)
                THROW_IE_EXCEPTION << "'" << spaceToBatchLayer->name << "' layer has incorrect number of input or output edges!";

            auto inData = spaceToBatchLayer->insData[0].lock();
            if (inData == nullptr)
                THROW_IE_EXCEPTION << "'" << spaceToBatchLayer->name << "' layer has nullable input data";

            if (inData->getLayout() != NCHW && inData->getLayout() != NCDHW)
                THROW_IE_EXCEPTION << "'" << spaceToBatchLayer->name << "' layer has unsupported layout: " << inData->getLayout();

            const auto precision = inData->getTensorDesc().getPrecision();
            const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8};
            if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end())
                THROW_IE_EXCEPTION << "'" << spaceToBatchLayer->name << "' layer has unsupported precision: " << precision.name();

            const SizeVector& in_dims = inData->getTensorDesc().getDims();
            const SizeVector& out_dims = layer->outData[0]->getTensorDesc().getDims();
            if (in_dims[1] != out_dims[1])
                THROW_IE_EXCEPTION << "'" << spaceToBatchLayer->name << "' layer has different IN and OUT channels number";

            _block_shape = spaceToBatchLayer->_block_shape;
            _pads_begin = spaceToBatchLayer->_pads_begin;
            _pads_end = spaceToBatchLayer->_pads_end;

            LayerConfig config;
            config.inConfs.resize(spaceToBatchLayer->insData.size());
            // TODO: remove Const layers
            for (int i = 0; i < spaceToBatchLayer->insData.size(); i++) {
                auto inData = spaceToBatchLayer->insData[i].lock();
                if (inData == nullptr)
                    THROW_IE_EXCEPTION << "'" << spaceToBatchLayer->name << "' layer has nullable input data";
                config.inConfs[i].desc = TensorDesc(inData->getTensorDesc().getPrecision(),
                        inData->getTensorDesc().getDims(),
                        inData->getTensorDesc().getLayout());
            }

            DataConfig outConfig;
            outConfig.desc = TensorDesc(layer->outData[0]->getTensorDesc().getPrecision(),
                    out_dims,
                    layer->outData[0]->getTensorDesc().getLayout());
            config.outConfs.push_back(outConfig);
            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[0]->getTensorDesc().getPrecision().size()) {
            case 1: {
                process_data<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs);
                break;
            }
            case 2: {
                process_data<PrecisionTrait<Precision::U16>::value_type>(inputs, outputs);
                break;
            }
            case 4: {
                process_data<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs);
                break;
            }
            case 8: {
                process_data<PrecisionTrait<Precision::U64>::value_type>(inputs, outputs);
                break;
            }
            default: {
                if (resp) {
                    std::string errorMsg = "SpaceToBatch layer does not support precision '"
                            + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
            }
        }

        return OK;
    }

    template<typename T>
    void process_data(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs) noexcept {
        const T* src_data = inputs[0]->cbuffer().as<const T*>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T* dst_data = outputs[0]->buffer().as<T*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto& inDims = inputs[0]->getTensorDesc().getDims();
        const size_t dims_size = inDims.size();
        const auto layout = inputs[0]->getTensorDesc().getLayout();

        const int64_t IB = inDims[0];
        const int64_t IC = inDims[1];
        const int64_t ID = layout == NCDHW ? inDims[dims_size - 3] : 1lu;
        const int64_t IH = inDims[dims_size - 2];
        const int64_t IW = inDims[dims_size - 1];

        const auto& outDims = outputs[0]->getTensorDesc().getDims();

        const size_t OB = outDims[0];
        const size_t OC = outDims[1];
        const size_t OD = layout == NCDHW ? outDims[dims_size - 3] : 1lu;
        const size_t OH = outDims[dims_size - 2];
        const size_t OW = outDims[dims_size - 1];

        const int64_t cBSD = layout == NCDHW ? _block_shape[dims_size - 3] : 1lu;  // Do not use name BSD. It affects MacOS build
        const int64_t BSH = _block_shape[dims_size - 2];
        const int64_t BSW = _block_shape[dims_size - 1];

        const int64_t PF = layout == NCDHW ? _pads_begin[dims_size - 3] : 0;
        const int64_t PT = _pads_begin[dims_size - 2];
        const int64_t PL = _pads_begin[dims_size - 1];

        const size_t OH_OW = OH * OW;
        const size_t IH_IW = IH * IW;
        const size_t ID_IH_IW = ID * IH_IW;
        const size_t IC_ID_IH_IW = IC * ID_IH_IW;

        const size_t work_amount = OB*OC*OD*OH*OW;

        auto thread_body = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(work_amount, nthr, ithr, start, end);
            if (start >= end)
                return;
            int64_t ob(0), oc(0), od(0), oh(0), ow(0);
            parallel_it_init(start, ob, OB, oc, OC, od, OD, oh, OH, ow, OW);

            for (; ob < OB; ob++) {
                const int64_t ib = ob % IB;
                const int64_t ib_k = ib * IC_ID_IH_IW;
                int64_t bi = ob / IB;
                const int64_t shift_w = bi % BSW - PL;
                bi /= BSW;
                const int64_t shift_h = (layout == NCDHW ? bi % BSH : bi) - PT;
                const int64_t shift_d = layout == NCDHW ? (bi / BSH - PF) : 0;
                for (; oc < OC; oc++) {
                    const int64_t ic_k = ib_k + oc * ID_IH_IW;
                    for (; od < OD; od++) {
                        const int64_t id = od * cBSD + shift_d;
                        if (id < 0 || id >= ID) {
                            std::fill(dst_data + start, dst_data + start + OH_OW, T(0));
                            start += OH_OW;
                            if (start >= end)
                                break;
                            continue;
                        }
                        const int64_t id_k = ic_k + id * IH_IW;
                        for (; oh < OH; oh++) {
                            const int64_t ih = oh * BSH + shift_h;
                            if (ih < 0 || ih >= IH) {
                                std::fill(dst_data + start, dst_data + start + OW, T(0));
                                start += OW;
                                if (start >= end)
                                    break;
                                continue;
                            }
                            const int64_t ih_k = id_k + ih * IW;
                            for (; ow < OW; ow++) {
                                const int64_t iw = ow * BSW + shift_w;
                                if (iw < 0 || iw >= IW) {
                                    dst_data[start] = T(0);
                                    start++;
                                    if (start >= end)
                                        break;
                                    continue;
                                }
                                const int64_t src_idx = ih_k + iw;
                                dst_data[start] = src_data[src_idx];
                                start++;
                                if (start >= end)
                                    break;
                            }
                            if (start >= end)
                                break;
                            ow = 0;
                        }
                        if (start >= end)
                            break;
                        oh = 0;
                    }
                    if (start >= end)
                        break;
                    od = 0;
                }
                if (start >= end)
                    break;
                oc = 0;
            }
        };

        parallel_nt(0, thread_body);
    }

private:
    std::vector<size_t> _block_shape;
    std::vector<size_t> _pads_begin;
    std::vector<size_t> _pads_end;
};

REG_FACTORY_FOR(SpaceToBatchImpl, SpaceToBatch);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
