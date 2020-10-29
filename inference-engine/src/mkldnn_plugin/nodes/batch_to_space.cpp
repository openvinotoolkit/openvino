// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <set>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class BatchToSpaceImpl: public ExtLayerBase {
public:
    explicit BatchToSpaceImpl(const CNNLayer* layer) {
        try {
            const auto batchToSpaceLayer = dynamic_cast<const BatchToSpaceLayer*>(layer);
            if (!batchToSpaceLayer)
                THROW_IE_EXCEPTION << "'" << layer->name << "' layer is not instance of BatchToSpaceLayer class";

            if (batchToSpaceLayer->insData.size() != 4 || batchToSpaceLayer->outData.size() != 1)
                THROW_IE_EXCEPTION << "'" << batchToSpaceLayer->name << "' layer has incorrect number of input or output edges!";

            auto inData = batchToSpaceLayer->insData[0].lock();
            if (inData == nullptr)
                THROW_IE_EXCEPTION << "'" << batchToSpaceLayer->name << "' layer has nullable input data";

            if (inData->getLayout() != NCHW && inData->getLayout() != NCDHW)
                THROW_IE_EXCEPTION << "'" << batchToSpaceLayer->name << "' layer has unsupported layout: " << inData->getLayout();

            const auto precision = inData->getTensorDesc().getPrecision();
            const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8};
            if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end())
                THROW_IE_EXCEPTION << "'" << batchToSpaceLayer->name << "' layer has unsupported precision: " << precision.name();

            const SizeVector& in_dims = inData->getTensorDesc().getDims();
            const SizeVector& out_dims = layer->outData[0]->getTensorDesc().getDims();
            if (in_dims[1] != out_dims[1])
                THROW_IE_EXCEPTION << "'" << batchToSpaceLayer->name << "' layer has different IN and OUT channels number";

            _block_shape = batchToSpaceLayer->_block_shape;
            _crops_begin = batchToSpaceLayer->_crops_begin;
            _crops_end = batchToSpaceLayer->_crops_end;

            LayerConfig config;
            config.inConfs.resize(batchToSpaceLayer->insData.size());
            // TODO: remove Const layers
            for (int i = 0; i < batchToSpaceLayer->insData.size(); i++) {
                auto inData = batchToSpaceLayer->insData[i].lock();
                if (inData == nullptr)
                    THROW_IE_EXCEPTION << "'" << batchToSpaceLayer->name << "' layer has nullable input data";
                config.inConfs[i].desc = TensorDesc(precision,
                        inData->getTensorDesc().getDims(),
                        inData->getTensorDesc().getLayout());
            }

            DataConfig outConfig;
            outConfig.desc = TensorDesc(precision,
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
                    std::string errorMsg = "BatchToSpace layer does not support precision '"
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

        const size_t IB = inDims[0];
        const size_t IC = inDims[1];
        const size_t ID = layout == NCDHW ? inDims[dims_size - 3] : 1lu;
        const size_t IH = inDims[dims_size - 2];
        const size_t IW = inDims[dims_size - 1];

        const auto& outDims = outputs[0]->getTensorDesc().getDims();

        const size_t OB = outDims[0];
        const size_t OC = outDims[1];
        const size_t OD = layout == NCDHW ? outDims[dims_size - 3] : 1lu;
        const size_t OH = outDims[dims_size - 2];
        const size_t OW = outDims[dims_size - 1];

        const int64_t cBSD = layout == NCDHW ? _block_shape[dims_size - 3] : 1lu;  // Do not use name BSD. It affects MacOS build
        const int64_t BSH = _block_shape[dims_size - 2];
        const int64_t BSW = _block_shape[dims_size - 1];

        const size_t crop_front = layout == NCDHW ? _crops_begin[dims_size - 3] : 0lu;
        const size_t crop_top = _crops_begin[dims_size - 2];
        const size_t crop_left = _crops_begin[dims_size - 1];

        const size_t OH_OW = OH * OW;
        const size_t OD_OH_OW = OD * OH_OW;
        const size_t OC_OD_OH_OW = OC * OD_OH_OW;
        const size_t IH_IW = IH * IW;

        const size_t work_amount = IB*IC*ID*IH*IW;

        auto thread_body = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(work_amount, nthr, ithr, start, end);
            if (start >= end)
                return;
            int64_t ib(0), ic(0), id(0), ih(0), iw(0);
            parallel_it_init(start, ib, IB, ic, IC, id, ID, ih, IH, iw, IW);

            for (; ib < IB; ib++) {
                const size_t ob = ib % OB;
                const size_t ob_k = ob * OC_OD_OH_OW;
                int64_t b_idx = ib / OB;
                const int64_t ow_add = b_idx % BSW - crop_left;
                b_idx /= BSW;
                const int64_t oh_add = (layout == NCDHW ? b_idx % BSH : b_idx) - crop_top;
                const int64_t od_add = layout == NCDHW ? (b_idx / BSH - crop_front) : 0;
                for (; ic < IC; ic++) {
                    const size_t oc_k = ob_k + ic * OD_OH_OW;
                    for (; id < ID; id++) {
                        const int64_t od = id * cBSD + od_add;
                        if (od < 0 || od >= OD) {
                            start += IH_IW;
                            if (start >= end)
                                break;
                            continue;
                        }
                        const size_t od_k = oc_k + od * OH_OW;
                        for (; ih < IH; ih++) {
                            const int64_t oh = ih * BSH + oh_add;
                            if (oh < 0 || oh >= OH) {
                                start += IW;
                                if (start >= end)
                                    break;
                                continue;
                            }
                            const size_t oh_k = od_k + oh * OW;
                            for (; iw < IW; iw++) {
                                const int64_t ow = iw * BSW + ow_add;
                                if (ow < 0 || ow >= OW) {
                                    start++;
                                    if (start >= end)
                                        break;
                                    continue;
                                }
                                const size_t dst_idx = oh_k + ow;
                                dst_data[dst_idx] = src_data[start];
                                start++;
                                if (start >= end)
                                    break;
                            }
                            if (start >= end)
                                break;
                            iw = 0;
                        }
                        if (start >= end)
                            break;
                        ih = 0;
                    }
                    if (start >= end)
                        break;
                    id = 0;
                }
                if (start >= end)
                    break;
                ic = 0;
            }
        };

        parallel_nt(0, thread_body);
    }

private:
    std::vector<size_t> _block_shape;
    std::vector<size_t> _crops_begin;
    std::vector<size_t> _crops_end;
};

REG_FACTORY_FOR(ImplFactory<BatchToSpaceImpl>, BatchToSpace);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
