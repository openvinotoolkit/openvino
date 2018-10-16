// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_base.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <cassert>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

inline int div_up(const int a, const int b) {
    assert(b);
    return (a + b - 1) / b;
}

StatusCode
ExtLayerBase::getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc *resp) noexcept {
    if (!errorMsg.empty()) {
        if (resp) {
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
        }
        return GENERAL_ERROR;
    }
    conf = confs;
    return OK;
}

StatusCode
ExtLayerBase::init(LayerConfig& config, ResponseDesc *resp) noexcept {
    for (auto& input : config.inConfs) {
        for (auto& offset : input.desc.getBlockingDesc().getOffsetPaddingToData()) {
            if (offset) {
                return GENERAL_ERROR;
            }
        }
        if (input.desc.getBlockingDesc().getOffsetPadding()) {
            return GENERAL_ERROR;
        }
    }
    for (auto& output : config.outConfs) {
        for (auto& offset : output.desc.getBlockingDesc().getOffsetPaddingToData()) {
            if (offset) {
                return GENERAL_ERROR;
            }
        }
        if (output.desc.getBlockingDesc().getOffsetPadding()) {
            return GENERAL_ERROR;
        }
    }
    return OK;
}

void ExtLayerBase::addConfig(const CNNLayer* layer, std::vector<DataConfigurator> in_l, std::vector<DataConfigurator> out_l, bool dynBatchSupport) {
    LayerConfig config;

    if (in_l.size() != layer->insData.size())
        THROW_IE_EXCEPTION << "Incorrect number of input edges. Expected " << layer->insData.size()
                           << " but layout specification provided for " << in_l.size();
    if (out_l.size() != layer->outData.size())
        THROW_IE_EXCEPTION << "Incorrect number of input edges. Expected " << layer->outData.size()
                           << " but layout specification provided for " << out_l.size();

    // Fill tensor parameters into config
    auto fill_port = [] (std::vector<DataConfig>& port, DataConfigurator conf, const DataPtr& data) {
        if (!data) THROW_IE_EXCEPTION << "Cannot get input data!";

        DataConfig dataConfig;
        dataConfig.inPlace = conf.inplace;
        dataConfig.constant = conf.constant;

        const TensorDesc& data_desc = data->getTensorDesc();
        const SizeVector& data_dims = data_desc.getDims();

        std::vector<size_t> blocks = data_dims;
        std::vector<size_t> order(blocks.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;

        if (conf.layout == ConfLayout::BLK8 || conf.layout == ConfLayout::BLK16) {
            if (data_dims.size() != 4)
                THROW_IE_EXCEPTION << "Inapplicable blocking layout."
                                   << "Tensor should be 4D.";

            int blk_size = conf.layout == ConfLayout::BLK8 ? 8 : 16;

            // Blocking through Channel dimension. Like [nChwXc]
            order.push_back(1);
            blocks[1] = div_up(blocks[1], blk_size);
            blocks.push_back(blk_size);
        }

        // All extension layers support only FP32 precision!
        InferenceEngine::Precision precision = conf.constant ? data_desc.getPrecision() : InferenceEngine::Precision(InferenceEngine::Precision::FP32);
        if (conf.layout == ConfLayout::ANY) {
            dataConfig.desc = TensorDesc(precision, data_dims, InferenceEngine::Layout::ANY);
        } else {
            dataConfig.desc = TensorDesc(precision, data_dims, {blocks, order});
        }
        port.push_back(dataConfig);
    };

    for (int i = 0; i < in_l.size(); i++)
        fill_port(config.inConfs, in_l[i], layer->insData[i].lock());

    for (int i = 0; i < out_l.size(); i++)
        fill_port(config.outConfs, out_l[i], layer->outData[i]);

    config.dynBatchSupport = dynBatchSupport;
    confs.push_back(config);
}


}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
