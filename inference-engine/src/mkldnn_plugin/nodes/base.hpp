// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <legacy/ie_util_internal.hpp>
#include "nodes/list.hpp"

#include <string>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ExtLayerBase: public ILayerExecImpl {
public:
    StatusCode getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc *resp) noexcept override {
        if (!errorMsg.empty()) {
            if (resp) {
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        conf = confs;
        return OK;
    }

    StatusCode init(LayerConfig& config, ResponseDesc *resp) noexcept override {
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

protected:
    enum class ConfLayout { ANY, PLN, BLK8, BLK16 };

    class DataConfigurator {
    public:
        explicit DataConfigurator(ConfLayout l):
            layout(l) {}

        DataConfigurator(ConfLayout l, bool constant, int inplace = -1, Precision::ePrecision prc = Precision::UNSPECIFIED):
            layout(l), constant(constant), inplace(inplace), prc(prc) {}

        DataConfigurator(ConfLayout l, Precision::ePrecision prc):
            layout(l), prc(prc) {}

        ConfLayout layout;
        bool constant = false;
        int inplace = -1;
        Precision::ePrecision prc = Precision::UNSPECIFIED;     // by default use the layer precision
    };

    void addConfig(const CNNLayer* layer, std::vector<DataConfigurator> in_l,
            std::vector<DataConfigurator> out_l, bool dynBatchSupport = false) {
        LayerConfig config;

        if (in_l.size() != layer->insData.size())
            THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << layer->name << ". Expected " << layer->insData.size()
                << " but layout specification provided for " << in_l.size();
        if (out_l.size() != layer->outData.size())
            THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << layer->name << ". Expected " << layer->outData.size()
                << " but layout specification provided for " << out_l.size();

        // Fill tensor parameters into config
        auto fill_port = [] (std::vector<DataConfig>& port, DataConfigurator conf, const DataPtr& data) {
            auto div_up = [](const int a, const int b) -> int {
                if (!b)
                    return 0;
                return (a + b - 1) / b;
            };
            if (!data) THROW_IE_EXCEPTION << "Cannot get input data!";

            DataConfig dataConfig;
            dataConfig.inPlace = conf.inplace;
            dataConfig.constant = conf.constant;

            const TensorDesc& data_desc = data->getTensorDesc();
            const SizeVector& data_dims = data_desc.getDims();

            std::vector<size_t> blocks = data_dims;
            std::vector<size_t> order(blocks.size());
            for (size_t i = 0; i < order.size(); i++) order[i] = i;

            const bool isInt8 = (data->getPrecision() == Precision::I8 || data->getPrecision() == Precision::U8);

            if (conf.layout == ConfLayout::BLK8 || conf.layout == ConfLayout::BLK16) {
                if (data_dims.size() < 4 || data_dims.size() > 5)
                    THROW_IE_EXCEPTION << "Inapplicable blocking layout."
                        << "Tensor should be 4D or 5D.";

                int blk_size = conf.layout == ConfLayout::BLK8 ? 8 : 16;

                // Blocking through Channel dimension. Like [nChwXc]
                order.push_back(1);
                blocks[1] = div_up(blocks[1], blk_size);
                blocks.push_back(blk_size);
            } else if (isInt8) {
                if (data_dims.size() == 4) {
                    order = {0, 2, 3, 1};
                    blocks = {data_dims[0], data_dims[2], data_dims[3], data_dims[1]};
                } else if (data_dims.size() == 5) {
                    order = {0, 2, 3, 4, 1};
                    blocks = {data_dims[0], data_dims[2], data_dims[3], data_dims[4], data_dims[1]};
                }  // all over keep original plain format

                conf.layout = ConfLayout::PLN;
            }

            InferenceEngine::Precision precision = (conf.prc == Precision::UNSPECIFIED) ? data_desc.getPrecision() : Precision(conf.prc);
            if (conf.layout == ConfLayout::ANY) {
                dataConfig.desc = TensorDesc(precision, data_dims, InferenceEngine::Layout::ANY);
            } else {
                dataConfig.desc = TensorDesc(precision, data_dims, {blocks, order});
            }
            port.push_back(dataConfig);
        };

        for (size_t i = 0; i < in_l.size(); i++)
            fill_port(config.inConfs, in_l[i], layer->insData[i].lock());

        for (size_t i = 0; i < out_l.size(); i++)
            fill_port(config.outConfs, out_l[i], layer->outData[i]);

        config.dynBatchSupport = dynBatchSupport;
        confs.push_back(config);
    }
    std::string errorMsg;
    std::vector<LayerConfig> confs;
};

template <class IMPL>
class ImplFactory : public ILayerImplFactory {
public:
    explicit ImplFactory(const CNNLayer *layer) {
        cnnLayer = InferenceEngine::clonelayer(*layer);
        cnnLayer->_fusedWith = layer->_fusedWith;
        cnnLayer->insData = layer->insData;
        cnnLayer->outData = layer->outData;
    }

    // First implementation has more priority than next
    StatusCode getImplementations(std::vector<ILayerImpl::Ptr>& impls, ResponseDesc *resp) noexcept override {
        impls.push_back(ILayerImpl::Ptr(new IMPL(cnnLayer.get())));
        return OK;
    }
protected:
    InferenceEngine::CNNLayerPtr cnnLayer;
};

template <typename __prim>
inline void extRegister(MKLDNNExtensions * extInstance, const char * __type) {
    extInstance->AddExt(__type,
                [](const CNNLayer* layer) -> InferenceEngine::ILayerImplFactory* {
                    return new __prim(layer);
                });
}

#define REG_FACTORY_FOR(__prim, __type) \
    void __prim ## __type(MKLDNNExtensions * extInstance) { \
        extRegister<ImplFactory<__prim>>(extInstance, #__type); \
    }

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
