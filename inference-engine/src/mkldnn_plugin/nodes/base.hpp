// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include "nodes/list.hpp"
#include "common/blocked_desc_creator.h"
#include "ngraph/descriptor/tensor.hpp"
#include <ie_ngraph_utils.hpp>
#include "cpu_types.h"

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
    MKLDNNPlugin::Algorithm getAlgorithm() const {
        return algorithm;
    }
    MKLDNNPlugin::Algorithm algorithm;

    class DataConfigurator {
    public:
        DataConfigurator(MKLDNNPlugin::TensorDescCreatorTypes tensorDescType, Precision prc = Precision::UNSPECIFIED, bool constant = false, int inplace = -1) :
                tensorDescCreator(getTensorDescCreator(tensorDescType)), prc(prc), constant(constant), inplace(inplace) {}

        DataConfigurator(const MKLDNNPlugin::BlockedDescCreator::CreatorConstPtr& tensorDescCreator, Precision prc = Precision::UNSPECIFIED,
                         bool constant = false, int inplace = -1) : tensorDescCreator(tensorDescCreator), prc(prc), constant(constant), inplace(inplace) {}

        const MKLDNNPlugin::BlockedDescCreator::CreatorConstPtr tensorDescCreator;
        const bool constant = false;
        const int inplace = -1;
        const Precision prc = Precision::UNSPECIFIED; // By default ngraph node precision is used
    private:
        static MKLDNNPlugin::BlockedDescCreator::CreatorConstPtr getTensorDescCreator(MKLDNNPlugin::TensorDescCreatorTypes tensorDescType) {
            auto& creators = MKLDNNPlugin::BlockedDescCreator::getCommonCreators();
            if (creators.find(tensorDescType) == creators.end()) {
                IE_THROW() << "Cannot find tensor descriptor creator";
            }
            return creators.at(tensorDescType);
        }
    };

    void addConfig(const std::shared_ptr<ngraph::Node>& op,
                   const std::vector<DataConfigurator>& inDataConfigurators,
                   const std::vector<DataConfigurator>& outDataConfigurators,
                   bool dynBatchSupport = false) {
        LayerConfig config;

        if (inDataConfigurators.size() != op->get_input_size())
            IE_THROW() << "Cannot add config for operation " << op->get_friendly_name() << ". Incorrect number of inputs: " <<
                                  "expected: " << op->get_input_size() << ", provided: " << inDataConfigurators.size();
        if (outDataConfigurators.size() != op->get_output_size())
            IE_THROW() << "Cannot add config for operation " << op->get_friendly_name() << ". Incorrect number of outputs: " <<
                               "expected: " << op->get_output_size() << ", provided: " << outDataConfigurators.size();

        auto fill_port = [] (const DataConfigurator& dataConfigurator, const ngraph::descriptor::Tensor& tensor, std::vector<DataConfig>& port) -> bool {
            // In order to simplify particular node initialization logic we just don't add config in case target shape is not supported by tensorDescCreator.
            // This should be suitable for major of scenarios since almost all nodes add `ncsp` tensorDescCreator which supports any shape rank.
            if (tensor.get_shape().size() < dataConfigurator.tensorDescCreator->getMinimalRank())
                return false;

            auto precision = dataConfigurator.prc != Precision::UNSPECIFIED ? dataConfigurator.prc : details::convertPrecision(tensor.get_element_type());

            DataConfig dataConfig;
            dataConfig.inPlace = dataConfigurator.inplace;
            dataConfig.constant = dataConfigurator.constant;
            dataConfig.desc = dataConfigurator.tensorDescCreator->createDesc(precision, tensor.get_shape());

            port.push_back(dataConfig);

            return true;
        };

        for (size_t i = 0; i < inDataConfigurators.size(); i++)
            if (!fill_port(inDataConfigurators[i], op->get_input_tensor(i), config.inConfs))
                return;

        for (size_t i = 0; i < outDataConfigurators.size(); i++)
            if (!fill_port(outDataConfigurators[i], op->get_output_tensor(i), config.outConfs))
                return;

        config.dynBatchSupport = dynBatchSupport;
        confs.push_back(config);
    }

    std::string errorMsg;
    std::vector<LayerConfig> confs;
};

template <class IMPL>
class ImplFactory : public ILayerImplFactory {
public:
    explicit ImplFactory(const std::shared_ptr<ngraph::Node>& op) : ngraphOp(op) {}

    // First implementation has more priority than next
    StatusCode getImplementations(std::vector<ILayerImpl::Ptr>& impls, ResponseDesc *resp) noexcept override {
        try {
            impls.push_back(ILayerImpl::Ptr(new IMPL(ngraphOp)));
        } catch (const InferenceEngine::Exception& ex) {
            strncpy(resp->msg, ex.what(), sizeof(resp->msg) - 1);
            IE_SUPPRESS_DEPRECATED_START
            return ex.getStatus() != OK ? ex.getStatus() : GENERAL_ERROR;
            IE_SUPPRESS_DEPRECATED_END
        }
        return OK;
    }
protected:
    const std::shared_ptr<ngraph::Node> ngraphOp;
};

#define REG_FACTORY_FOR(__prim, __type) \
    void __prim ## __type(MKLDNNExtensions * extInstance) { \
        using namespace MKLDNNPlugin; \
        extInstance->layersFactory.registerNodeIfRequired(MKLDNNPlugin, __type, OV_PP_TOSTRING(__type), ImplFactory<__prim>); \
    }

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
