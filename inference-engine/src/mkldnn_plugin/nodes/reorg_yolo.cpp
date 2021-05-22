// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include <vector>
#include <ngraph/opsets/opset2.hpp>

using namespace MKLDNNPlugin;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ReorgYoloImpl: public ExtLayerBase {
public:
    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            const auto reorgYolo = std::dynamic_pointer_cast<const ngraph::opset2::ReorgYolo>(op);
            if (!reorgYolo) {
                errorMessage = "Only opset2 ReorgYolo operation is supported";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    explicit ReorgYoloImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            errorPrefix = std::string(op->get_type_name()) + " node with name '" + op->get_friendly_name() + "'";
            if (op->get_input_size() != 1 || op->get_output_size() != 1)
                IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

            const auto reorgYolo = std::dynamic_pointer_cast<const ngraph::opset2::ReorgYolo>(op);
            const auto strides = reorgYolo->get_strides();
            if (strides.empty())
                IE_THROW() << errorPrefix << " has empty strides";
            stride = strides[0];

            addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32}},
                          {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const auto *src_data = inputs[0]->cbuffer().as<const float *>();
        auto *dst_data = outputs[0]->buffer().as<float *>();

        int IW = (inputs[0]->getTensorDesc().getDims().size() > 3) ? inputs[0]->getTensorDesc().getDims()[3] : 1;
        int IH = (inputs[0]->getTensorDesc().getDims().size() > 2) ? inputs[0]->getTensorDesc().getDims()[2] : 1;
        int IC = (inputs[0]->getTensorDesc().getDims().size() > 1) ? inputs[0]->getTensorDesc().getDims()[1] : 1;
        int B = (inputs[0]->getTensorDesc().getDims().size() > 0) ? inputs[0]->getTensorDesc().getDims()[0] : 1;

        int ic_off = IC / (stride * stride);
        int ih_off = IH * stride;
        int iw_off = IW * stride;
        for (int b = 0; b < B; b++) {
            for (int ic = 0; ic < IC; ic++) {
                for (int ih = 0; ih < IH; ih++) {
                    for (int iw = 0; iw < IW; iw++) {
                        int dstIndex = b * IC * IH * IW + ic * IH * IW + ih * IW + iw;

                        int oc = ic % ic_off;
                        int offset = ic / ic_off;

                        int ow = iw * stride + offset % stride;
                        int oh = ih * stride + offset / stride;

                        int srcIndex = b * ic_off * ih_off * iw_off + oc * ih_off * iw_off + oh * iw_off + ow;

                        dst_data[dstIndex] = src_data[srcIndex];
                    }
                }
            }
        }
        return OK;
    }

private:
    int stride;

    std::string errorPrefix;
};

REG_FACTORY_FOR(ReorgYoloImpl, ReorgYolo);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
