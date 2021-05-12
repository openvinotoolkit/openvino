// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include <algorithm>
#include <cassert>
#include <vector>
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset6.hpp>

using MKLDNNPlugin::TensorDescCreatorTypes;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ExperimentalDetectronTopKROIsImpl: public ExtLayerBase {
private:
    // Inputs:
    //      rois, shape [n, 4]
    //      rois_probs, shape [n]
    // Outputs:
    //      top_rois, shape [max_rois, 4]

    const int INPUT_ROIS {0};
    const int INPUT_PROBS {1};

    const int OUTPUT_ROIS {0};

    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            const auto topKROI = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronTopKROIs>(op);
            if (!topKROI) {
                errorMessage = "Only opset6 ExperimentalDetectronTopKROIs operation is supported";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    std::string errorPrefix;

public:
    explicit ExperimentalDetectronTopKROIsImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
              IE_THROW(NotImplemented) << errorMessage;
            }

            errorPrefix = "ExperimentalDetectronTopKROIs layer with name '" + op->get_friendly_name() + "'";
            const auto topKROI = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronTopKROIs>(op);
            if (op->get_input_size() != 2 || op->get_output_size() != 1)
                IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

            if (op->get_input_shape(INPUT_ROIS).size() != 2 || op->get_input_shape(INPUT_PROBS).size() != 1)
                IE_THROW() << errorPrefix << " has nsupported input shape";

            max_rois_num_ = topKROI->get_max_rois();

            addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32},
                           {TensorDescCreatorTypes::ncsp, Precision::FP32}},
                          {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const int input_rois_num = inputs[INPUT_ROIS]->getTensorDesc().getDims()[0];
        const int top_rois_num = (std::min)(max_rois_num_, input_rois_num);

        auto *input_rois = inputs[INPUT_ROIS]->buffer().as<const float *>();
        auto *input_probs = inputs[INPUT_PROBS]->buffer().as<const float *>();
        auto *output_rois = outputs[OUTPUT_ROIS]->buffer().as<float *>();

        std::vector<size_t> idx(input_rois_num);
        iota(idx.begin(), idx.end(), 0);
        // FIXME. partial_sort is enough here.
        sort(idx.begin(), idx.end(), [&input_probs](size_t i1, size_t i2) {return input_probs[i1] > input_probs[i2];});

        for (int i = 0; i < top_rois_num; ++i) {
            cpu_memcpy(output_rois + 4 * i, input_rois + 4 * idx[i], 4 * sizeof(float));
        }

        return OK;
    }

private:
    int max_rois_num_;
};

REG_FACTORY_FOR(ExperimentalDetectronTopKROIsImpl, ExperimentalDetectronTopKROIs);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
