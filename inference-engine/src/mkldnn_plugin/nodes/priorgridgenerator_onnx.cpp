// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include <algorithm>
#include <cassert>
#include <vector>
#include <ngraph/opsets/opset6.hpp>

using MKLDNNPlugin::TensorDescCreatorTypes;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

const int INPUT_PRIORS {0};
const int INPUT_FEATUREMAP {1};
const int INPUT_IMAGE {2};

const int OUTPUT_ROIS {0};

class ExperimentalDetectronPriorGridGeneratorImpl: public ExtLayerBase {
private:
    // Inputs:
    //      priors, shape [n, 4]
    //      [feature_map], shape [b, c, h, w]
    //      [im_data], shape [b, 3, im_h, im_w]
    // Outputs:
    //      priors_grid, shape [m, 4]

    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            const auto priorGridGen = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronPriorGridGenerator>(op);
            if (!priorGridGen) {
                errorMessage = "Only opset6 ExperimentalDetectronPriorGridGenerator operation is supported";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    std::string errorPrefix;

public:
    explicit ExperimentalDetectronPriorGridGeneratorImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            errorPrefix = "ExperimentalDetectronPriorGridGenerator layer with name '" + op->get_friendly_name() + "'";
            const auto priorGridGen = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronPriorGridGenerator>(op);
            if (op->get_input_size() != 3 || op->get_output_size() != 1)
                IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

            if (op->get_input_shape(INPUT_PRIORS).size() != 2 ||
                op->get_input_shape(INPUT_FEATUREMAP).size() != 4 ||
                    op->get_input_shape(INPUT_IMAGE).size() != 4)
                IE_THROW() << errorPrefix << " has unsupported input shape";

            const auto &attr = priorGridGen->get_attrs();
            grid_w_ = attr.w;
            grid_h_ = attr.h;
            stride_h_ = attr.stride_y;
            stride_w_ = attr.stride_x;

            addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32},
                           {TensorDescCreatorTypes::ncsp, Precision::FP32},
                           {TensorDescCreatorTypes::ncsp, Precision::FP32}},
                          {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const int num_priors_ = inputs[INPUT_PRIORS]->getTensorDesc().getDims()[0];
        assert(inputs[INPUT_PRIORS]->getTensorDesc().getDims()[1] == 4);

        // Execute
        const int layer_width = grid_w_ ? grid_w_ : inputs[INPUT_FEATUREMAP]->getTensorDesc().getDims()[3];
        const int layer_height = grid_h_ ? grid_h_ : inputs[INPUT_FEATUREMAP]->getTensorDesc().getDims()[2];
        const float step_w = stride_w_ ? stride_w_ : static_cast<float>(inputs[INPUT_IMAGE]->getTensorDesc().getDims()[3]) / layer_width;
        const float step_h = stride_h_ ? stride_h_ : static_cast<float>(inputs[INPUT_IMAGE]->getTensorDesc().getDims()[2]) / layer_height;

        const auto *bottom_data_0 = inputs[0]->buffer().as<const float *>();
        auto *top_data_0 = outputs[OUTPUT_ROIS]->buffer().as<float *>();

        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width; ++w) {
                for (int s = 0; s < num_priors_; ++s) {
                    top_data_0[0] = bottom_data_0[4 * s + 0] + step_w * (w + 0.5f);
                    top_data_0[1] = bottom_data_0[4 * s + 1] + step_h * (h + 0.5f);
                    top_data_0[2] = bottom_data_0[4 * s + 2] + step_w * (w + 0.5f);
                    top_data_0[3] = bottom_data_0[4 * s + 3] + step_h * (h + 0.5f);
                    top_data_0 += 4;
                }
            }
        }

        return OK;
    }

private:
    int grid_w_;
    int grid_h_;
    float stride_w_;
    float stride_h_;
};


REG_FACTORY_FOR(ExperimentalDetectronPriorGridGeneratorImpl, ExperimentalDetectronPriorGridGenerator);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
