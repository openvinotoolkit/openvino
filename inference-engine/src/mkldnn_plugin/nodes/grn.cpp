// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include "ie_parallel.hpp"
#include <ngraph/opsets/opset1.hpp>

using namespace MKLDNNPlugin;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class GRNImpl: public ExtLayerBase {
    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            const auto grn = std::dynamic_pointer_cast<const ngraph::opset1::GRN>(op);
            if (!grn) {
                errorMessage = "Only opset1 GRN operation is supported";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    std::string errorPrefix;

public:
    explicit GRNImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            errorPrefix = "GRN layer with name '" + op->get_friendly_name() + "'";
            const auto grn = std::dynamic_pointer_cast<const ngraph::opset1::GRN>(op);

            if (op->get_input_size() != 1 || op->get_output_size() != 1)
                IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

            bias = grn->get_bias();

            addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32, false, 0}},
                          {{TensorDescCreatorTypes::ncsp, Precision::FP32, false, 0}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        SizeVector dims = inputs[0]->getTensorDesc().getDims();

        int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
        int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
        int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
        int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

        parallel_for3d(N, H, W, [&](int b, int h, int w) {
            double variance = 0;
            for (int c = 0; c < C; c++) {
                variance += std::pow(src_data[b*C*H*W + c*H*W + h*W + w], 2);
            }
            variance = std::pow(variance + bias, 0.5f);
            for (int c = 0; c < C; c++) {
                dst_data[b*C*H*W + c*H*W + h*W + w] = src_data[b*C*H*W + c*H*W + h*W + w] / static_cast<float>(variance);
            }
        });
        return OK;
    }

private:
    float bias = 1.0f;
};

REG_FACTORY_FOR(GRNImpl, GRN);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
