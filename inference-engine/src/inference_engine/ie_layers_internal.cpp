// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include "ie_layers_internal.hpp"

namespace InferenceEngine {

Paddings getConvPaddings(const ConvolutionLayer &convLayer) {
    std::string errorPrefix = "Failed to calculate padding for Convolution: ";
    const std::map<std::string, std::string> &params = convLayer.params;
    const std::vector<DataWeakPtr> &insData = convLayer.insData;
    try {
        auto it = params.find("auto_pad");
        std::string padType;
        if (it != params.end()) {
            if (it->second == "valid") {
                return {PropertyVector<unsigned>(2, 0), PropertyVector<unsigned>(2, 0)};
            } else {
                if (insData.size() != 1) THROW_IE_EXCEPTION << "number of inputs should be equal 1";
                auto firstInput = insData[0].lock();
                if (!firstInput) THROW_IE_EXCEPTION << "input is empty";
                auto shape = firstInput->getTensorDesc().getDims();
                if (shape.size() != 4) THROW_IE_EXCEPTION << "input shape must be 4D";

                int SH = convLayer._stride[Y_AXIS];
                int SW = convLayer._stride[X_AXIS];

                int IH = shape[2];
                int IW = shape[3];

                int KH = 0, KW = 0;
                if (convLayer._dilation[Y_AXIS])
                    KH = (convLayer._kernel[Y_AXIS] - 1) * convLayer._dilation[Y_AXIS] + 1;
                else
                    KH = convLayer._kernel[Y_AXIS];
                if (convLayer._dilation[X_AXIS])
                    KW = (convLayer._kernel[X_AXIS] - 1) * convLayer._dilation[X_AXIS] + 1;
                else
                    KW = convLayer._kernel[X_AXIS];
                int PAH, PAW;
                if (IH % SH == 0) {
                    PAH = std::max(KH - SH, 0);
                } else {
                    PAH = std::max(KH - (IH % SH), 0);
                }
                if (IW % SW == 0) {
                    PAW = std::max(KW - SW, 0);
                } else {
                    PAW = std::max(KW - (IW % SW), 0);
                }

                unsigned top = PAH / 2;
                unsigned bottom = PAH - top;
                unsigned left = PAW / 2;
                unsigned right = PAW - left;

                PropertyVector<unsigned int> pad_begin;
                pad_begin.insert(X_AXIS, left);
                pad_begin.insert(Y_AXIS, top);

                PropertyVector<unsigned int> pad_end;
                pad_end.insert(X_AXIS, right);
                pad_end.insert(Y_AXIS, bottom);
                return {pad_begin, pad_end};
            }
        }
        return {convLayer._padding, convLayer._pads_end};
    } catch (const InferenceEngine::details::InferenceEngineException &iee) {
        THROW_IE_EXCEPTION << errorPrefix << iee.what();
    }
}

}  // namespace InferenceEngine
