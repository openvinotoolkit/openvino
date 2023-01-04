// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [fft_kernel:header]
#pragma once

#include <ie_iextension.h>

#include <ngraph/ngraph.hpp>

namespace TemplateExtension {

class FFTImpl : public InferenceEngine::ILayerExecImpl {
public:
    explicit FFTImpl(const std::shared_ptr<ngraph::Node>& node);
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                                                           InferenceEngine::ResponseDesc* resp) noexcept override;
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config,
                                     InferenceEngine::ResponseDesc* resp) noexcept override;
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs,
                                        std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                        InferenceEngine::ResponseDesc* resp) noexcept override;

private:
    ngraph::Shape inpShape;
    ngraph::Shape outShape;
    bool inverse;
    std::string error;
};

}  // namespace TemplateExtension
//! [fft_kernel:header]
