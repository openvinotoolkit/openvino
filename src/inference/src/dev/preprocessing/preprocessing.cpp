// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing.hpp"

#include "add_scale.hpp"
#include "mean_image_or_value.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"

ov::pass::AddPreprocessing::AddPreprocessing(const InferenceEngine::InputsDataMap& inputInfoMap)
    : m_inputInfoMap(inputInfoMap) {}

bool ov::pass::AddPreprocessing::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(AddPreprocessing);
    ov::pass::AddMeanSubtract::MeanMap meanMap;
    ov::pass::AddScale::ScaleMap scaleMap;

    for (const auto& it : m_inputInfoMap) {
        bool has_scales = false, has_mean_values = false, has_mean_image = false;
        const InferenceEngine::PreProcessInfo& pInfo = it.second->getPreProcess();
        const auto& inputDims = it.second->getTensorDesc().getDims();
        const size_t cn = pInfo.getNumberOfChannels();
        std::vector<float> meanValues(cn), stdScales(cn);
        InferenceEngine::Blob::Ptr meanImage = nullptr;

        for (size_t c = 0; c < cn; ++c) {
            if ((stdScales[c] = pInfo[c]->stdScale) != 1.0f) {
                has_scales = true;
            }

            if ((meanValues[c] = pInfo[c]->meanValue) != 0.0f) {
                has_mean_values = true;
            }

            if (pInfo[c]->meanData != nullptr) {
                has_mean_image = true;
                if (c == 0) {
                    meanImage = pInfo[c]->meanData;
                    OPENVINO_ASSERT(
                        meanImage->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32,
                        "Only InferenceEngine::Precision::FP32 precision is supported for PreProcessChannel::meanData");
                } else {
                    OPENVINO_ASSERT(meanImage->getTensorDesc() == pInfo[c]->meanData->getTensorDesc(),
                                    "TensorDesc for PreProcessChannel::meanData must be equal");
                }
            }
        }

        // no preprocessing for current input
        if (!has_mean_values && !has_scales && !has_mean_image) {
            continue;
        }

        OPENVINO_ASSERT(!(has_mean_image && has_scales),
                        "Only PreProcessChannel::meanData or PreProcessChannel::meanValue can be set.");

        if (has_scales) {
            ov::Shape shape(inputDims.size(), 1);
            shape[1] = stdScales.size();  // C
            scaleMap[it.first] = ov::op::v0::Constant::create(ov::element::f32, shape, stdScales);
        }

        if (has_mean_values) {
            ov::Shape shape(inputDims.size(), 1);
            shape[1] = meanValues.size();  // C
            meanMap[it.first] = ov::op::v0::Constant::create(ov::element::f32, shape, meanValues);
        } else if (has_mean_image) {
            ov::Shape shape = {cn};
            auto dims = meanImage->getTensorDesc().getDims();
            std::copy(dims.begin(), dims.end(), std::back_inserter(shape));

            std::vector<float> meanImageData(ov::shape_size(shape));
            for (size_t c = 0, i = 0; c < cn; ++c) {
                auto lm = pInfo[c]->meanData->buffer();
                const float* data = lm.as<const float*>();

                std::memcpy(&meanImageData[i], data, meanImage->byteSize());
                i += meanImage->size();
            }

            meanMap[it.first] = ov::op::v0::Constant::create(ov::element::f32, shape, meanImageData);
        }
    }

    ov::pass::Manager manager(get_pass_config());
    auto preproc = manager.register_pass<ov::pass::GraphRewrite>();

    if (!scaleMap.empty()) {
        preproc->add_matcher<ov::pass::AddScale>(scaleMap);
    }
    if (!meanMap.empty()) {
        preproc->add_matcher<ov::pass::AddMeanSubtract>(meanMap);
    }

    manager.run_passes(f);

    return false;
}
