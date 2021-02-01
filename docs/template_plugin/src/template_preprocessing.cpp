// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_preprocessing.hpp"

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::AddPreprocessingMatcher, "AddPreprocessingMatcher", 0);

ngraph::pass::AddPreprocessingMatcher::AddPreprocessingMatcher(const InferenceEngine::InputsDataMap & inputInfoMap) {
    auto param = ngraph::pattern::wrap_type<ngraph::opset3::Parameter>();

    ngraph::matcher_pass_callback callback = [=] (pattern::Matcher& m) {
        auto param = std::dynamic_pointer_cast<ngraph::opset3::Parameter>(m.get_match_root());
        if (!param) {
            return false;
        }

        auto it = inputInfoMap.find(param->get_friendly_name());
        NGRAPH_CHECK(it != inputInfoMap.end(),
                     "Input ", param->get_friendly_name(), " is not found in the network.");

        const InferenceEngine::PreProcessInfo & pInfo = it->second->getPreProcess();

        bool has_scales = false, has_mean_values = false, has_mean_image = false;
        const size_t cn = pInfo.getNumberOfChannels();
        std::vector<float> meanValues(cn), stdScales(cn);
        InferenceEngine::Blob::Ptr meanImage = nullptr;

        for (size_t c = 0; c < cn; ++c) {
            if (pInfo[c]->stdScale != 1.0f) {
                stdScales[c] = pInfo[c]->stdScale;
                has_scales = true;
            }
            if (pInfo[c]->meanValue != 0.0f) {
                meanValues[c] = pInfo[c]->meanValue;
                has_mean_values = true;
            }
            if (pInfo[c]->meanData != nullptr) {
                has_mean_image = true;
                if (c == 0) {
                    meanImage = pInfo[c]->meanData;
                    NGRAPH_CHECK(meanImage->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32,
                        "Only InferenceEngine::Precision::FP32 precision is supported for PreProcessChannel::meanData");
                } else {
                    NGRAPH_CHECK(meanImage->getTensorDesc() == pInfo[c]->meanData->getTensorDesc(),
                        "TensorDesc for PreProcessChannel::meanData must be equal");
                }
            }
        }

        // no preprocessing
        if (!has_mean_values && !has_scales && !has_mean_image) {
            return false;
        }

        NGRAPH_CHECK(!(has_mean_image && has_scales),
                     "Only PreProcessChannel::meanData or PreProcessChannel::meanValue can be set.");

        std::shared_ptr<ngraph::op::Op> mul = nullptr, sub = nullptr;
        auto copy_param = param->clone_with_new_inputs({});

        if (has_mean_values) {
            auto mean_values_const = ngraph::opset3::Constant::create(ngraph::element::f32,
                ngraph::Shape{1, meanValues.size(), 1, 1}, meanValues);
            sub = std::make_shared<ngraph::opset3::Subtract>(copy_param, mean_values_const);
            sub->set_friendly_name(param->get_friendly_name() + "_mean_values");
        } else if (has_mean_image) {
            ngraph::Shape shape = { 1, cn };
            auto dims = meanImage->getTensorDesc().getDims();
            std::copy(dims.begin(), dims.end(), std::back_inserter(shape));

            std::vector<float> meanImageData(ngraph::shape_size(shape));
            for (size_t c = 0, i = 0; c < cn; ++c) {
                auto lm = pInfo[c]->meanData->buffer();
                const float *data = lm.as<const float *>();

                std::memcpy(&meanImageData[i], data, meanImage->byteSize());
                i += meanImage->size();
            }

            auto mean_values_const = ngraph::opset3::Constant::create(ngraph::element::f32,
                shape, meanImageData);
            sub = std::make_shared<ngraph::opset3::Subtract>(copy_param, mean_values_const);
            sub->set_friendly_name(param->get_friendly_name() + "_mean_image");
        }

        if (has_scales) {
            auto std_scales_const = ngraph::opset3::Constant::create(ngraph::element::f32,
                ngraph::Shape{1, stdScales.size(), 1, 1}, stdScales);
            auto parent = sub ? sub : copy_param;
            mul = std::make_shared<ngraph::opset3::Multiply>(parent, std_scales_const);
            mul->set_friendly_name(parent->get_friendly_name() + "_scales");
        }

        ngraph::replace_node(param, mul ? mul : sub);
        (sub ? sub: mul)->set_argument(0, param);

        // Return true as the root node was changed
        return true;
    };

    // Register pattern with Parameter operation as a pattern root node
    auto m = std::make_shared<ngraph::pattern::Matcher>(param, "AddPreprocessingMatcher");
    // Register Matcher
    register_matcher(m, callback);
}
