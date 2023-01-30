// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing.hpp"

#include "dev/converter_utils.hpp"
#include "ie_ngraph_utils.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"

bool ov::pass::AddPreprocessing::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(AddPreprocessing);
    ov::preprocess::PrePostProcessor preproc(model);

    for (size_t i = 0; i < model->inputs().size(); i++) {
        ov::Output<const Node> const_input(model->input(i).get_node(), model->input(i).get_index());
        InferenceEngine::InputInfo::Ptr input_info;
        // I don't remove rt info to have information in InputsInfo about pre-processing in legacy
        // ExecutableNetwork
        ov::legacy_convert::fill_input_info(const_input, input_info);
        OPENVINO_ASSERT(input_info);

        auto& legacy_preproc = input_info->getPreProcess();

        preproc.input(i).tensor().set_element_type(
            InferenceEngine::details::convertPrecision(input_info->getPrecision()));
        std::stringstream stream;
        stream << input_info->getLayout();
        preproc.input(i).tensor().set_layout(ov::Layout{stream.str()});

        // Resize
        switch (legacy_preproc.getResizeAlgorithm()) {
        case InferenceEngine::ResizeAlgorithm::RESIZE_AREA:
            preproc.input(i).preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST);
            preproc.input(i).tensor().set_spatial_dynamic_shape();
            break;
        case InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR:
            preproc.input(i).preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
            preproc.input(i).tensor().set_spatial_dynamic_shape();
            break;
        default:
            // nothing to do
            break;
        }
        preproc.input(i).model().set_layout("NCHW");

        switch (legacy_preproc.getMeanVariant()) {
        case InferenceEngine::MEAN_IMAGE: {
            ov::Shape shape(input_info->getTensorDesc().getDims());
            std::vector<float> scale;
            std::vector<float> meanImageData(ov::shape_size(shape));
            for (size_t c = 0, i = 0; c < legacy_preproc.getNumberOfChannels(); ++c) {
                auto blob = legacy_preproc[i]->meanData;

                auto lm = blob->buffer();
                const float* data = lm.as<const float*>();

                std::memcpy(&meanImageData[i], data, blob->byteSize());
                i += blob->size();
                scale.emplace_back(legacy_preproc[i]->stdScale);
            }
            preproc.input(i).preprocess().mean(meanImageData).scale(scale);
            break;
        }
        case InferenceEngine::MEAN_VALUE: {
            std::vector<float> mean, scale;
            for (size_t i = 0; i < legacy_preproc.getNumberOfChannels(); i++) {
                mean.emplace_back(legacy_preproc[i]->meanValue);
                scale.emplace_back(legacy_preproc[i]->stdScale);
            }
            preproc.input(i).preprocess().mean(mean).scale(scale);
            break;
        }
        default:
            break;
        }
    }
    auto& non_const_ptr = const_cast<std::shared_ptr<ov::Model>&>(model);
    non_const_ptr = preproc.build();

    return false;
}
