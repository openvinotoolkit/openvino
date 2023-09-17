// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing.hpp"

#include "dev/converter_utils.hpp"
#include "dev/preprocessing/mean_image.hpp"
#include "ie_common.h"
#include "ie_ngraph_utils.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/preprocess/color_format.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"

bool ov::pass::AddPreprocessing::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(AddPreprocessing);
    ov::preprocess::PrePostProcessor preproc(model);
    ov::pass::AddMeanImage::MeanMap meanMap;

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

        if (input_info->getLayout() != InferenceEngine::Layout::BLOCKED &&
            input_info->getLayout() != InferenceEngine::Layout::SCALAR) {
            std::stringstream stream;
            stream << input_info->getLayout();
            preproc.input(i).tensor().set_layout(ov::Layout{stream.str()});
        }

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

        switch (legacy_preproc.getMeanVariant()) {
        case InferenceEngine::MEAN_IMAGE: {
            ov::Shape shape(input_info->getTensorDesc().getDims());
            std::vector<float> scale;
            std::vector<float> meanImageData(ov::shape_size(shape));
            for (size_t c = 0, i = 0; c < legacy_preproc.getNumberOfChannels(); ++c) {
                auto blob = legacy_preproc[c]->meanData;

                auto lm = blob->buffer();
                const float* data = lm.as<const float*>();

                std::memcpy(&meanImageData[i], data, blob->byteSize());
                i += blob->size();
                scale.emplace_back(legacy_preproc[c]->stdScale);
            }
            meanMap[input_info->name()] = ov::op::v0::Constant::create(ov::element::f32, shape, meanImageData);
            preproc.input(i).preprocess().scale(scale);
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

        switch (legacy_preproc.getColorFormat()) {
        case InferenceEngine::ColorFormat::BGR:
            preproc.input(i).tensor().set_color_format(ov::preprocess::ColorFormat::BGR);
            preproc.input(i).preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
            break;
        case InferenceEngine::ColorFormat::RGB:
            preproc.input(i).tensor().set_color_format(ov::preprocess::ColorFormat::RGB);
            preproc.input(i).preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
            break;
        case InferenceEngine::ColorFormat::RGBX:
            preproc.input(i).tensor().set_color_format(ov::preprocess::ColorFormat::RGBX);
            preproc.input(i).preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
            break;
        case InferenceEngine::ColorFormat::BGRX:
            preproc.input(i).tensor().set_color_format(ov::preprocess::ColorFormat::BGRX);
            preproc.input(i).preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
            break;
        default:
            break;
        }

        if (const_input.get_partial_shape().is_static() && const_input.get_shape().size() == 4)
            preproc.input(i).model().set_layout("NCHW");
    }
    std::vector<std::string> legacy_names(model->get_output_size());
    for (size_t i = 0; i < model->get_output_size(); i++) {
        ov::Output<const Node> const_output(model->output(i).get_node(), model->output(i).get_index());
        legacy_names[i] = ov::op::util::create_ie_output_name(const_output.get_node()->input_value(0));
        InferenceEngine::DataPtr output_info;
        // I don't remove rt info to have information in InputsInfo about pre-processing in legacy
        // ExecutableNetwork
        ov::legacy_convert::fill_output_info(const_output, output_info);
        OPENVINO_ASSERT(output_info);
        auto element_type = InferenceEngine::details::convertPrecision(output_info->getPrecision());
        if (element_type != model->output(i).get_element_type()) {
            preproc.output(i).tensor().set_element_type(element_type);
        }
        if (output_info->getLayout() != InferenceEngine::Layout::BLOCKED &&
            output_info->getLayout() != InferenceEngine::Layout::SCALAR) {
            std::stringstream stream;
            stream << output_info->getLayout();
            preproc.output(i).tensor().set_layout(ov::Layout{stream.str()});
        }

        if (const_output.get_partial_shape().is_static() && const_output.get_shape().size() == 4)
            preproc.output(i).model().set_layout("NCHW");
    }

    ov::pass::Manager manager(get_pass_config());
    auto rewrite = manager.register_pass<ov::pass::GraphRewrite>();
    if (!meanMap.empty()) {
        rewrite->add_matcher<ov::pass::AddMeanImage>(meanMap);
    }
    manager.run_passes(model);

    preproc.build();

    for (size_t i = 0; i < model->get_output_size(); i++) {
        ov::descriptor::set_ov_tensor_legacy_name(model->output(i).get_node()->input_value(0).get_tensor(),
                                                  legacy_names[i]);
    }

    return false;
}
