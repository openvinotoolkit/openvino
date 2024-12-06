//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simulation/layers_reader.hpp"

#include <opencv2/core.hpp>  // CV_*
#include <openvino/openvino.hpp>

#include "utils/error.hpp"

#include <fstream>

class OpenVINOLayersReader::Impl {
public:
    InOutLayers readLayers(const OpenVINOParams& params, const bool use_results_names);

private:
    InOutLayers readFromBlob(const std::string& blob, const std::string& device,
                             const std::map<std::string, std::string>& config);

    InOutLayers readFromModel(const std::string& xml, const std::string& bin, const OpenVINOParams& params,
                              const bool use_results_names);

private:
    ov::Core m_core;
};

OpenVINOLayersReader::OpenVINOLayersReader(): m_impl(new OpenVINOLayersReader::Impl{}) {
}

static ov::element::Type toElementType(int cvdepth) {
    switch (cvdepth) {
    case CV_8U:
        return ov::element::u8;
    case CV_32S:
        return ov::element::i32;
    case CV_32F:
        return ov::element::f32;
    case CV_16F:
        return ov::element::f16;
    }
    throw std::logic_error("Failed to convert opencv depth to ov::element::Type");
}

static std::vector<int> toDims(const std::vector<size_t>& sz_vec) {
    std::vector<int> result;
    result.reserve(sz_vec.size());
    for (auto sz : sz_vec) {
        // FIXME: Probably requires some check...
        result.push_back(static_cast<int>(sz));
    }
    return result;
}

static int toPrecision(ov::element::Type prec) {
    switch (prec) {
    case ov::element::u8:
        return CV_8U;
    case ov::element::i32:
        return CV_32S;
    case ov::element::f32:
        return CV_32F;
    case ov::element::f16:
        return CV_16F;
    case ov::element::i64:
        return CV_32S;
    }
    throw std::logic_error("Unsupported OV precision");
}

template <typename InfoVec>
std::vector<LayerInfo> ovToLayersInfo(const InfoVec& vec) {
    std::vector<LayerInfo> layers;
    layers.reserve(vec.size());
    std::transform(vec.begin(), vec.end(), std::back_inserter(layers), [](const auto& node) {
        return LayerInfo{node.get_any_name(), toDims(node.get_shape()), toPrecision(node.get_element_type())};
    });
    return layers;
};

static void cfgInputPreproc(ov::preprocess::PrePostProcessor& ppp, const std::shared_ptr<ov::Model>& model,
                            const AttrMap<int>& input_precision, const AttrMap<std::string>& input_layout,
                            const AttrMap<std::string>& input_model_layout) {
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        auto& ii = ppp.input(name);

        const auto ip = lookUp(input_precision, name);
        if (ip.has_value()) {
            ii.tensor().set_element_type(toElementType(*ip));
        }

        const auto il = lookUp(input_layout, name);
        if (il.has_value()) {
            ii.tensor().set_layout(ov::Layout(*il));
        }

        const auto iml = lookUp(input_model_layout, name);
        if (iml.has_value()) {
            ii.model().set_layout(ov::Layout(*iml));
        }
    }
}

static void cfgOutputPostproc(ov::preprocess::PrePostProcessor& ppp, const std::shared_ptr<ov::Model>& model,
                              const AttrMap<int>& output_precision, const AttrMap<std::string>& output_layout,
                              const AttrMap<std::string> output_model_layout) {
    for (const auto& output : model->outputs()) {
        const auto& name = output.get_any_name();
        auto& oi = ppp.output(name);

        const auto op = lookUp(output_precision, name);
        if (op.has_value()) {
            oi.tensor().set_element_type(toElementType(*op));
        }

        const auto ol = lookUp(output_layout, name);
        if (ol.has_value()) {
            oi.tensor().set_layout(ov::Layout(*ol));
        }

        const auto oml = lookUp(output_model_layout, name);
        if (oml.has_value()) {
            oi.model().set_layout(ov::Layout(*oml));
        }
    }
}

static std::vector<std::string> extractLayerNames(const std::vector<ov::Output<ov::Node>>& nodes) {
    std::vector<std::string> names;
    std::transform(nodes.begin(), nodes.end(), std::back_inserter(names), [](const auto& node) {
        return node.get_any_name();
    });
    return names;
}

InOutLayers OpenVINOLayersReader::Impl::readFromModel(const std::string& model_path, const std::string& bin_path,
                                                      const OpenVINOParams& params, const bool use_results_names) {
    auto model = m_core.read_model(model_path, bin_path);
    {
        ov::preprocess::PrePostProcessor ppp(model);

        const auto& input_names = extractLayerNames(model->inputs());
        const auto ip_map = unpackLayerAttr(params.input_precision, input_names, "input precision");
        const auto il_map = unpackLayerAttr(params.input_layout, input_names, "input layout");
        const auto iml_map = unpackLayerAttr(params.input_model_layout, input_names, "input model layout");
        cfgInputPreproc(ppp, model, ip_map, il_map, iml_map);

        const auto& output_names = extractLayerNames(model->outputs());
        const auto op_map = unpackLayerAttr(params.output_precision, output_names, "output precision");
        const auto ol_map = unpackLayerAttr(params.output_layout, output_names, "output layout");
        const auto oml_map = unpackLayerAttr(params.output_model_layout, output_names, "output model layout");
        cfgOutputPostproc(ppp, model, op_map, ol_map, oml_map);

        model = ppp.build();
    }

    auto input_layers = ovToLayersInfo(model->inputs());
    auto output_layers = ovToLayersInfo(model->outputs());

    // FIXME: UGLY WA in order to use layer names obtained by OV reader in ONNXRT.
    // Ideally there should be corresponding ONNXRT reader instead!!!
    // Result nodes friendly names preserve the names from original model,
    // so the could be used in different framework (not only OpenVINO)
    if (use_results_names) {
        const auto& results = model->get_results();
        for (int i = 0; i < results.size(); ++i) {
            auto result_name = results[i]->get_friendly_name();
            // This suffix is hardcoded at the OpenVINO side
            const std::string suffix = "/sink_port_0";
            const auto kSuffixStartPos = result_name.length() - suffix.length();
            // Check that suffix is still presented at the OpenVINO side
            ASSERT(result_name.substr(kSuffixStartPos) == suffix);
            // Drop the suffix as it's not needed and update the name
            result_name = result_name.substr(0, kSuffixStartPos);
            output_layers[i].name = result_name;
        }
    }

    return {std::move(input_layers), std::move(output_layers)};
}

InOutLayers OpenVINOLayersReader::Impl::readFromBlob(const std::string& blob, const std::string& device,
                                                     const std::map<std::string, std::string>& config) {
    std::ifstream file(blob, std::ios_base::in | std::ios_base::binary);
    if (!file.is_open()) {
        THROW_ERROR("Failed to import model from: " << blob);
    }

    auto compiled_model = m_core.import_model(file, device, {config.begin(), config.end()});

    auto input_layers = ovToLayersInfo(compiled_model.inputs());
    auto output_layers = ovToLayersInfo(compiled_model.outputs());

    return {std::move(input_layers), std::move(output_layers)};
}

InOutLayers OpenVINOLayersReader::Impl::readLayers(const OpenVINOParams& params, const bool use_results_names) {
    if (std::holds_alternative<OpenVINOParams::ModelPath>(params.path)) {
        const auto& path = std::get<OpenVINOParams::ModelPath>(params.path);
        return readFromModel(path.model, path.bin, params, use_results_names);
    }
    ASSERT(std::holds_alternative<OpenVINOParams::BlobPath>(params.path));
    // NB: use_results_names is WA for reading layer names for the further usage in ONNXRT
    // since ONNXRT is always ModelPath case (*.onnx format), no need to handle this for *.blob's
    ASSERT(!use_results_names);
    const auto& path = std::get<OpenVINOParams::BlobPath>(params.path);
    return readFromBlob(path.blob, params.device, params.config);
}

InOutLayers OpenVINOLayersReader::readLayers(const OpenVINOParams& params, const bool use_results_names) {
    return m_impl->readLayers(params, use_results_names);
}
