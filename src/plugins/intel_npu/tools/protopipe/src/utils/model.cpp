// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/model.hpp"
#include "utils/error.hpp"
#include "utils/logger.hpp"

#include <filesystem>

#include <openvino/opsets/opset1.hpp>
#include <opencv2/core.hpp> // CV_*

namespace utils {

std::set<std::filesystem::path> ModelHelper::s_tempFiles;

static int get_output_cv_type(const std::string& name, const LayerVariantAttr<int>& output_precision) {
    if (std::holds_alternative<int>(output_precision)) {
        return std::get<int>(output_precision);
    } else if (std::holds_alternative<AttrMap<int>>(output_precision)) {
        const auto& map = std::get<AttrMap<int>>(output_precision);
        auto it = map.find(name);
        if (it != map.end()) {
            return it->second;
        }
    }
    return -1;
}

static inline std::pair<double, double> get_cv_type_range(int cv_type) {
    switch (cv_type) {
        case CV_8U:  return {0.0, 255.0};
        case CV_32S: return {static_cast<double>(INT32_MIN), static_cast<double>(INT32_MAX)};
        case CV_32F: return {-FLT_MAX, FLT_MAX};
        case CV_16F: return {-65504.0, 65504.0};
        default:     return {0.0, 0.0};
    }
}

static inline const char* cv_type_to_str(int cv_type) {
    switch (cv_type) {
        case CV_8U:  return "U8";
        case CV_32S: return "I32";
        case CV_32F: return "F32";
        case CV_16F: return "F16";
        default:     return "UNKNOWN";
    }
}

ModelHelper::ModelHelper(const OpenVINOParams& params) : m_params(params) {
    ov::Core core;
    if (std::holds_alternative<OpenVINOParams::ModelPath>(m_params.path)) {
        const auto& model_path = std::get<OpenVINOParams::ModelPath>(m_params.path);
        m_model = core.read_model(model_path.model, model_path.bin);
        m_xmlPath = model_path.model;
        m_binPath = model_path.bin;
    } else {
        THROW_ERROR("Unsupported path type for OpenVINO model.");
    }
}

void ModelHelper::cleanupAllTempFiles() {
    if (s_tempFiles.empty()) {
        return ;
    }

    LOG_DEBUG() << "Deleting all temp files." << std::endl;
    for (const auto& path : s_tempFiles) {
        if (std::filesystem::exists(path)) {
            if (!std::filesystem::remove(path)) {
                // TODO: warn the user the temp file couldn't be removed
            }
        }
    }
    s_tempFiles.clear();
}

void ModelHelper::prepareModel() {
    bool need_save = ensureNamedTensors();

    if (m_params.clamp_outputs) {
        need_save = clampOutputs() || need_save;
    }

    if (need_save) {
        if (!saveTempModel()) {
            THROW_ERROR("Failed to save the temporary model.");
        }
    }
}

bool ModelHelper::ensureNamedTensors() {
    bool need_save = false;
    for (auto& input : m_model->inputs()) {
        if (input.get_names().empty()) {
            need_save = true;
            input.set_names({make_default_tensor_name(input)});
        }
    }
    for (auto& output : m_model->outputs()) {
        if (output.get_names().empty()) {
            need_save = true;
            output.set_names({make_default_tensor_name(output)});
        }
    }

    return need_save;
}

bool ModelHelper::clampOutputs() {
    bool need_save = false;
    auto results = m_model->get_results();
    for (const auto& result : results) {
        const auto& output = result->input_value(0);
        const auto& name = output.get_any_name();

        auto output_cv_type = get_output_cv_type(name, m_params.output_precision);
        if (output_cv_type == -1) {
            continue;
        }
        auto [min_val, max_val] = get_cv_type_range(output_cv_type);

        LOG_DEBUG() << "Clamping output '" << name << "' to " << cv_type_to_str(output_cv_type) << std::endl;

        auto clamp = std::make_shared<ov::opset1::Clamp>(output, min_val, max_val);
        clamp->set_friendly_name(result->get_friendly_name() + "_clamped");
        result->input(0).replace_source_output(clamp->output(0));
        need_save = true;
    }

    return need_save;
}

bool ModelHelper::saveTempModel() {
    std::filesystem::path tmp_xml = std::filesystem::temp_directory_path() / ("tmp_" + m_xmlPath.filename().string());
    std::filesystem::path tmp_bin = tmp_xml;
    tmp_bin.replace_extension(".bin");

    LOG_DEBUG() << "Saving temp model to path: " << tmp_xml.string() << std::endl;

    ov::pass::Serialize(tmp_xml.string(), "").run_on_model(m_model);

    if (std::filesystem::exists(tmp_bin)) {
        if (!std::filesystem::remove(tmp_bin)) {
            // TODO: warn the user tmp_bin couldn't be removed
        }
    }
    if (std::filesystem::exists(tmp_xml)) {
        m_xmlPath = tmp_xml;
        s_tempFiles.insert(tmp_xml);

        return true;
    }
    return false;
}

const std::string ModelHelper::getXmlPath() const {
    return m_xmlPath.string();
}

const std::string ModelHelper::getBinPath() const {
    return m_binPath.string();
}

void cleanupTempFiles() {
    ModelHelper::cleanupAllTempFiles();
}

std::string make_default_tensor_name(const ov::Output<const ov::Node>& output) {
    auto default_name = output.get_node()->get_friendly_name();
    if (output.get_node()->get_output_size() > 1) {
        default_name += ':' + std::to_string(output.get_index());
    }
    return default_name;
}

} // namespace utils
