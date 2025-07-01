#include "utils/model.hpp"
#include "utils/error.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace utils {

std::vector<std::filesystem::path> ModelHelper::s_tempFiles;

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
    for (const auto& path : s_tempFiles) {
        if (std::filesystem::exists(path)) {
            std::filesystem::remove(path);
        }
    }
    s_tempFiles.clear();
}

void ModelHelper::prepareModel() {
    bool need_save = false;
    need_save = ensureNamedTensors() || need_save;

    if (need_save) {
        saveTempModel();
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

bool ModelHelper::saveTempModel() {
    std::filesystem::path tmp_xml = std::filesystem::temp_directory_path() / ("tmp_" + m_xmlPath.filename().string());
    std::filesystem::path tmp_bin = tmp_xml;
    tmp_bin.replace_extension(".bin");

    ov::pass::Serialize(tmp_xml.string(), "").run_on_model(m_model);

    if (std::filesystem::exists(tmp_bin)) {
        std::filesystem::remove(tmp_bin);
    }
    if (std::filesystem::exists(tmp_xml)) {
        m_xmlPath = tmp_xml;
        s_tempFiles.push_back(tmp_xml);

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
