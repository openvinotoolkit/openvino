#include "utils/model.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace utils {

std::string make_default_tensor_name(const ov::Output<const ov::Node>& output) {
    auto default_name = output.get_node()->get_friendly_name();
    if (output.get_node()->get_output_size() > 1) {
        default_name += ':' + std::to_string(output.get_index());
    }
    return default_name;
}

OpenVINOParams::ModelPath ensureNamedModel(const OpenVINOParams::ModelPath& modelPath) {
    ov::Core core;
    auto model = core.read_model(modelPath.model, modelPath.bin);
    bool need_save = false;
    for (auto& input : model->inputs()) {
        if (input.get_names().empty()) {
            need_save = true;
            input.set_names({make_default_tensor_name(input)});
        }
    }
    for (auto& output : model->outputs()) {
        if (output.get_names().empty()) {
            need_save = true;
            output.set_names({make_default_tensor_name(output)});
        }
    }

    if (!need_save) {
        return modelPath;
    }

    std::filesystem::path tmp_xml = std::filesystem::temp_directory_path() / ("named_" + std::filesystem::path(modelPath.model).filename().string());
    std::filesystem::path tmp_bin = tmp_xml;
    tmp_bin.replace_extension(".bin");

    ov::pass::Serialize(tmp_xml.string(), "").run_on_model(model);

    if (std::filesystem::exists(tmp_bin)) {
        std::filesystem::remove(tmp_bin);
    }

    return OpenVINOParams::ModelPath{tmp_xml.string(), modelPath.bin};
}

} // namespace utils
