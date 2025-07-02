#pragma once

#include <vector>
#include <string>

#include <openvino/openvino.hpp>
#include "scenario/inference.hpp"

namespace utils {

std::string make_default_tensor_name(const ov::Output<const ov::Node>& output);

void cleanupTempFiles();

class ModelHelper {
public:
    ModelHelper(const OpenVINOParams& params);
    ~ModelHelper() = default;

    static void cleanupAllTempFiles();

    void prepareModel();

    const std::string getXmlPath() const;
    const std::string getBinPath() const;

private:
    bool ensureNamedTensors();
    void clampOutputs();

    bool saveTempModel();

private:
    std::shared_ptr<ov::Model> m_model;
    const OpenVINOParams& m_params;

    std::filesystem::path m_xmlPath;
    std::filesystem::path m_binPath;

    static std::vector<std::filesystem::path> s_tempFiles;

    bool m_tempModelSaved = false;
};

} // namespace utils
