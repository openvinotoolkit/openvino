// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow_lite/frontend.hpp"

#include "graph_iterator_flatbuffer.hpp"
#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/util/common_util.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow_lite;

FrontEnd::FrontEnd() {
    m_op_translators = tensorflow::op::get_supported_lite_ops();
}

/// \brief Check if FrontEndTensorflow can recognize model from given parts
bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() != 1)
        return false;

    if (variants[0].is<std::string>()) {
        std::string suffix = ".tflite";
        std::string model_path = variants[0].as<std::string>();
        if (ov::util::ends_with(model_path, suffix.c_str())) {
            return true;
        }
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        std::wstring suffix = L".tflite";
        std::wstring model_path = variants[0].as<std::wstring>();
        if (ov::util::ends_with(model_path, suffix)) {
            return true;
        }
    }
#endif
    return false;
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() == 1) {
        if (variants[0].is<std::string>()) {
            std::string suffix = ".tflite";
            std::string model_path = variants[0].as<std::string>();
            if (ov::util::ends_with(model_path, suffix.c_str())) {
                return std::make_shared<tensorflow_lite::InputModel>(std::make_shared<GraphIteratorFlatBuffer>(model_path),
                                                                     m_telemetry);
            }
        }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        else if (variants[0].is<std::wstring>()) {
            std::wstring suffix = L".tflite";
            std::wstring model_path = variants[0].as<std::wstring>();
            if (ov::util::ends_with(model_path, suffix)) {
                return std::make_shared<InputModel>(
                    std::make_shared<::ov::frontend::tensorflow::GraphIteratorFlatBuffer>(model_path),
                    m_telemetry);
            }
        }
#endif
    }
    return nullptr;
}

std::shared_ptr<ov::Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr &model) const {
    auto model_tf = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(model_tf != nullptr, "Invalid input model");
    std::shared_ptr<ov::Model> f = std::make_shared<ov::Model>(ResultVector{}, ParameterVector{}, "Tensorflow Lite IR");
    // translate inputs
    model->get_inputs();
    // translate intermediates

    // translate output
    std::cout << "FrontEnd::convert" << std::endl;
    return f;
}

void FrontEnd::translate_graph(const InputModel::Ptr &model, const std::string &model_name, bool fail_fast,
                               bool no_conversion, std::shared_ptr<ov::Model> &ng_function) const {
    const auto& model_lite = std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::InputModel>(model);
    FRONT_END_GENERAL_CHECK(model_lite, "nullptr for InputModel is given for translation into OV Model");
    model_lite->get_inputs();
}
