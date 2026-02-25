// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_utils.hpp"

#include <openvino/runtime/core.hpp>
#include <vector>

#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend;

// For compatibility purposes, need to remove when will be unused
const std::string ONNX_FE = "onnx";

namespace ov {
namespace frontend {
namespace onnx {
namespace tests {

const std::string ONNX_FE = ::ONNX_FE;

static FrontEnd::Ptr get_onnx_frontend(bool default_front_end = true) {
    static FrontEnd::Ptr _front_end = nullptr;

    FrontEnd::Ptr front_end = nullptr;

    if (default_front_end) {
        if (_front_end == nullptr) {
            auto fem = FrontEndManager();
            _front_end = fem.load_by_framework(ONNX_FE);
        }
        front_end = _front_end;
    } else {
        auto fem = FrontEndManager();
        front_end = fem.load_by_framework(ONNX_FE);
    }

    if (!front_end) {
        throw "ONNX FrontEnd is not initialized";
    }

    return front_end;
}

shared_ptr<Model> convert_model(const string& model_path, const ov::frontend::ConversionExtensionBase::Ptr& conv_ext) {
    auto front_end = get_onnx_frontend(conv_ext == nullptr);

    if (conv_ext) {
        front_end->add_extension(conv_ext);
    }

    auto full_path = FrontEndTestUtils::make_model_path(string(TEST_ONNX_MODELS_DIRNAME) + model_path);
    InputModel::Ptr input_model = front_end->load(full_path);
    if (!input_model) {
        throw "Input Model is not loaded";
    }

    shared_ptr<Model> model = front_end->convert(input_model);
    if (!model) {
        throw "Model is not converted";
    }

    return model;
}

shared_ptr<Model> convert_model(ifstream& model_stream) {
    auto front_end = get_onnx_frontend();

    InputModel::Ptr input_model = front_end->load(dynamic_cast<istream*>(&model_stream));
    if (!input_model) {
        throw "Input Model is not loaded";
    }

    shared_ptr<Model> model = front_end->convert(input_model);
    if (!model) {
        throw "Model is not converted";
    }

    return model;
}

shared_ptr<Model> convert_partially(const string& model_path) {
    auto front_end = get_onnx_frontend();

    auto full_path = FrontEndTestUtils::make_model_path(string(TEST_ONNX_MODELS_DIRNAME) + model_path);
    InputModel::Ptr input_model = front_end->load(full_path);
    if (!input_model) {
        throw "Input Model is not loaded";
    }

    shared_ptr<Model> model = front_end->convert_partially(input_model);
    if (!model) {
        throw "Model is not converted";
    }

    return model;
}

InputModel::Ptr load_model(const string& model_path, FrontEnd::Ptr* return_front_end) {
    auto front_end = get_onnx_frontend();

    auto full_path = FrontEndTestUtils::make_model_path(string(TEST_ONNX_MODELS_DIRNAME) + model_path);
    InputModel::Ptr input_model = front_end->load(full_path);
    if (!input_model) {
        throw "Input Model is not loaded";
    }

    if (return_front_end != nullptr) {
        *return_front_end = front_end;
    }

    return input_model;
}

InputModel::Ptr load_model(const wstring& model_path, FrontEnd::Ptr* return_front_end) {
    auto front_end = get_onnx_frontend();

    auto full_path =
        FrontEndTestUtils::make_model_path(string(TEST_ONNX_MODELS_DIRNAME) + ov::util::wstring_to_string(model_path));
    InputModel::Ptr input_model = front_end->load(ov::util::string_to_wstring(full_path));
    if (!input_model) {
        throw "Input Model is not loaded";
    }

    if (return_front_end != nullptr) {
        *return_front_end = front_end;
    }

    return input_model;
}

std::string onnx_backend_manifest(const std::string& manifest) {
    return ov::util::path_join({ov::test::utils::getExecutableDirectory(), manifest}).string();
}

}  // namespace tests
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
