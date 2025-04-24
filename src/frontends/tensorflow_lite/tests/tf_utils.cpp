// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tf_utils.hpp"

#include <openvino/runtime/core.hpp>
#include <vector>

#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend;

// For compatibility purposes, need to remove when will be unused
const std::string TF_LITE_FE = "tflite";

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace tests {

const std::string TF_LITE_FE = ::TF_LITE_FE;

static FrontEnd::Ptr get_tflite_frontend(bool default_front_end = true) {
    static FrontEnd::Ptr _front_end = nullptr;

    FrontEnd::Ptr front_end = nullptr;

    if (default_front_end) {
        if (_front_end == nullptr) {
            auto fem = FrontEndManager();
            _front_end = fem.load_by_framework(TF_LITE_FE);
        }
        front_end = _front_end;
    } else {
        auto fem = FrontEndManager();
        front_end = fem.load_by_framework(TF_LITE_FE);
    }

    if (!front_end) {
        throw "TensorFlow Lite FrontEnd is not initialized";
    }

    return front_end;
}

shared_ptr<Model> convert_model(const string& model_path, const ov::frontend::ConversionExtensionBase::Ptr& conv_ext) {
    auto front_end = get_tflite_frontend(conv_ext == nullptr);

    if (conv_ext) {
        front_end->add_extension(conv_ext);
    }

    auto full_path = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) + model_path);
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

}  // namespace tests
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
