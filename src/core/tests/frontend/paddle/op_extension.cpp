// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_extension.hpp"

#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/paddle/extension/op.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "openvino/runtime/core.hpp"
#include "paddle_utils.hpp"
#include "so_extension.hpp"
#include "utils.hpp"

using namespace ov::frontend;

using ONNXOpExtensionTest = FrontEndOpExtensionTest;

static const std::string translator_name = "Relu";

class Relu1 : public Relu {
public:
    OPENVINO_OP("relu");
    OPENVINO_FRAMEWORK_MAP_PADDLE({"X"}, {"Out"})
};

TEST(FrontEndOpExtensionTest, opextension_relu) {
    ov::Core core;
    const auto extensions = std::vector<std::shared_ptr<ov::Extension>>{std::make_shared<ov::OpExtension<Relu1>>()};
    core.add_extension(extensions);
    std::string m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    std::string m_modelName = "relu/relu.pdmodel";
    auto model = core.read_model(FrontEndTestUtils::make_model_path(m_modelsPath + m_modelName));
    bool has_relu = false;
    for (const auto& op : model->get_ops()) {
        std::string name = op->get_type_info().name;
        std::string version = op->get_type_info().version_id;
        if (name == "relu" && version == "extension")
            has_relu = true;
    }
    EXPECT_TRUE(has_relu);
}
