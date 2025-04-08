// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <gtest/gtest.h>
#include <stdio.h>

#include "common_test_utils/file_utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/util/file_util.hpp"

inline std::string get_extension_path() {
    return ov::util::make_plugin_library_name<char>(ov::test::utils::getExecutableDirectory(),
                                                    std::string("openvino_template_extension") + OV_BUILD_POSTFIX);
}

inline std::wstring get_extension_wdir() {
    std::wstring dir = ov::util::string_to_wstring(ov::test::utils::getExecutableDirectory());
    dir.push_back(ov::util::FileTraits<wchar_t>::file_separator);
    dir += ov::util::string_to_wstring("晚安_путь_к_файлу");
    dir.push_back(ov::util::FileTraits<wchar_t>::file_separator);
    ov::util::create_directory_recursive(dir);
    return dir;
}

TEST(extension, load_extension) {
    EXPECT_NO_THROW(ov::detail::load_extensions(get_extension_path()));
}

#if defined(_WIN32)
TEST(extension, load_extension_wstring) {
    std::wstring wdir = get_extension_wdir();
    std::wstring wdir_ext_path = wdir + ov::util::string_to_wstring(ov::util::make_plugin_library_name<char>(
                                            "",
                                            std::string("openvino_template_extension") + OV_BUILD_POSTFIX));
    _wrename(ov::util::string_to_wstring(get_extension_path()).c_str(), wdir_ext_path.c_str());
    EXPECT_NO_THROW(ov::detail::load_extensions(wdir_ext_path));
    _wrename(wdir_ext_path.c_str(), ov::util::string_to_wstring(get_extension_path()).c_str());
    _wrmdir(wdir.c_str());
}
#endif

TEST(extension, load_extension_and_cast) {
    std::vector<ov::Extension::Ptr> so_extensions = ov::detail::load_extensions(get_extension_path());
    ASSERT_LE(1, so_extensions.size());
    std::vector<ov::Extension::Ptr> extensions;
    std::vector<std::shared_ptr<void>> so;
    for (const auto& ext : so_extensions) {
        if (auto so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(ext)) {
            extensions.emplace_back(so_ext->extension());
            so.emplace_back(so_ext->shared_object());
        }
    }
    so_extensions.clear();
    EXPECT_LE(1, extensions.size());
    EXPECT_NE(nullptr, dynamic_cast<ov::BaseOpExtension*>(extensions[0].get()));
    EXPECT_NE(nullptr, std::dynamic_pointer_cast<ov::BaseOpExtension>(extensions[0]));
    extensions.clear();
}

namespace {
class DummyAdapter : public ov::AttributeVisitor {
public:
    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {}
};
}  // namespace

TEST(extension, create_model_from_extension) {
    std::vector<ov::Extension::Ptr> so_extensions = ov::detail::load_extensions(get_extension_path());
    ASSERT_LE(1, so_extensions.size());
    std::vector<ov::Extension::Ptr> extensions;
    std::vector<std::shared_ptr<void>> so;
    for (const auto& ext : so_extensions) {
        if (auto so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(ext)) {
            extensions.emplace_back(so_ext->extension());
            so.emplace_back(so_ext->shared_object());
        }
    }
    so_extensions.clear();
    EXPECT_LE(1, extensions.size());
    auto op_extension = std::dynamic_pointer_cast<ov::BaseOpExtension>(extensions[0]);
    EXPECT_NE(nullptr, op_extension);
    {
        // Create model to check evaluate for custom operation
        std::shared_ptr<ov::Model> model;
        {
            auto parameter = std::make_shared<ov::opset9::Parameter>(ov::element::i32, ov::Shape{1, 2, 2, 2});

            DummyAdapter visitor;

            auto outputs = op_extension->create(ov::OutputVector{parameter}, visitor);

            EXPECT_EQ(1, outputs.size());
            EXPECT_NE(nullptr, outputs[0].get_node());
            const std::string ref_name = "Identity";
            EXPECT_EQ(ref_name, outputs[0].get_node()->get_type_info().name);
            auto result = std::make_shared<ov::opset9::Result>(outputs[0]);
            model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
        }

        auto fill_tensor = [](ov::Tensor& tensor) {
            int32_t* data = tensor.data<int32_t>();
            for (size_t i = 0; i < tensor.get_size(); i++)
                data[i] = static_cast<int32_t>(i);
        };

        ov::TensorVector inputs;
        inputs.emplace_back(ov::Tensor(ov::element::i32, ov::Shape{1, 2, 2, 2}));
        fill_tensor(*inputs.begin());
        ov::TensorVector outputs;
        outputs.emplace_back(ov::Tensor(ov::element::i32, ov::Shape{1, 2, 2, 2}));
        EXPECT_NE(std::memcmp(inputs.begin()->data(), outputs.begin()->data(), inputs.begin()->get_byte_size()), 0);
        model->evaluate(outputs, inputs);
        EXPECT_EQ(std::memcmp(inputs.begin()->data(), outputs.begin()->data(), inputs.begin()->get_byte_size()), 0);
    }
    extensions.clear();
}
