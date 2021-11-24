// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <frontend_manager/frontend_manager.hpp>
#include <tensorflow_frontend/extension.hpp>
#include <tensorflow_frontend/frontend.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace ngraph;
using namespace ov::frontend;

inline std::string get_extension_path() {
    return ov::util::make_plugin_library_name<char>({}, std::string("tf_conversion_extensions") + IE_BUILD_POSTFIX);
}

TEST(LoadExtensions, DynamicLibrary) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_FE));
    ASSERT_NE(frontEnd, nullptr);

    // Add SOExtension
    frontEnd->add_extension(get_extension_path());
}

TEST(LoadExtensions, Lambda) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_FE));
    ASSERT_NE(frontEnd, nullptr);

    frontEnd->add_extension(std::make_shared<ov::frontend::ConversionExtension>(
        "Stub",
        [](const ov::frontend::tf::NodeContext& node) -> ov::OutputVector {
            return OutputVector();
        }));
}