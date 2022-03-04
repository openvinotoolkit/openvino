// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

namespace TestDataHelpers {

static const char kPathSeparator =
#if defined _WIN32 || defined __CYGWIN__
        '\\';
#else
        '/';
#endif

std::string getModelPathNonFatal() noexcept {
    if (const auto envVar = std::getenv("MODELS_PATH")) {
        return envVar;
    }

#ifdef MODELS_PATH
    return MODELS_PATH;
#else
    return "";
#endif
}

std::string get_models_path() {
    return getModelPathNonFatal() + kPathSeparator + std::string("models");
};

std::string get_data_path() {
    if (const auto envVar = std::getenv("DATA_PATH")) {
        return envVar;
    }

#ifdef DATA_PATH
    return DATA_PATH;
#else
    return "";
#endif
}

std::string generate_model_path(std::string dir, std::string filename) {
    return get_models_path() + kPathSeparator + dir + kPathSeparator + filename;
}

std::string generate_image_path(std::string dir, std::string filename) {
    return get_data_path() + kPathSeparator + "validation_set" + kPathSeparator + dir + kPathSeparator + filename;
}

std::string generate_ieclass_xml_path(std::string filename) {
    return getModelPathNonFatal() + kPathSeparator + "ie_class" + kPathSeparator + filename;
}
} // namespace TestDataHelpers
