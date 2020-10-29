// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <list>
#include <sstream>
#include <memory>
#include <map>
#include <vector>
#include <parsers.h>
#include <fstream>
#include <stdio.h>
#include "cpp/ie_cnn_network.h"
#include <gtest/gtest.h>
#include <ie_icnn_network_stats.hpp>

namespace testing {

class XMLHelper {
public:
    explicit XMLHelper(InferenceEngine::details::IFormatParser* p);
    
    void loadContent(const std::string &fileContent);
    void loadFile(const std::string &filename);
    void parse();

    InferenceEngine::details::CNNNetworkImplPtr parseWithReturningNetwork();
    void setWeights(const InferenceEngine::TBlob<uint8_t>::Ptr &weights);

    std::string readFileContent(const std::string & filePath);

private:
    std::string getXmlPath(const std::string & filePath);

    const char kPathSeparator =
#if defined _WIN32 || defined __CYGWIN__
    '\\';
#else
    '/';
#endif
    const std::string parentDir = std::string("..") + kPathSeparator;

    std::string getParentDir(std::string currentFile) const {
        return parentDir + currentFile;
    }

    std::unique_ptr<InferenceEngine::details::IFormatParser> parser;
    std::vector<std::string> _classes;

    // hide pugixml from public dependencies
    class impl;
    std::shared_ptr<impl> _impl;
};

InferenceEngine::NetworkStatsMap loadStatisticFromFile(const std::string& xmlPath);

}
