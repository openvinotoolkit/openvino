// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <list>
#include <sstream>
#include <memory>
#include <map>
#include <vector>
#include "inference_engine/parsers.h"
#include "pugixml.hpp"
#include <fstream>
#include <stdio.h>
#include "cpp/ie_cnn_network.h"
#include <gtest/gtest.h>
#include "ie_icnn_network_stats.hpp"

namespace testing {
	class XMLHelper {
    public:
        XMLHelper(InferenceEngine::details::IFormatParser* p) {
            parser.reset(p);
            _doc.reset(new pugi::xml_document());
            _root.reset(new pugi::xml_node());
        }
        void loadContent(const std::string &fileContent) {
            auto res = _doc->load_string(fileContent.c_str());
            EXPECT_EQ(pugi::status_ok, res.status) << res.description() << " at offset " << res.offset;
            *_root = _doc->document_element();
        }

        void loadFile(const std::string &filename) {
            auto res = _doc->load_file(filename.c_str());
            EXPECT_EQ(pugi::status_ok, res.status) << res.description() << " at offset " << res.offset;
            *_root = _doc->document_element();
        }

        void parse() {
            parser->Parse(*_root);
        }

        InferenceEngine::details::CNNNetworkImplPtr parseWithReturningNetwork() {
            return parser->Parse(*_root);
        }

        void setWeights(const InferenceEngine::TBlob<uint8_t>::Ptr &weights) {
            parser->SetWeights(weights);
        }

        std::string readFileContent(const std::string & filePath) {
            const auto openFlags = std::ios_base::ate | std::ios_base::binary;
            std::ifstream fp (getXmlPath(filePath), openFlags);
            EXPECT_TRUE(fp.is_open());

            std::streamsize size = fp.tellg();
            EXPECT_GE( size , 1) << "file is empty: " << filePath;

            std::string str;

            str.reserve((size_t)size);
            fp.seekg(0, std::ios::beg);

            str.assign((std::istreambuf_iterator<char>(fp)),
                       std::istreambuf_iterator<char>());
            return str;
        }

    private:
        std::string getXmlPath(const std::string & filePath){
            std::string xmlPath = filePath;
            const auto openFlags = std::ios_base::ate | std::ios_base::binary;
            std::ifstream fp (xmlPath, openFlags);
            //TODO: Dueto multi directory build systems, and single directory build system
            //, it is usualy a problem to deal with relative paths.
            if (!fp.is_open()) {
                fp.open(getParentDir(xmlPath), openFlags);
                EXPECT_TRUE(fp.is_open())
                << "cannot open file " << xmlPath <<" or " << getParentDir(xmlPath);
                fp.close();
                xmlPath = getParentDir(xmlPath);
            }
            return xmlPath;
        }

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
        std::unique_ptr<pugi::xml_node> _root;
        std::unique_ptr<pugi::xml_document> _doc;
	};

inline InferenceEngine::NetworkStatsMap loadStatisticFromFile(const std::string& xmlPath) {
    auto splitParseCommas = [&](const std::string& s) ->std::vector<float> {
        std::vector<float> res;
        std::stringstream ss(s);

        float val;

        while (ss >> val) {
            res.push_back(val);

            if (ss.peek() == ',')
                ss.ignore();
        }

        return res;
    };

    InferenceEngine::NetworkStatsMap newNetNodesStats;

    pugi::xml_document doc;

    pugi::xml_parse_result pr = doc.load_file(xmlPath.c_str());


    if (!pr) {
        THROW_IE_EXCEPTION << "Can't load stat file " << xmlPath;
    }

    auto stats = doc.child("stats");
    auto layers = stats.child("layers");

    InferenceEngine::NetworkNodeStatsPtr nodeStats;
    size_t offset;
    size_t size;
    size_t count;

    for (auto layer : layers.children("layer")) {
        nodeStats = InferenceEngine::NetworkNodeStatsPtr(new InferenceEngine::NetworkNodeStats());

        std::string name = layer.child("name").text().get();

        newNetNodesStats[name] = nodeStats;

        nodeStats->_minOutputs = splitParseCommas(layer.child("min").text().get());
        nodeStats->_maxOutputs = splitParseCommas(layer.child("max").text().get());
    }

    return newNetNodesStats;
}

}
