// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <list>
#include <sstream>
#include <memory>
#include <map>
#include <vector>
#include "parsers.h"
#include "pugixml.hpp"
#include <fstream>
#include <stdio.h>
#include "cpp/ie_cnn_network.h"
#include <gtest/gtest.h>
#include "ie_icnn_network_stats.hpp"
#include "xml_helper.hpp"

#include <pugixml.hpp>

namespace testing {

class XMLHelper::impl {
public:
    std::unique_ptr<pugi::xml_node> _root;
    std::unique_ptr<pugi::xml_document> _doc;
};

XMLHelper::XMLHelper(InferenceEngine::details::IFormatParser* p) {
    parser.reset(p);
    _impl = std::make_shared<impl>();
    _impl->_doc.reset(new pugi::xml_document());
    _impl->_root.reset(new pugi::xml_node());
}
 
void XMLHelper::loadContent(const std::string &fileContent) {
    auto res = _impl->_doc->load_string(fileContent.c_str());
    EXPECT_EQ(pugi::status_ok, res.status) << res.description() << " at offset " << res.offset;
    *_impl->_root = _impl->_doc->document_element();
}

void XMLHelper::loadFile(const std::string &filename) {
    auto res = _impl->_doc->load_file(filename.c_str());
    EXPECT_EQ(pugi::status_ok, res.status) << res.description() << " at offset " << res.offset;
    *_impl->_root = _impl->_doc->document_element();
}

void XMLHelper::parse() {
    parser->Parse(*_impl->_root);
}

InferenceEngine::details::CNNNetworkImplPtr XMLHelper::parseWithReturningNetwork() {
    return parser->Parse(*_impl->_root);
}

void XMLHelper::setWeights(const InferenceEngine::TBlob<uint8_t>::Ptr &weights) {
    parser->SetWeights(weights);
}

std::string XMLHelper::readFileContent(const std::string & filePath) {
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

std::string XMLHelper::getXmlPath(const std::string & filePath){
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

InferenceEngine::NetworkStatsMap loadStatisticFromFile(const std::string& xmlPath) {
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
