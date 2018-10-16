// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <list>
#include <sstream>
#include <memory>
#include "inference_engine/parsers.h"
#include "pugixml.hpp"
#include <fstream>
#include <stdio.h>
#include "cpp/ie_cnn_network.h"

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
}
