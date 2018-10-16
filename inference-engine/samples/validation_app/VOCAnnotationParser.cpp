/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <sstream>
#include <exception>
#include <string>
#include <list>

#include "VOCAnnotationParser.hpp"

#include "user_exception.hpp"

std::string VOCAnnotationParser::parseString(const pugi::xml_node& node, const std::string& def) {
    if (node && node.child_value()) {
        return node.child_value();
    } else {
        return def;
    }
}

int VOCAnnotationParser::parseInt(const pugi::xml_node& node, const int def) {
    if (!node) return def;
    std::string val = parseString(node);
    try {
        return std::stoi(val);
    } catch (const std::invalid_argument& e) {
        THROW_USER_EXCEPTION(1) << "Can't convert node <" << node.name()
            << "> value \"" << val << "\" to integer";
    }
}

bool VOCAnnotationParser::parseBool(const pugi::xml_node& node, bool def) {
    if (!node) return def;
    std::string val = parseString(node);
    if (val == "1") return true;
    if (val == "0") return false;

    THROW_USER_EXCEPTION(1) << "Can't convert node <" << node.name()
        << "> value \"" << val << "\" to boolean";
}


VOCAnnotation VOCAnnotationParser::parse(const std::string& filename) {
    using namespace pugi;
    xml_document doc;

    xml_parse_result result = doc.load_file(filename.c_str());

    if (result.status != pugi::status_ok) {
        throw UserException(result.status)
                << "parsing failed at offset " << result.offset << ": " << result.description();
    }

    xml_node annNode = doc.child("annotation");
    if (!annNode) {
        THROW_USER_EXCEPTION(1) << "No root <annotation> tag";
    }
    VOCAnnotation ann;

    try {
        ann.filename = parseString(annNode.child("filename"));
        ann.folder = parseString(annNode.child("folder"));
        ann.segmented = parseBool(annNode.child("segmented"));

        xml_node sizeNode = annNode.child("size");
        ann.size.depth = parseInt(sizeNode.child("depth"));
        ann.size.height = parseInt(sizeNode.child("height"));
        ann.size.width = parseInt(sizeNode.child("width"));

        xml_node sourceNode = annNode.child("source");
        ann.source.annotation = parseString(sourceNode.child("annotation"));
        ann.source.database = parseString(sourceNode.child("database"));
        ann.source.image = parseString(sourceNode.child("image"));


        for (xml_node objNode = annNode.child("object"); objNode; objNode = objNode.next_sibling("object")) {
            VOCObject obj;
            obj.name = parseString(objNode.child("name"));
            obj.difficult = parseBool(objNode.child("difficult"));
            obj.occluded = parseBool(objNode.child("occluded"));
            obj.pose = parseString(objNode.child("pose"));
            obj.truncated = parseBool(objNode.child("truncated"));

            xml_node bndboxNode = objNode.child("bndbox");
            obj.bndbox.xmin = parseInt(bndboxNode.child("xmin"));
            obj.bndbox.xmax = parseInt(bndboxNode.child("xmax"));
            obj.bndbox.ymin = parseInt(bndboxNode.child("ymin"));
            obj.bndbox.ymax = parseInt(bndboxNode.child("ymax"));

            ann.objects.push_back(obj);
        }
    }
    catch (const std::invalid_argument& e) {
        THROW_USER_EXCEPTION(1) << "conversion error: " << e.what();
    }

    return ann;
}

VOCAnnotationParser::~VOCAnnotationParser() {
}

inline bool ends_with(std::string const & value, std::string const & ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

VOCAnnotationCollector::VOCAnnotationCollector(const std::string& path) {
    VOCAnnotationParser parser;
    if (ends_with(path, ".xml")) {
        // A single file
        _annotations.push_back(parser.parse(path));
    } else {
        std::list<std::string> baseDirContents = getDirContents(path, true);
        for (const auto& sub : baseDirContents) {
            std::list<std::string> annotationDirContents = getDirContents(sub, true);
            if (annotationDirContents.size() == 0 && ends_with(sub, ".xml")) {
                _annotations.push_back(parser.parse(sub));
            } else {
                for (const auto& file : annotationDirContents) {
                    if (ends_with(file, ".xml")) {
                        _annotations.push_back(parser.parse(file));
                    }
                }
            }
        }
    }
}

