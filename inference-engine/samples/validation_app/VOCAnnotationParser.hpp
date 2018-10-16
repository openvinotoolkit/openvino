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

#pragma once

#include "classification_set_generator.hpp"
#include "pugixml/pugixml.hpp"

#include <string>
#include <vector>

#pragma pack(1)
struct VOCBoundingBox {
    int xmin, xmax, ymin, ymax;
};

struct VOCObject {
    std::string name;
    VOCBoundingBox bndbox;
    int difficult;
    int occluded;
    int truncated;
    std::string pose;
};

struct VOCAnnotation {
    std::string filename;
    std::string folder;
    std::vector<VOCObject> objects;
    int segmented;

    struct Size {
        uint16_t depth;
        size_t width;
        size_t height;
    } size;

    struct Source {
        std::string annotation;
        std::string database;
        std::string image;
    } source;
};
#pragma pack()

class VOCAnnotationParser {
private:
    static std::string parseString(const pugi::xml_node& node, const std::string& def = "");
    static int parseInt(const pugi::xml_node& node, const int def = 0);
    static bool parseBool(const pugi::xml_node& node, bool def = false);

public:
    VOCAnnotationParser() { }
    VOCAnnotation parse(const std::string& filename);
    virtual ~VOCAnnotationParser();
};

class VOCAnnotationCollector : public ClassificationSetGenerator {
private:
    std::vector<VOCAnnotation> _annotations;
public:
    explicit VOCAnnotationCollector(const std::string& path);
    const std::vector<VOCAnnotation>& annotations() const { return _annotations; }
    const VOCAnnotation* annotationByFile(const std::string& filename) const {
        for (auto& ann : _annotations) {
            if (ann.filename == filename) return &ann;
        }
        return nullptr;
    }
};
