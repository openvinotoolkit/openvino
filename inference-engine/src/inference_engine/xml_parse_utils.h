// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include "pugixml.hpp"
#include "ie_common.h"
#include "ie_api.h"
#include <string>
#include <ie_precision.hpp>

#define FOREACH_CHILD(c, p, tag) for (auto c = p.child(tag); !c.empty(); c = c.next_sibling(tag))

namespace XMLParseUtils {

INFERENCE_ENGINE_API_CPP(int) GetIntAttr(const pugi::xml_node &node, const char *str);

INFERENCE_ENGINE_API_CPP(int) GetIntAttr(const pugi::xml_node &node, const char *str, int defVal);

INFERENCE_ENGINE_API_CPP(uint64_t) GetUInt64Attr(const pugi::xml_node &node, const char *str);

INFERENCE_ENGINE_API_CPP(uint64_t) GetUInt64Attr(const pugi::xml_node &node, const char *str, uint64_t defVal);

INFERENCE_ENGINE_API_CPP(unsigned int) GetUIntAttr(const pugi::xml_node &node, const char *str);

INFERENCE_ENGINE_API_CPP(unsigned int) GetUIntAttr(const pugi::xml_node &node, const char *str, unsigned int defVal);

INFERENCE_ENGINE_API_CPP(std::string) GetStrAttr(const pugi::xml_node &node, const char *str);

INFERENCE_ENGINE_API_CPP(std::string) GetStrAttr(const pugi::xml_node &node, const char *str, const char *def);

INFERENCE_ENGINE_API_CPP(float) GetFloatAttr(const pugi::xml_node &node, const char *str);

INFERENCE_ENGINE_API_CPP(float) GetFloatAttr(const pugi::xml_node &node, const char *str, float defVal);

INFERENCE_ENGINE_API_CPP(InferenceEngine::Precision) GetPrecisionAttr(const pugi::xml_node &node, const char *str);

INFERENCE_ENGINE_API_CPP(InferenceEngine::Precision)
GetPrecisionAttr(const pugi::xml_node &node, const char *str, InferenceEngine::Precision def);

INFERENCE_ENGINE_API_CPP(int) GetIntChild(const pugi::xml_node &node, const char *str, int defVal);

INFERENCE_ENGINE_API_CPP(std::string) NameFromFilePath(const char *filepath);

}  // namespace XMLParseUtils
