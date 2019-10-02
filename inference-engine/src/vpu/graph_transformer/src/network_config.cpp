// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/network_config.hpp>

#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <unordered_map>

#include <pugixml.hpp>

#include <vpu/utils/extra.hpp>
#include <vpu/utils/string.hpp>

namespace vpu {

namespace {

template <class Set>
void parseStringSet(const std::string& str, Set& set) {
    splitStringList(str, set, ',');
}

SmallVector<std::string> splitString(const std::string& str, char sep) {
    SmallVector<std::string> out;
    splitStringList(str, out, sep);
    return out;
}

template <typename K, typename V>
V getValFromMap(const std::unordered_map<K, V>& map, const K& key, const V& def) {
    auto it = map.find(key);
    if (it == map.end())
        return def;
    return it->second;
}

template <typename K, typename V>
const V* getValFromMap(const std::unordered_map<K, V>& map, const K& key) {
    auto it = map.find(key);
    if (it == map.end())
        return nullptr;
    return &it->second;
}

template <typename K1, typename K2, typename V>
const V* getValFromMap(const std::unordered_map<K1, std::unordered_map<K2, V>>& map,
                       const K1& key1, const K2& key2) {
    auto it1 = map.find(key1);
    if (it1 == map.end())
        return nullptr;

    auto it2 = it1->second.find(key2);
    if (it2 == it1->second.end())
        return nullptr;

    return &it2->second;
}

float parseScale(const std::string& val) {
    float scale = 0.0f;
    try {
        scale = std::stof(val);
    } catch (...) {
        VPU_THROW_EXCEPTION
            << "Invalid scale value " << val;
    }

    if (scale <= 0.0f) {
        VPU_THROW_EXCEPTION
            << "Invalid scale value " << scale;
    }

    return scale;
}

template <typename K> K xmlAttrToVal(const pugi::xml_attribute& attr);
template<> std::string xmlAttrToVal<std::string>(const pugi::xml_attribute& attr) {
    return attr.as_string("");
}

template <typename K>
std::map<K, pugi::xml_node> xmlCollectChilds(const pugi::xml_node& xmlParent,
                                             const char* childName,
                                             const char* keyName) {
    std::map<K, pugi::xml_node> out;

    for (auto xmlChild = xmlParent.child(childName);
         !xmlChild.empty();
         xmlChild = xmlChild.next_sibling(childName)) {
        auto xmlKey = xmlChild.attribute(keyName);
        if (xmlKey.empty()) {
            VPU_THROW_EXCEPTION << "Missing " << keyName << " attribute in " << childName;
        }

        auto key = xmlAttrToVal<K>(xmlKey);

        if (out.count(key) != 0) {
            VPU_THROW_EXCEPTION
                << "" << xmlParent.name() << " already has " << childName
                << " with " << keyName << " " << key;
        }

        out[key] = xmlChild;
    }

    return out;
}

template <typename T> T parseVal(const std::string& val);
template <> bool parseVal<bool>(const std::string& val) {
    ie::details::CaselessEq<std::string> cmp;

    if (cmp(val, "true"))
        return true;
    if (cmp(val, "false"))
        return false;

    VPU_THROW_EXCEPTION << "Invalid bool value " << val;
}
template <> float parseVal<float>(const std::string& val) {
    return parseScale(val);
}

template <typename K, typename V>
void xmlUpdateMap(const pugi::xml_node& xmlNode,
                  std::unordered_map<K, V>& map,
                  const K& key) {
    if (xmlNode.empty())
        return;

    map[key] = parseVal<V>(xmlNode.child_value());
}

}  // namespace

bool NetworkConfig::skipAllLayers() const {
    if (_noneLayers.size() == 1) {
        auto val = *_noneLayers.begin();
        return val == "*";
    }
    return false;
}

bool NetworkConfig::hwDisabled(const std::string& layerName) const {
    if (!_hwWhiteList.empty()) {
        return _hwWhiteList.count(layerName) == 0;
    }

    if (!_hwBlackList.empty()) {
        return _hwBlackList.count(layerName) != 0;
    }

    return false;
}

void NetworkConfig::parse(const CompilationConfig& config) {
    parseStringSet(config.noneLayers, _noneLayers);
    parseStringSet(config.hwWhiteList, _hwWhiteList);
    parseStringSet(config.hwBlackList, _hwBlackList);

    if (config.networkConfig.empty())
        return;

    auto props = splitString(config.networkConfig, ',');

    std::string configFileName;
    std::string curDataName;
    bool hasOption = false;

    auto checkChildOption = [&curDataName, &hasOption]() {
        if (!curDataName.empty() && !hasOption) {
            VPU_THROW_EXCEPTION
                << "Incorrect VPU_NETWORK_CONFIG option : "
                << "data " << curDataName << " doesn't have any option";
        }
    };

    for (const auto& prop : props) {
        auto propTokens = splitString(prop, '=');
        if (propTokens.size() != 2) {
            VPU_THROW_EXCEPTION
                << "Incorrect VPU_NETWORK_CONFIG option : "
                << "it should be <key>=<value> list separated by `,`";
        }

        auto propName = propTokens[0];
        auto propValue = propTokens[1];

        if (propName == "file") {
            if (!configFileName.empty()) {
                VPU_THROW_EXCEPTION
                    << "Incorrect VPU_NETWORK_CONFIG option : "
                    << "can't use `file` key more than once";
            }

            checkChildOption();

            configFileName = propValue;

            continue;
        }

        if (propName == "data") {
            checkChildOption();

            curDataName = propValue;
            hasOption = false;

            continue;
        }

        if (propName == "scale") {
            if (curDataName.empty()) {
                VPU_THROW_EXCEPTION
                    << "Incorrect VPU_NETWORK_CONFIG option : "
                    << "missing data name for scale parameter";
            }

            if (_dataScale.count(curDataName) != 0) {
                VPU_THROW_EXCEPTION
                    << "Incorrect VPU_NETWORK_CONFIG option : "
                    << "data " << curDataName << " already have scale factor";
            }

            _dataScale[curDataName] = parseScale(propValue);

            curDataName = "";
            hasOption = false;

            continue;
        }

        VPU_THROW_EXCEPTION
            << "Incorrect VPU_NETWORK_CONFIG option : "
            << "unknown option " << propName;
    }

    checkChildOption();

    if (configFileName.empty())
        return;

    pugi::xml_document xmlDoc;
    auto xmlRes = xmlDoc.load_file(configFileName.c_str());
    if (xmlRes.status != pugi::status_ok) {
        std::ifstream file(configFileName);
        if (!file.is_open()) {
            VPU_THROW_EXCEPTION << "Can't open network config file " << configFileName;
        }

        std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        size_t line = 1;
        size_t pos = 0;
        for (auto token : str) {
            if (token != '\n') {
                pos++;
            } else {
                line++;
                pos = 0;
            }

            if (pos >= xmlRes.offset)
                break;
        }

        VPU_THROW_EXCEPTION
            << "Error loading XML file: " << configFileName
            << ", " << xmlRes.description()
            << " at line: " << line << " pos: " << pos;
    }

    auto xmlRoot = xmlDoc.document_element();
    std::string docName(xmlRoot.name());
    if (docName != "vpu_net_config") {
        VPU_THROW_EXCEPTION
            << "Invalid network config file " << configFileName
            << " : is is not a VPU network config";
    }

    auto docVersion = xmlRoot.attribute("version").as_int(0);
    if (docVersion != 1) {
        VPU_THROW_EXCEPTION
            << "Invalid network config file " << configFileName
            << " : unsupported version " << docVersion;
    }

    auto datas = xmlCollectChilds<std::string>(xmlRoot.child("data"), "data", "name");
    auto layers = xmlCollectChilds<std::string>(xmlRoot.child("layers"), "layer", "name");

    for (const auto& dataInfo : datas) {
        const auto& dataName = dataInfo.first;
        const auto& xmlData = dataInfo.second;

        xmlUpdateMap(xmlData.child("scale"), _dataScale, dataName);
    }

    for (const auto& layerInfo : layers) {
        const auto& layerName = layerInfo.first;
        const auto& xmlLayer = layerInfo.second;

        if (auto xmlHw = xmlLayer.child("hw")) {
            if (auto xmlEnable = xmlHw.child("enable")) {
                if (!parseVal<bool>(xmlEnable.child_value()))
                    _hwBlackList.insert(layerName);
            }
        }
    }
}

}  // namespace vpu
