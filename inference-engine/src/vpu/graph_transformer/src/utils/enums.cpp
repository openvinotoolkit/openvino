// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/enums.hpp>

#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include <vpu/utils/string.hpp>

namespace vpu {

namespace {

void removeCharFromString(std::string& str, char ch) {
    str.erase(std::remove(str.begin(), str.end(), ch), str.end());
}

}  // namespace

std::unordered_map<int32_t, std::string> generateEnumMap(const std::string& strMap) {
    std::unordered_map<int32_t, std::string> retMap;

    std::string strMapCopy = strMap;

    removeCharFromString(strMapCopy, ' ');
    removeCharFromString(strMapCopy, '(');

    std::vector<std::string> enumTokens;
    splitStringList(strMapCopy, enumTokens, ',');

    int32_t inxMap = 0;
    for (const auto& token : enumTokens) {
        // Token: [EnumName | EnumName=EnumValue]
        std::string enumName;
        if (token.find('=') == std::string::npos) {
            enumName = token;
        } else {
            std::vector<std::string> enumNameValue;
            splitStringList(token, enumNameValue, '=');
            IE_ASSERT(enumNameValue.size() == 2);

            enumName = enumNameValue[0];
            inxMap = std::stoi(enumNameValue[1], nullptr, 0);
        }

        retMap[inxMap] = enumName;

        ++inxMap;
    }

    return retMap;
}

}  // namespace vpu
