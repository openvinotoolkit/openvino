// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/enums.hpp>

#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include <vpu/utils/string.hpp>
#include <vpu/utils/small_vector.hpp>

#include <ie_common.h>

namespace vpu {

namespace {

void removeCharFromString(std::string& str, char ch) {
    str.erase(std::remove(str.begin(), str.end(), ch), str.end());
}

}  // namespace

std::ostream& printValue(std::ostream& os, const std::string& strMap, int32_t val) {
    std::string strMapCopy = strMap;

    removeCharFromString(strMapCopy, ' ');
    removeCharFromString(strMapCopy, '(');

    SmallVector<std::string> enumTokens;
    splitStringList(strMapCopy, enumTokens, ',');

    int32_t inxMap = 0;
    for (const auto& token : enumTokens) {
        // Token: [EnumName | EnumName=EnumValue]
        std::string enumName;
        if (token.find('=') == std::string::npos) {
            enumName = token;
        } else {
            SmallVector<std::string, 2> enumNameValue;
            splitStringList(token, enumNameValue, '=');
            IE_ASSERT(enumNameValue.size() == 2);

            enumName = enumNameValue[0];
            inxMap = std::stoi(enumNameValue[1], nullptr, 0);
        }

        if (inxMap == val) {
            os << enumName;
            return os;
        }

        ++inxMap;
    }

    os << std::to_string(val);
    return os;
}

}  // namespace vpu
