// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <details/os/os_filesystem.hpp>

#ifdef ENABLE_UNICODE_PATH_SUPPORT

static void fixSlashes(std::string &str) {
    std::replace(str.begin(), str.end(), '/', '\\');
}

static void fixSlashes(std::wstring &str) {
    std::replace(str.begin(), str.end(), L'/', L'\\');
}

static std::wstring stringToWString(std::string input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring result = converter.from_bytes(input);
    return result;
}

static bool copyFile(std::wstring source_path, std::wstring dest_path) {
#ifndef _WIN32
    std::ifstream source(InferenceEngine::details::wStringtoMBCSstringChar(source_path), std::ios::binary);
    std::ofstream dest(InferenceEngine::details::wStringtoMBCSstringChar(dest_path), std::ios::binary);
#else
    fixSlashes(source_path);
    fixSlashes(dest_path);
    std::ifstream source(source_path, std::ios::binary);
    std::ofstream dest(dest_path, std::ios::binary);
#endif
    bool result = source && dest;
    std::istreambuf_iterator<char> begin_source(source);
    std::istreambuf_iterator<char> end_source;
    std::ostreambuf_iterator<char> begin_dest(dest);
    copy(begin_source, end_source, begin_dest);

    source.close();
    dest.close();
    return result;
}

static bool copyFile(std::string source_path, std::wstring dest_path) {
    return copyFile(stringToWString(source_path), dest_path);
}

static std::wstring addUnicodePostfixToPath(std::string source_path, std::wstring postfix) {
    fixSlashes(source_path);
    std::wstring result = stringToWString(source_path);
    std::wstring file_name = result.substr(0, result.size() - 4);
    std::wstring extension = result.substr(result.size() - 4, result.size());
    result = file_name + postfix + extension;
    return result;
}

static void removeFile(std::wstring path) {
    int result = 0;
    if (!path.empty()) {
#ifdef _WIN32
        result = _wremove(path.c_str());
#else
        result = remove(InferenceEngine::details::wStringtoMBCSstringChar(path).c_str());
#endif
    }
}

static const std::vector<std::wstring> test_unicode_postfix_vector = {
    L"unicode_Яㅎあ",
    L"ひらがな日本語",
    L"大家有天分",
    L"עפצקרשתםןףץ",
    L"ث خ ذ ض ظ غ",
    L"그것이정당하다",
    L"АБВГДЕЁЖЗИЙ",
    L"СТУФХЦЧШЩЬЮЯ"
};

#endif  // ENABLE_UNICODE_PATH_SUPPORT
