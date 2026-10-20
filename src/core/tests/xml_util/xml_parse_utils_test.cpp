// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/xml_parse_utils.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <pugixml.hpp>
#include <string>

#include "common_test_utils/file_utils.hpp"

namespace ov::test {

class ParseXmlTest : public testing::Test {
protected:
    std::filesystem::path m_xml_path;

    void SetUp() override {
        m_xml_path = ov::test::utils::generateTestFilePrefix() + "_test.xml";
    }

    void TearDown() override {
        if (std::filesystem::exists(m_xml_path)) {
            std::filesystem::remove(m_xml_path);
        }
    }
};

TEST_F(ParseXmlTest, utf16_cjk_expanded_buffer) {
    // Build the UTF-16 LE test file, which matches the UTF-16 LE format that pugixml recognise via the U+FEFF BOM.
    // The file contains:
    //   u"\uFEFF"   -> FF FE on disk (BOM, tells pugixml this is UTF-16 LE)
    //   u"<a>"      -> 3C 00  61 00  3E 00 (opening tag)
    //   u'\u4E00'   -> 00 4E per char (CJK '一': 2 bytes UTF-16, 3 bytes UTF-8)
    // File is truncated (no closing </a>), so pugixml returns a parse error whose offset
    // is into the expanded UTF-8 decoded buffer (~305 bytes), which exceeds the 208 raw bytes on disk.
    {
        const std::u16string content = u"\uFEFF<a>" + std::u16string(100, u'\u4E00');
        const std::streamsize raw_file_size = content.size() * sizeof(char16_t);
        std::ofstream out(m_xml_path, std::ios::binary);
        ASSERT_TRUE(out.is_open());
        out.write(reinterpret_cast<const char*>(content.data()), raw_file_size);
    }

    // trigger condition (offset > file.size())
    pugi::xml_document doc;
    const auto pugi_result = doc.load_file(m_xml_path.c_str());
    ASSERT_NE(pugi_result.status, pugi::status_ok);

    std::ifstream fs(m_xml_path);
    const std::string raw(std::istreambuf_iterator<char>{fs}, std::istreambuf_iterator<char>{});
    // the crafted file triggers offset > file.size().
    ASSERT_GT(static_cast<size_t>(pugi_result.offset), raw.size());

    const auto result = ov::util::pugixml::parse_xml(m_xml_path);
    // The file is genuinely malformed – parse_xml must always report an error.
    ASSERT_FALSE(result.error_msg.empty());
}

}  // namespace ov::test
