// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace ov {
struct TLVFormat {
    using tag_type = uint32_t;
    using length_type = uint64_t;

    using value_writer_callable = std::function<void(std::ostream&)>;

    static void write_entry(std::ostream& stream, tag_type tag, length_type size, const uint8_t* data);
    static void write_entry(std::ostream& stream, tag_type tag, const value_writer_callable& writer);

    static bool read_entry(std::istream& stream, tag_type& tag, length_type& size, std::vector<uint8_t>& data);
    static bool read_entry(std::istream& stream, tag_type& tag, length_type& size, std::string& data);

    using value_reader_callable = std::function<void(std::istream& stream, length_type size)>;
    using value_scanners = std::unordered_map<tag_type, value_reader_callable>;

    /// @brief Scan entries in the stream and call corresponding reader from scanners for each entry with matching tag.
    /// If tag doesn't exist in scanners, entry will be skipped.
    /// @note This function doesn't support nested TLV entries and is expected to be used for top-level entries only.
    /// @param stream Input stream to read entries from. It should be positioned at the beginning of an entry.
    /// @param scanners Map of tag to value reader callable. Value reader will be called with the stream positioned at
    /// the beginning of entry value and size of the value as parameters. Value reader is expected to read exactly size
    /// bytes from the stream. Failure to do so may result in incorrect parsing of subsequent entries.
    /// @param rewind If true, the stream will be rewound to the original position after scanning. If false, the stream
    /// will be left at the position after the last scanned entry.
    static void scan_entries(std::istream& stream, const value_scanners& scanners, bool rewind);
};
}  // namespace ov
