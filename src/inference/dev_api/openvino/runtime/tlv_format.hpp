// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ov::runtime {
/**
 * @brief Utility for reading and writing data in TLV (Tag-Length-Value) format to/from streams. TLV format allows to
 * store multiple entries in a single stream with each entry having a tag to identify the type of the entry and length
 * to identify the size of the entry value. This format is useful for storing multiple pieces of data in a single file
 * or stream without requiring a predefined structure, as each entry can be read independently based on its tag and
 * length.
 */
struct TLVFormat {
    /** @brief Type of the tag field in TLV entry. */
    using TagType = uint32_t;

    /** @brief Type of the length field in TLV entry. */
    using LengthType = uint64_t;

    /**
     * @brief Write a TLV entry to the stream.
     * @param stream Output stream to write the entry to.
     * @param tag Tag of the entry.
     * @param size Size of the entry value.
     * @param data Pointer to the entry value data.
     */
    static void write_entry(std::ostream& stream, TagType tag, LengthType size, const char* data);

    /**
     * @brief Type of callable for writing TLV entry values based on their tag.
     * The callable takes an output stream to write the entry value to as a parameter.
     */
    using ValueWriter = std::function<void(std::ostream&)>;

    /**
     * @brief Write a TLV entry to the stream using a value writer callable.
     * @param stream Output stream to write the entry to.
     * @param tag Tag of the entry.
     * @param writer Callable that writes the entry value to the stream.
     */
    static void write_entry(std::ostream& stream, TagType tag, const ValueWriter& writer);

    /**
     * @brief Read a TLV entry from the stream into a container.
     * @param stream Input stream to read the entry from. It should be positioned at the beginning of an entry.
     * @param tag Output parameter to store the tag of the read entry.
     * @param size Output parameter to store the size of the read entry value.
     * @param data Output parameter to store the read entry value.
     * @return True if the entry was successfully read, false otherwise.
     */
    static bool read_entry(std::istream& stream, TagType& tag, LengthType& size, std::vector<char>& data);

    /**
     * @brief Type of callable for reading TLV entry values based on their tag.
     * The callable takes an input stream positioned at the beginning of the entry value and the size of the value as
     * parameters.
     */
    using ValueReader = std::function<bool(std::istream& stream, LengthType size)>;

    /** @brief Map of tag to value reader callable. */
    using ValueScanner = std::unordered_map<TagType, ValueReader>;

    /**
     * @brief Scan entries in the stream and call corresponding reader from scanners for each entry with matching tag.
     * If tag doesn't exist in scanners, entry will be skipped.
     * @note This function doesn't support nested TLV entries and is expected to be used for top-level entries only.
     * @param stream Input stream to read entries from. It should be positioned at the beginning of an entry.
     * @param scanners Map of tag to value reader callable. Value reader will be called with the stream positioned at
     * the beginning of entry value and size of the value as parameters. Value reader is expected to read exactly size
     * bytes from the stream. Failure to do so may result in incorrect parsing of subsequent entries.
     * @param rewind If true, the stream will be rewound to the original position after scanning. If false, the stream
     * will be left at the position after the last scanned entry.
     * @return False if an error occurred during scanning, true otherwise.
     */
    static bool scan_entries(std::istream& stream, const ValueScanner& scanners, bool rewind);
};
}  // namespace ov::runtime
