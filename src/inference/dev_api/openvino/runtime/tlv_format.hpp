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
 * @brief Utility for reading and writing records in TLV (Tag-Length-Value) scheme to/from streams. TLV scheme allows to
 * store multiple records in a single stream with each record having a tag to identify the type of the record and length
 * to identify the size of the record value. Such scheme is useful for storing multiple pieces of data in a single file
 * or stream without requiring a predefined structure, as each record can be read independently based on its tag and
 * length.
 */
struct TLVTraits {
    /** @brief Type of the tag field in TLV record. */
    using TagType = uint32_t;

    /** @brief Type of the length field in TLV record. */
    using LengthType = uint64_t;
};

/**
 * @brief Write a TLV record to the stream.
 * @param stream Output stream to write the record to.
 * @param tag Tag of the record.
 * @param size Size of the record value.
 * @param data Pointer to the record value data.
 */
void write_tlv_record(std::ostream& stream, TLVTraits::TagType tag, TLVTraits::LengthType size, const char* data);

/**
 * @brief Type of callable for writing TLV record values based on their tag.
 * The callable takes an output stream to write the record value to as a parameter.
 */
using TLVValueWriter = std::function<void(std::ostream&)>;

/**
 * @brief Write a TLV record to the stream using a value writer callable.
 * @param stream Output stream to write the record to.
 * @param tag Tag of the record.
 * @param writer Callable that writes the record value to the stream.
 */
void write_tlv_record(std::ostream& stream, TLVTraits::TagType tag, const TLVValueWriter& writer);

/**
 * @brief Read a TLV record from the stream into a container.
 * @param stream Input stream to read the record from. It should be positioned at the beginning of a record.
 * @param tag Output parameter to store the tag of the read record.
 * @param size Output parameter to store the size of the read record value.
 * @param data Output parameter to store the read record value.
 * @return True if the record was successfully read, false otherwise.
 */
bool read_tlv_record(std::istream& stream,
                     TLVTraits::TagType& tag,
                     TLVTraits::LengthType& size,
                     std::vector<char>& data);

/**
 * @brief Type of callable for reading TLV record values based on their tag.
 * The callable takes an input stream positioned at the beginning of the record value and the size of the value as
 * parameters.
 */
using TLVValueReader = std::function<bool(std::istream& stream, TLVTraits::LengthType size)>;

/** @brief Map of tag to value reader callable. */
using TLVValueScanner = std::unordered_map<TLVTraits::TagType, TLVValueReader>;

/**
 * @brief Scan records in the stream and call corresponding reader from scanners for each record with matching tag.
 * If tag doesn't exist in scanners, record will be skipped.
 * @note This function doesn't support nested TLV records and is expected to be used for top-level records only.
 * @param stream Input stream to read records from. It should be positioned at the beginning of a record.
 * @param scanners Map of tag to value reader callable. Value reader will be called with the stream positioned at
 * the beginning of record value and size of the value as parameters. Value reader is expected to read exactly size
 * bytes from the stream. Failure to do so may result in incorrect parsing of subsequent records.
 * @return False if an error occurred during scanning, true otherwise.
 */
bool scan_tlv_records(std::istream& stream, const TLVValueScanner& scanners);

}  // namespace ov::runtime
