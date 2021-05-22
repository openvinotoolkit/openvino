// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_model_validator.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <map>
#include <vector>

namespace detail {
namespace onnx {
    enum Field {
        IR_VERSION = 1,
        PRODUCER_NAME = 2,
        PRODUCER_VERSION = 3,
        DOMAIN_ = 4, // DOMAIN collides with some existing symbol in MSVC thus - underscore
        MODEL_VERSION = 5,
        DOC_STRING = 6,
        GRAPH = 7,
        OPSET_IMPORT = 8,
        METADATA_PROPS = 14,
        TRAINING_INFO = 20
    };

    enum WireType {
        VARINT = 0,
        BITS_64 = 1,
        LENGTH_DELIMITED = 2,
        START_GROUP = 3,
        END_GROUP = 4,
        BITS_32 = 5
    };

    // A PB key consists of a field number (defined in onnx.proto) and a type of data that follows this key
    using PbKey = std::pair<char, char>;

    // This pair represents a key found in the encoded model and optional size of the payload
    // that follows the key (in bytes). They payload should be skipped for the fast check purposes.
    using ONNXField = std::pair<Field, uint32_t>;

    bool is_correct_onnx_field(const PbKey& decoded_key) {
        static const std::map<Field, WireType> onnx_fields = {
            {IR_VERSION, VARINT},
            {PRODUCER_NAME, LENGTH_DELIMITED},
            {PRODUCER_VERSION, LENGTH_DELIMITED},
            {DOMAIN_, LENGTH_DELIMITED},
            {MODEL_VERSION, VARINT},
            {DOC_STRING, LENGTH_DELIMITED},
            {GRAPH, LENGTH_DELIMITED},
            {OPSET_IMPORT, LENGTH_DELIMITED},
            {METADATA_PROPS, LENGTH_DELIMITED},
            {TRAINING_INFO, LENGTH_DELIMITED},
        };

        if (!onnx_fields.count(static_cast<Field>(decoded_key.first))) {
            return false;
        }

        return onnx_fields.at(static_cast<Field>(decoded_key.first)) == static_cast<WireType>(decoded_key.second);
    }

    /**
     * Only 7 bits in each component of a varint count in this algorithm. The components form
     * a decoded number when they are concatenated bitwise in a reverse order. For example:
     * bytes = [b1, b2, b3, b4]
     * varint = b4 ++ b3 ++ b2 ++ b1  <== only 7 bits of each byte should be extracted before concat
     *
     *             b1         b2
     * bytes = [00101100, 00000010]
     *             b2         b1
     * varint = 0000010 ++ 0101100 = 100101100 => decimal: 300
     * Each consecutive varint byte needs to be left shifted "7 x its position in the vector"
     * and bitwise added to the accumulator afterwards.
     */
    uint32_t varint_bytes_to_number(const std::vector<char>& bytes) {
        uint32_t accumulator = 0u;

        for (size_t i = 0; i < bytes.size(); ++i) {
            uint32_t b = bytes[i];
            b <<= 7 * i;
            accumulator |= b;
        }

        return accumulator;
    }

    uint32_t decode_varint(std::istream& model) {
        std::vector<char> bytes;
        // max 4 bytes for a single value because this function returns a 32-bit long decoded varint
        const size_t MAX_VARINT_BYTES = 4u;
        // optimization to avoid allocations during push_back calls
        bytes.reserve(MAX_VARINT_BYTES);

        char key_component = 0;
        model.get(key_component);

        // keep reading all bytes from the stream which have the MSB on
        while (key_component & 0x80 && bytes.size() < MAX_VARINT_BYTES) {
            // drop the most significant bit
            const char component = key_component & ~0x80;
            bytes.push_back(component);
            model.get(key_component);
        }
        // add the last byte - the one with MSB off
        bytes.push_back(key_component);

        return varint_bytes_to_number(bytes);
    }

    PbKey decode_key(const char key) {
        // 3 least significant bits
        const char wire_type = key & 0b111;
        // remaining bits
        const char field_number = key >> 3;
        return {field_number, wire_type};
    }

    ONNXField decode_next_field(std::istream& model) {
        char key = 0;
        model.get(key);

        const auto decoded_key = decode_key(key);

        if (!is_correct_onnx_field(decoded_key)) {
            throw std::runtime_error{"Incorrect field detected in the processed model"};
        }

        const auto onnx_field = static_cast<Field>(decoded_key.first);

        switch (decoded_key.second) {
            case VARINT: {
                // the decoded varint is the payload in this case but its value doesnt matter
                // in the fast check process so we just discard it
                decode_varint(model);
                return {onnx_field, 0};
            }
            case LENGTH_DELIMITED:
                // the varint following the key determines the payload length
                return {onnx_field, decode_varint(model)};
            case BITS_64:
                return {onnx_field, 8};
            case BITS_32:
                return {onnx_field, 4};
            case START_GROUP:
            case END_GROUP:
                throw std::runtime_error{"StartGroup and EndGroup are not used in ONNX models"};
            default:
                throw std::runtime_error{"Unknown WireType encountered in the model"};
        }
    }

    inline void skip_payload(std::istream& model, uint32_t payload_size) {
        model.seekg(payload_size, std::ios::cur);
    }
} // namespace onnx

namespace prototxt {
    bool contains_onnx_model_keys(const std::string& model, const size_t expected_keys_num) {
        size_t keys_found = 0;

        const std::vector<std::string> onnx_keys = {
            "ir_version", "producer_name", "producer_version", "domain", "model_version",
            "doc_string", "graph", "opset_import", "metadata_props", "training_info"
        };

        size_t search_start_pos = 0;

        while (keys_found < expected_keys_num) {
            const auto key_finder = [&search_start_pos, &model](const std::string& key) {
                const auto key_pos = model.find(key, search_start_pos);
                if (key_pos != model.npos) {
                    // don't search from the beginning each time
                    search_start_pos = key_pos + key.size();
                    return true;
                } else {
                    return false;
                }
            };

            const auto found = std::any_of(std::begin(onnx_keys), std::end(onnx_keys), key_finder);
            if (!found) {
                break;
            } else {
                ++keys_found;
            }
        }

        return keys_found == expected_keys_num;
    }
} // namespace prototxt
} // namespace detail

namespace InferenceEngine {
    bool is_valid_model(std::istream& model, onnx_format) {
        // the model usually starts with a 0x08 byte indicating the ir_version value
        // so this checker expects at least 2 valid ONNX keys to be found in the validated model
        const unsigned int EXPECTED_FIELDS_FOUND = 2u;
        unsigned int valid_fields_found = 0u;
        try {
            while (!model.eof() && valid_fields_found < EXPECTED_FIELDS_FOUND) {
                const auto field = detail::onnx::decode_next_field(model);

                ++valid_fields_found;

                if (field.second > 0) {
                    detail::onnx::skip_payload(model, field.second);
                }
            }

            return valid_fields_found == EXPECTED_FIELDS_FOUND;
        } catch (...) {
            return false;
        }
    }

    bool is_valid_model(std::istream& model, prototxt_format) {
        std::array<char, 512> head_of_file;

        model.seekg(0, model.beg);
        model.read(head_of_file.data(), head_of_file.size());
        model.clear();
        model.seekg(0, model.beg);

        return detail::prototxt::contains_onnx_model_keys(
            std::string{std::begin(head_of_file), std::end(head_of_file)}, 2);
    }
} // namespace InferenceEngine
