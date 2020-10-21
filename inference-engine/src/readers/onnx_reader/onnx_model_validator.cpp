// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_model_validator.hpp"

#include <algorithm>
#include <array>
#include <exception>
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

    bool is_correct_onnx_field(const char decoded_field) {
        constexpr Field allowed_fields[] = {
            IR_VERSION, PRODUCER_NAME, PRODUCER_VERSION, DOMAIN_, MODEL_VERSION, DOC_STRING,
            GRAPH, OPSET_IMPORT, METADATA_PROPS, TRAINING_INFO
        };

        const auto is_allowed = [&decoded_field](const Field field) {
            return field == static_cast<Field>(decoded_field);
        };

        return std::any_of(std::begin(allowed_fields), std::end(allowed_fields), is_allowed);
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
        bytes.reserve(4);

        char key_component = 0;
        model.get(key_component);

        while (key_component & 0x80) {
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

        if (!is_correct_onnx_field(decoded_key.first)) {
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

        auto next_key_to_find = onnx_keys.begin();
        size_t search_start_pos = 0;

        while (keys_found < expected_keys_num) {
            for (auto key_to_find = next_key_to_find; key_to_find != onnx_keys.end(); ++key_to_find) {
                const auto key_pos = model.find(*key_to_find, search_start_pos);

                if (key_pos != model.npos) {
                    ++keys_found;
                    // don't search for the same key twice
                    ++next_key_to_find;
                    // don't search from the beginning each time
                    search_start_pos = key_pos + key_to_find->size();
                    break;
                }
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
