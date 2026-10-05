// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_common/onnx_model_validator.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <map>
#include <stdexcept>
#include <unordered_set>

namespace {
namespace onnx {
enum Field {
    IR_VERSION = 1,
    PRODUCER_NAME = 2,
    PRODUCER_VERSION = 3,
    DOMAIN_ = 4,  // DOMAIN collides with some existing symbol in MSVC thus - underscore
    MODEL_VERSION = 5,
    DOC_STRING = 6,
    GRAPH = 7,
    OPSET_IMPORT = 8,
    METADATA_PROPS = 14,
    TRAINING_INFO = 20,
    FUNCTIONS = 25,
    CONFIGURATION = 26
};

enum WireType { VARINT = 0, BITS_64 = 1, LENGTH_DELIMITED = 2, START_GROUP = 3, END_GROUP = 4, BITS_32 = 5 };

// A PB key consists of a field number (defined in onnx.proto) and a type of data that follows this key
using PbKey = std::pair<uint32_t, uint32_t>;

// This pair represents a key found in the encoded model and optional size of the payload
// that follows the key (in bytes). The payload should be skipped for fast check purposes.
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
        {FUNCTIONS, LENGTH_DELIMITED},
        {CONFIGURATION, LENGTH_DELIMITED},
    };

    if (!onnx_fields.count(static_cast<Field>(decoded_key.first))) {
        return false;
    }

    return onnx_fields.at(static_cast<Field>(decoded_key.first)) == static_cast<WireType>(decoded_key.second);
}

uint32_t decode_varint(std::istream& model) {
    uint32_t value = 0;
    for (uint32_t shift = 0; shift < 32; shift += 7) {
        const auto byte = model.get();
        if (byte == std::char_traits<char>::eof() || (shift == 28 && (byte & 0xf0) != 0)) {
            throw std::runtime_error{"Invalid protobuf varint"};
        }
        value |= static_cast<uint32_t>(byte & 0x7f) << shift;
        if ((byte & 0x80) == 0) {
            return value;
        }
    }
    throw std::runtime_error{"Invalid protobuf varint"};
}

PbKey decode_key(uint32_t key) {
    // 3 least significant bits
    const auto wire_type = key & 0b111;
    // remaining bits
    const auto field_number = key >> 3;
    return {field_number, wire_type};
}

ONNXField decode_next_field(std::istream& model) {
    const auto decoded_key = decode_key(decode_varint(model));

    if (!is_correct_onnx_field(decoded_key)) {
        throw std::runtime_error{"Incorrect field detected in the processed model"};
    }

    const auto onnx_field = static_cast<Field>(decoded_key.first);

    switch (decoded_key.second) {
    case VARINT: {
        // the decoded varint is the payload in this case but its value does not matter
        // in the fast check process so it can be discarded
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
}  // namespace onnx
}  // namespace

namespace ov::frontend::onnx::common {
bool is_valid_model(std::istream& model) {
    // the model usually starts with a 0x08 byte indicating the ir_version value
    // so this checker expects at least 3 valid ONNX keys to be found in the validated model
    const size_t EXPECTED_FIELDS_FOUND = 3u;
    std::unordered_set<::onnx::Field, std::hash<int>> onnx_fields_found = {};
    try {
        while (!model.eof() && onnx_fields_found.size() < EXPECTED_FIELDS_FOUND) {
            const auto field = ::onnx::decode_next_field(model);

            if (onnx_fields_found.count(field.first) > 0) {
                // if the same field is found twice, this is not a valid ONNX model
                return false;
            } else {
                onnx_fields_found.insert(field.first);
                ::onnx::skip_payload(model, field.second);
            }
        }

        return onnx_fields_found.size() == EXPECTED_FIELDS_FOUND;
    } catch (...) {
        return false;
    }
}

}  // namespace ov::frontend::onnx::common
