// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Core-local internal property enums. The ov::Property declarations that
// referenced the full runtime (ShapePredictor, ImplForcingMap,
// WeightlessCacheAttribute, ...) are intentionally dropped in the standalone
// core; only the enums actually used by the runtime are kept.

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>

#include "except.hpp"

namespace ov::intel_gpu {

enum class QueueTypes : int16_t {
    in_order,
    out_of_order
};

inline std::ostream& operator<<(std::ostream& os, const QueueTypes& val) {
    switch (val) {
        case QueueTypes::in_order: os << "in-order"; break;
        case QueueTypes::out_of_order: os << "out-of-order"; break;
        default: os << "unknown";
    }

    return os;
}

inline std::istream& operator>>(std::istream& is, QueueTypes& val) {
    std::string str;
    is >> str;
    if (str == "in-order") {
        val = QueueTypes::in_order;
    } else if (str == "out-of-order") {
        val = QueueTypes::out_of_order;
    } else {
        OPENVINO_THROW("Unsupported QueueTypes value: ", str);
    }
    return is;
}

enum class DumpFormat : uint8_t {
    binary = 0,
    text = 1,
    text_raw = 2,
};

inline std::ostream& operator<<(std::ostream& os, const DumpFormat& val) {
    switch (val) {
        case DumpFormat::binary: os << "binary"; break;
        case DumpFormat::text: os << "text"; break;
        case DumpFormat::text_raw: os << "text_raw"; break;
        default: os << "unknown";
    }

    return os;
}

inline std::istream& operator>>(std::istream& is, DumpFormat& val) {
    std::string str;
    is >> str;
    if (str == "binary") {
        val = DumpFormat::binary;
    } else if (str == "text") {
        val = DumpFormat::text;
    } else if (str == "text_raw") {
        val = DumpFormat::text_raw;
    } else {
        OPENVINO_THROW("Unsupported DumpFormat value: ", str);
    }
    return is;
}

enum class DumpTensors : uint8_t {
    all = 0,
    in = 1,
    out = 2,
};

inline std::ostream& operator<<(std::ostream& os, const DumpTensors& val) {
    switch (val) {
        case DumpTensors::all: os << "all"; break;
        case DumpTensors::in: os << "in"; break;
        case DumpTensors::out: os << "out"; break;
        default: os << "unknown";
    }

    return os;
}

inline std::istream& operator>>(std::istream& is, DumpTensors& val) {
    std::string str;
    is >> str;
    if (str == "all") {
        val = DumpTensors::all;
    } else if (str == "in") {
        val = DumpTensors::in;
    } else if (str == "out") {
        val = DumpTensors::out;
    } else {
        OPENVINO_THROW("Unsupported DumpTensors value: ", str);
    }
    return is;
}

}  // namespace ov::intel_gpu
