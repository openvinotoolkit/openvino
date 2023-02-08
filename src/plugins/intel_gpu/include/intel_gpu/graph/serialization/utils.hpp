// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define RUN_ALL_MODEL_CACHING_TESTS

#include <unordered_map>
#include "ie/ie_common.h"

namespace cldnn {
class serial_util {
public:
    static InferenceEngine::Layout layout_from_string(const std::string& name) {
        static const std::unordered_map<std::string, InferenceEngine::Layout> layouts = {
            { "ANY", InferenceEngine::Layout::ANY },
            { "NCHW", InferenceEngine::Layout::NCHW },
            { "NHWC", InferenceEngine::Layout::NHWC },
            { "NCDHW", InferenceEngine::Layout::NCDHW },
            { "NDHWC", InferenceEngine::Layout::NDHWC },
            { "OIHW", InferenceEngine::Layout::OIHW },
            { "GOIHW", InferenceEngine::Layout::GOIHW },
            { "OIDHW", InferenceEngine::Layout::OIDHW },
            { "GOIDHW", InferenceEngine::Layout::GOIDHW },
            { "SCALAR", InferenceEngine::Layout::SCALAR },
            { "C", InferenceEngine::Layout::C },
            { "CHW", InferenceEngine::Layout::CHW },
            { "HWC", InferenceEngine::Layout::HWC },
            { "HW", InferenceEngine::Layout::HW },
            { "NC", InferenceEngine::Layout::NC },
            { "CN", InferenceEngine::Layout::CN },
            { "BLOCKED", InferenceEngine::Layout::BLOCKED }
        };
        auto it = layouts.find(name);
        if (it != layouts.end()) {
            return it->second;
        }
        IE_THROW(NetworkNotRead) << "Unknown layout with name '" << name << "'";
    }
};

class membuf : public std::streambuf {
public:
    membuf() : _pos(0) { }
    std::vector<int_type>::iterator begin() { return _buf.begin(); }
    std::vector<int_type>::iterator end() { return _buf.end(); }

protected:
    int_type overflow(int_type c) override {
        _buf.emplace_back(c);
        return c;
    }

    int_type uflow() override {
        return (_pos < _buf.size()) ? _buf[_pos++] : EOF;
    }

private:
    std::vector<int_type> _buf;
    size_t _pos;
};
}  // namespace cldnn
