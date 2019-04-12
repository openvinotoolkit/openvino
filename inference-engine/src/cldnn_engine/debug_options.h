// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include "cpp/ie_cnn_network.h"
#include <CPP/memory.hpp>
#include <CPP/primitive.hpp>
#include <CPP/network.hpp>

// Debugging options flags
// #define _DEBUG_LAYER_CONTENT
// #define _DEBUG_LAYER_CONTENT_FULL
// #define _DEBUG_LAYER_FORMAT
// #define _PLUGIN_PERF_PRINTS

namespace CLDNNPlugin {

class DebugOptions {
public:
    bool m_bDebugLayerContent;
    bool m_bDebugLayerContentIndexed;
    bool m_bDebugLayerFormat;
    bool m_bPluginPerfPrints;
    cldnn::tensor::value_type m_maxPrintSize;

    DebugOptions();
    void PrintOptions() const;
    static std::string GetFormatName(cldnn::format::type format);
    static std::string GetDataTypeName(cldnn::data_types dataType);
    void PrintInput(const InferenceEngine::TBlob<float>& input) const;
    void PrintIndexedValue(const cldnn::memory& mem, const cldnn::tensor index) const;
    static uint32_t CalcLinearIndex(const cldnn::layout& memLayout, const cldnn::tensor index);

    void PrintNetworkOutputs(std::map<cldnn::primitive_id, cldnn::network_output>& outputsMap) const;
    void DumpSingleOutput(cldnn::primitive_id name, std::map<cldnn::primitive_id, cldnn::network_output>& outputs, bool bSingleFeatureMap = false)const;

    // the functions below will work in release unlike the rest
    void AddTimedEvent(std::string eventName, std::string startingAt = std::string());
    void PrintTimedEvents();
    void ClearTimedEvents();

    void EnableWA(std::string name);
    void DisableWA(std::string name);
    bool IsWAActive(std::string name);

    static std::string IELayoutToString(InferenceEngine::Layout layout);

protected:
    std::map<std::string, std::chrono::steady_clock::time_point> m_TimedEventTimestamp;
    std::map<std::string, std::string> m_TimedEventStart;
    std::set<std::string> m_workaroundNames;

    static float SimpleConvertFP16toFP32(uint16_t u16val);

    template <typename T>
    static void DumpElementsRaw(cldnn::memory& mem, const std::vector<size_t>& pitches, size_t numElements) {
#ifndef NDEBUG
        auto layout = mem.get_layout();
        auto ptr = mem.pointer<T>();
        auto data = ptr.data();  // +offset;
        auto elements = std::min(layout.count(), numElements);
        cldnn::status_t status = CLDNN_SUCCESS;
        for (size_t i = 0; i < elements;) {
            // size_t linearAddress = ... // todo calc linear with pitches
            std::cout << std::setprecision(10)
                      << ((layout.data_type == cldnn::data_types::f32) ? data[i] : cldnn_half_to_float(uint16_t(data[i]), &status))
                      << ", ";
            i++;
            for (auto& pitch : pitches) {
                if ((i % pitch) == 0) {
                    std::cout << std::endl;
                }
            }
        }
#endif  // NDEBUG
    }
};

};  // namespace CLDNNPlugin