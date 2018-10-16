// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <iomanip>
#ifndef NDEBUG
    #include <algorithm>
    #include <cmath>
#endif

#include "debug_options.h"

namespace CLDNNPlugin {

DebugOptions::DebugOptions() {
    m_bDebugLayerContent =
#ifdef _DEBUG_LAYER_CONTENT
        true;
#else
        false;
#endif

    m_bDebugLayerContentIndexed =
#ifdef _DEBUG_LAYER_CONTENT_INDEXED
        true;
#else
        false;
#endif

    m_bDebugLayerFormat =
#ifdef _DEBUG_LAYER_FORMAT
        true;
#else
        false;
#endif

    m_bPluginPerfPrints =
#ifdef _PLUGIN_PERF_PRINTS
        true;
#else
        false;
#endif

    m_maxPrintSize =
#ifdef _DEBUG_LAYER_CONTENT_FULL
        1000000000;
#else
        3;
#endif
}

void DebugOptions::PrintOptions() const {
#ifndef NDEBUG
    std::cout << "Debug Options:" << std::endl;
    std::cout << "\tDebug Layer Content: " << m_bDebugLayerContent << std::endl;
    std::cout << "\tDebug Layer Content Indexed: " << m_bDebugLayerContentIndexed << std::endl;
    std::cout << "\tDebug Layers Format: " << m_bDebugLayerFormat << std::endl;
    std::cout << "\tPlugin Performance Prints: " << m_bPluginPerfPrints << std::endl;
    std::cout << "\tPrint Size: " << m_maxPrintSize << std::endl;
#endif  // NDEBUG
}

std::string DebugOptions::GetFormatName(cldnn::format::type format) {
    switch (format) {
    case cldnn::format::yxfb:
        return "yxfb";
    case cldnn::format::byxf:
        return "byxf";
    case cldnn::format::bfyx:
        return "bfyx";
    case cldnn::format::fyxb:
        return "fyxb";
    default:
        return "Unknown Format";
    }
}

std::string DebugOptions::GetDataTypeName(cldnn::data_types dataType) {
    switch (dataType) {
    case cldnn::data_types::f16:
        return "f16";
    case cldnn::data_types::f32:
        return "f32";
    default:
        return "Unknown Data Type";
    }
}

void DebugOptions::PrintInput(const InferenceEngine::TBlob<float>& input) const {
#ifndef NDEBUG
    const float* inputBlobPtr = input.readOnly();

    if (m_bDebugLayerContent) {
        std::cout << "Input (" << input.size() << ") = ";
        for (size_t i = 0; i < std::min<size_t>(m_maxPrintSize, input.size()); i++) {
            std::cout << inputBlobPtr[i] << ", ";
        }
        std::cout << std::endl;
    }
#endif  // NDEBUG
}

float DebugOptions::SimpleConvertFP16toFP32(uint16_t u16val) {
#ifndef NDEBUG
    // convert to fp32 (1,5,10)->(1,8,23)
    // trivial conversion not handling inf/denorm
    uint32_t sign = (u16val & 0x8000U) << 16;
    uint32_t mantissa = (u16val & 0x3FFU) << 13;
    uint32_t exp_val_f16 = (u16val & 0x7C00U) >> 10;
    uint32_t exp = (exp_val_f16 == 0x1FU ? 0xFFU : exp_val_f16 + 127 - 15) << 23;;
    uint32_t val = sign | exp | mantissa;
    float fval = *(reinterpret_cast<float*>(&val));
    return (fabs(fval) < 1e-4f) ? 0.0f : fval;  // clamp epsilon fp16 to 0
#endif  // NDEBUG
    return 0;
}
void DebugOptions::PrintIndexedValue(const cldnn::memory& mem, const cldnn::tensor index) const {
#ifndef NDEBUG
    auto layout = mem.get_layout();
    float fval;
    switch (layout.data_type) {
    case cldnn::data_types::f32: {
        auto p32 = mem.pointer<float>();
        auto resPtrF32 = p32.data();
        fval = resPtrF32[CalcLinearIndex(layout, index)];
    }
    break;
    case cldnn::data_types::f16:
    {
        auto p16 = mem.pointer<uint16_t>();
        auto resPtrU16 = p16.data();
        fval = SimpleConvertFP16toFP32(resPtrU16[CalcLinearIndex(layout, index)]);
    }
    break;
    default:
        assert(0);  // unhandled data type
        fval = 0.0f;
    }

    if (m_bDebugLayerContentIndexed) {
        std::cout << "\t[";
        for (size_t i = 0; i < index.raw.size(); i++) {
            std::cout << index.raw[i] << ",";
        }
        std::cout << "] = " << fval << "\n";
    } else {
        std::cout << fval << ", ";
    }
#endif  // NDEBUG
}

uint32_t DebugOptions::CalcLinearIndex(const cldnn::layout& memLayout, const cldnn::tensor index) {
#ifndef NDEBUG
    uint32_t bPitch, fPitch, xPitch, yPitch;
    switch (memLayout.format) {
    case cldnn::format::yxfb:
        bPitch = 1;
        fPitch = memLayout.size.batch[0] * bPitch;
        xPitch = memLayout.size.feature[0] * fPitch;
        yPitch = memLayout.size.spatial[1] * xPitch;
        return (index.batch[0] * bPitch)
            + (index.feature[0] * fPitch)
            + (index.spatial[1] * xPitch)
            + (index.spatial[0] * yPitch);
        break;
    case cldnn::format::bfyx:
        xPitch = 1;
        yPitch = memLayout.size.spatial[1] * xPitch;
        fPitch = memLayout.size.spatial[0] * yPitch;
        bPitch = memLayout.size.feature[0] * fPitch;
        return (index.batch[0] * bPitch)
            + (index.feature[0] * fPitch)
            + (index.spatial[1] * xPitch)
            + (index.spatial[0] * yPitch);
        break;
    default:
        assert(0);
        return 0;
    }
#endif  // NDEBUG
    return 0;
}

void DebugOptions::PrintNetworkOutputs(std::map<cldnn::primitive_id, cldnn::network_output>& outputsMap) const {
#ifndef NDEBUG
    if (!m_bDebugLayerContent && !m_bDebugLayerFormat) {
        return;
    }

    std::chrono::nanoseconds total(0);
    for (auto& layer : outputsMap) {
        std::cout << layer.first << ":\n";
        auto mem = layer.second.get_memory();
        auto layout = mem.get_layout();
        if (m_bDebugLayerFormat) {
            std::string formatName = GetFormatName(layout.format);
            std::string datatypeName = GetDataTypeName(layout.data_type);
            std::cout << "  Layout: ( " <<
                GetDataTypeName(layout.data_type) << ", " <<
                GetFormatName(layout.format) << ", [";
            for (auto s : layout.size.sizes()) {
                std::cout << s << ",";
            }
            std::cout << "] )\n";
        }
        if (m_bDebugLayerContent) {
            DumpSingleOutput(layer.first, outputsMap);
            std::cout << "\n";
        }
    }
#endif  // NDEBUG
}

void DebugOptions::DumpSingleOutput(cldnn::primitive_id name, std::map<cldnn::primitive_id, cldnn::network_output>& outputs, bool bSingleFeatureMap) const {
#ifndef NDEBUG
    if (outputs.find(name) == outputs.end()) {
        std::cout << "Couldn't find output: " << name << std::endl;
        return;
    }

    auto output = outputs.at(name);
    std::cout << name << ":\n";
    auto mem = output.get_memory();
    auto layout = mem.get_layout();
    cldnn::tensor lowerPad = layout.data_padding.lower_size();
    cldnn::tensor upperPad = layout.data_padding.upper_size();
    {   // format
        std::string formatName = GetFormatName(layout.format);
        std::string datatypeName = GetDataTypeName(layout.data_type);
        std::cout << "  Layout: ( " <<
            GetDataTypeName(layout.data_type) << ", " <<
            GetFormatName(layout.format) << ", [";
        for (auto s : layout.size.sizes()) {
            std::cout << s << ",";
        }
        std::cout << "] [";
        for (auto p : layout.data_padding.lower_size().sizes()) {
            std::cout << p << ",";
        }
        std::cout << "] [";
        for (auto p : layout.data_padding.upper_size().sizes()) {
            std::cout << p << ",";
        }
        std::cout << "] )\n";
    }
    {   // content
        switch (layout.format) {
        case cldnn::format::bfyx:
        {
            std::vector<size_t> pitches;
            size_t elements = 1;
            if (bSingleFeatureMap) {
                elements = layout.size.spatial[1] * layout.size.spatial[0];
            } else {
                for (int i = 0; i < 4; i++) {
                    elements *= layout.size.sizes()[i] + lowerPad.sizes()[i] + upperPad.sizes()[i];
                }
            }
            pitches.push_back(layout.size.spatial[0] + lowerPad.spatial[0] + upperPad.spatial[0]);  // x or width - rowpitch
            pitches.push_back(pitches[0] * (layout.size.spatial[1] + lowerPad.spatial[1] + upperPad.spatial[1]));  // slice pitch
            pitches.push_back(pitches[0] * pitches[1] * layout.size.feature[0]);  // depth/feature pitch
            if (layout.data_type == cldnn::data_types::f32)
                DumpElementsRaw<float>(mem, pitches, elements);
            else
                DumpElementsRaw<uint16_t>(mem, pitches, elements);
            break;
        }
        default:
            assert(0);  // unhandled format
            return;
        }
        std::cout << "\n";
    }
#endif  // NDEBUG
}

void DebugOptions::AddTimedEvent(std::string eventName, std::string startingAt) {
#ifdef _PLUGIN_PERF_PRINTS
    m_TimedEventTimestamp[eventName] = std::chrono::steady_clock::now();
    if (startingAt.compare(std::string()) == 0) {
        startingAt = eventName;
    }
    m_TimedEventStart[eventName] = startingAt;
#endif  // _PLUGIN_PERF_PRINTS
}

void DebugOptions::PrintTimedEvents() {
#ifdef _PLUGIN_PERF_PRINTS
    for (auto& e : m_TimedEventStart) {
        if (e.first.compare(e.second)) {
            std::cout << "[Plugin Internal Metric]: \t" << e.first << " took: " <<
                std::chrono::duration_cast<std::chrono::duration<double, std::chrono::milliseconds::period>>
                (m_TimedEventTimestamp[e.first] - m_TimedEventTimestamp[e.second]).count() << " ms\n";
        }
    }
#endif  // _PLUGIN_PERF_PRINTS
}

void DebugOptions::ClearTimedEvents() {
#ifdef _PLUGIN_PERF_PRINTS
    m_TimedEventStart.clear();
    m_TimedEventTimestamp.clear();
#endif  // _PLUGIN_PERF_PRINTS
}

void DebugOptions::EnableWA(std::string name) {
#ifndef NDEBUG
    m_workaroundNames.insert(name);
#endif  // NDEBUG
}

void DebugOptions::DisableWA(std::string name) {
#ifndef NDEBUG
    m_workaroundNames.erase(name);
#endif  // NDEBUG
}

bool DebugOptions::IsWAActive(std::string name) {
#ifndef NDEBUG
    return (m_workaroundNames.find(name) != m_workaroundNames.end());
#else
    return false;
#endif  // NDEBUG
}

std::string DebugOptions::IELayoutToString(InferenceEngine::Layout layout) {
    switch (layout) {
    case InferenceEngine::ANY: return "ANY";
    case InferenceEngine::NCHW: return "NCHW";
    case InferenceEngine::NHWC: return "NHWC";
    case InferenceEngine::OIHW: return "OIHW";
    case InferenceEngine::C: return "C";
    case InferenceEngine::CHW: return "CHW";
    case InferenceEngine::HW: return "HW";
    case InferenceEngine::NC: return "NC";
    default: return "Unknown";
    }
}

};  // namespace CLDNNPlugin