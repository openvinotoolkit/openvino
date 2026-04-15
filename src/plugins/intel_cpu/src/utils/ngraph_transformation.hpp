// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <bitset>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#ifdef CPU_DEBUG_CAPS

#    include <openvino/pass/manager.hpp>
#    include <openvino/pass/serialize.hpp>
#    include <openvino/pass/visualize_tree.hpp>

#    include "debug_caps_config.h"
#    include "openvino/util/file_util.hpp"
#    include "utils/platform.h"

namespace ov::intel_cpu {

class TransformationDumper {
public:
    explicit TransformationDumper(const DebugCapsConfig& config,
                                  const DebugCapsConfig::TransformationFilter::Type type,
                                  const std::shared_ptr<ov::Model>& model)
        : config(config),
          model(model),
          type(type) {
        for (auto prev = infoMap.at(type).prev; prev != TransformationType::NumOfTypes; prev = infoMap.at(prev).prev) {
            // no need to serialize input graph if there was no transformations from previous dump
            if (config.disable.transformations.filter[prev]) {
                continue;
            }
            if (!config.dumpIR.transformations.filter[prev]) {
                break;
            }
            if (wasDumped(model->get_friendly_name())[prev]) {
                return;
            }
        }
        dump("_in");
    }
    ~TransformationDumper() {
        dump("_out");
        wasDumped(model->get_friendly_name()).set(type);
    }

private:
    const DebugCapsConfig& config;
    const std::shared_ptr<ov::Model>& model;
    using TransformationType = DebugCapsConfig::TransformationFilter::Type;
    const TransformationType type;

    struct TransformationInfo {
        std::string name;
        TransformationType prev;
    };
    const std::unordered_map<TransformationType, TransformationInfo> infoMap = {
        {TransformationType::PreLpt, {"preLpt", TransformationType::NumOfTypes}},
        {TransformationType::Lpt, {"lpt", TransformationType::PreLpt}},
        {TransformationType::PostLpt, {"postLpt", TransformationType::Lpt}},
        {TransformationType::Snippets, {"snippets", TransformationType::PostLpt}},
        {TransformationType::Specific, {"cpuSpecific", TransformationType::Snippets}}};
    static std::bitset<TransformationType::NumOfTypes>& wasDumped(const std::string& modelName) {
        static std::unordered_map<std::string, std::bitset<TransformationType::NumOfTypes>> wasDumpedPerModel;
        return wasDumpedPerModel[modelName];
    }

    void dump(const std::string&& postfix) {
        static std::unordered_map<std::string, int> numPerModel;
        const std::filesystem::path dir{config.dumpIR.dir};
        // include model name to a path so more than one model can be dumped without overriding dumps of each other
        const std::filesystem::path dumpDir{dir / model->get_friendly_name()};
        // add a serial number to the prefix to ensure the correct order in 'ls' output
        auto& num = numPerModel[model->get_friendly_name()];
        const std::filesystem::path irFileName{"ir_" + std::to_string(num) + '_' + infoMap.at(type).name + postfix};
        // fullPath example: intel_cpu_dump/<model_name>/ir_0_preLpt_in.xml
        const auto fullPath = dumpDir / irFileName;

        ov::util::create_directory_recursive(dumpDir);

        ov::pass::Manager serializer;

        if (config.dumpIR.format.filter[DebugCapsConfig::IrFormatFilter::XmlBin]) {
            auto xmlPath = fullPath;
            xmlPath.replace_extension(".xml");
            serializer.register_pass<ov::pass::Serialize>(xmlPath, std::filesystem::path{});
        }

        if (config.dumpIR.format.filter[DebugCapsConfig::IrFormatFilter::Xml]) {
            auto xmlFile = fullPath;
            xmlFile.replace_extension(".xml");

            serializer.register_pass<ov::pass::Serialize>(xmlFile, std::filesystem::path{NULL_STREAM});
        }

        if (config.dumpIR.format.filter[DebugCapsConfig::IrFormatFilter::Svg]) {
            auto svgFile = fullPath;
            svgFile.replace_extension(".svg");
            serializer.register_pass<ov::pass::VisualizeTree>(svgFile);
        }

        if (config.dumpIR.format.filter[DebugCapsConfig::IrFormatFilter::Dot]) {
            auto dotFile = fullPath;
            dotFile.replace_extension(".dot");
            serializer.register_pass<ov::pass::VisualizeTree>(dotFile);
        }

        serializer.run_passes(model);
        num++;
    }
};

}  // namespace ov::intel_cpu

// 'EXPAND' wrapper is necessary to ensure __VA_ARGS__ behaves the same on all the platforms
#    define CPU_DEBUG_CAP_EXPAND(x) x
#    define CPU_DEBUG_CAP_IS_TRANSFORMATION_DISABLED(_config, _type) \
        _config.disable.transformations.filter[DebugCapsConfig::TransformationFilter::Type::_type]
#    define CPU_DEBUG_CAP_IS_TRANSFORMATION_ENABLED(...) \
        CPU_DEBUG_CAP_EXPAND(!CPU_DEBUG_CAP_IS_TRANSFORMATION_DISABLED(__VA_ARGS__))
#    define CPU_DEBUG_CAP_TRANSFORMATION_DUMP(_this, _type)                                                     \
        OPENVINO_ASSERT(CPU_DEBUG_CAP_IS_TRANSFORMATION_ENABLED((_this)->config.debugCaps, _type));             \
        auto dumperPtr = (_this)->config.debugCaps.dumpIR.transformations                                       \
                                 .filter[DebugCapsConfig::TransformationFilter::Type::_type]                    \
                             ? std::unique_ptr<TransformationDumper>(                                           \
                                   new TransformationDumper((_this)->config.debugCaps,                          \
                                                            DebugCapsConfig::TransformationFilter::Type::_type, \
                                                            (_this)->model))                                    \
                             : nullptr
#    define CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(_this, _type)                            \
        if (CPU_DEBUG_CAP_IS_TRANSFORMATION_DISABLED((_this)->config.debugCaps, _type)) \
            return;                                                                     \
        CPU_DEBUG_CAP_TRANSFORMATION_DUMP(_this, _type)
#else
#    define CPU_DEBUG_CAP_IS_TRANSFORMATION_DISABLED(_config, _type) false
#    define CPU_DEBUG_CAP_IS_TRANSFORMATION_ENABLED(...)             true
#    define CPU_DEBUG_CAP_TRANSFORMATION_DUMP(_this, _type)
#    define CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(_this, _type)
#endif  // CPU_DEBUG_CAPS
