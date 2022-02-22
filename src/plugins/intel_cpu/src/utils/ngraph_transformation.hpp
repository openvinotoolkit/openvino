// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#include "config.h"
#include <openvino/pass/serialize.hpp>

namespace ov {
namespace intel_cpu {

class TransformationDumper {
public:
    explicit TransformationDumper(const Config& config, const Config::TransformationFilter::Type type,
                                  const std::shared_ptr<ngraph::Function>& nGraphFunc)
        : config(config), nGraphFunc(nGraphFunc) {
        it = std::find_if(stages.begin(), stages.end(),
                          [type] (const StageDescription& stage) { return stage.type == type; });
        IE_ASSERT(it != stages.end()) << "Unsupported transformation type: " << type;
        if (it != stages.begin()) {
            // no need to serialize input graph if there was no transformations from previous dump
            for (auto prevIt = std::reverse_iterator<StageIt>(it); prevIt < stages.rend(); prevIt++) {
                if (config.disable.transformations.filter[prevIt->type])
                    continue;
                if (!config.dumpIR.transformations.filter[prevIt->type])
                    break;
                if (wasDumped()[prevIt->type])
                    return;
            }
        }
        dump(it, true);
    }
    ~TransformationDumper() {
        dump(it);
        wasDumped().set(it->type);
    }

private:
    const Config& config;
    const std::shared_ptr<ngraph::Function>& nGraphFunc;

    using TransformationType = Config::TransformationFilter::Type;
    struct StageDescription {
        TransformationType type;
        std::string name;
    };
    const std::vector<StageDescription> stages =
        {{ TransformationType::PreLpt,     "preLpt" },
         { TransformationType::Lpt,        "lpt" },
         { TransformationType::PostLpt,    "postLpt" },
         { TransformationType::Snippets,   "snippets" },
         { TransformationType::Specific,   "cpuSpecificOpSet" }};
    using StageIt = std::vector<StageDescription>::const_iterator;
    StageIt it;
    std::bitset<TransformationType::numOfTypes>& wasDumped(void) {
        static std::bitset<TransformationType::numOfTypes> wasDumped;
        return wasDumped;
    }
    void dump(const StageIt& it, const bool isInput = false) {
        const auto xmlPath = config.dumpIR.dir + "/ir_transformation_" +
                             std::to_string(std::distance(stages.begin(), it)) + '_' +
                             it->name + (isInput ? "_input" : "_output")  + ".xml";
        ov::pass::Serialize serializer(xmlPath, "");
        serializer.run_on_model(nGraphFunc);
    }
};

}   // namespace intel_cpu
}   // namespace ov

#  define CPU_DEBUG_CAP_TRANSFORMATION_RETURN_OR_DUMP(_type)                                            \
    if (config.disable.transformations.filter[Config::TransformationFilter::Type::_type])               \
        return;                                                                                         \
    auto dumperPtr = config.dumpIR.transformations.filter[Config::TransformationFilter::Type::_type] ?  \
        std::unique_ptr<TransformationDumper>(new TransformationDumper(config,                          \
                                              Config::TransformationFilter::Type::_type, nGraphFunc)) : \
        nullptr
#else
#  define CPU_DEBUG_CAP_TRANSFORMATION_RETURN_OR_DUMP(_type)
#endif // CPU_DEBUG_CAPS
