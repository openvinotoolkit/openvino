// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_fullyconnected.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

template <typename Config, int idx>
ov::element::Type memoryDescType(const Config& config) {
    return config.descs.at(idx)->getPrecision();
}

template <typename Config>
ov::element::Type srcType(const Config& config) {
    return memoryDescType<Config, ARG_SRC>(config);
}

template <typename Config>
ov::element::Type weiType(const Config& config) {
    return memoryDescType<Config, ARG_WEI>(config);
}

template <typename Config>
ov::element::Type biaType(const Config& config) {
    return memoryDescType<Config, ARG_BIAS>(config);
}

template <typename Config, int idx = 0>
ov::element::Type dstType(const Config& config) {
    return memoryDescType<Config, ARG_DST>(config);
}

template <typename Config, int idx>
ov::element::Type dims(const Config& config) {
    return config.descs.at(idx)->getShape().getDims();
}

template <typename Config>
const VectorDims& srcDims(const Config& config) {
    return dims<Config, ARG_SRC>(config);
}

template <typename Config>
const VectorDims& weiDims(const Config& config) {
    return dims<Config, ARG_WEI>(config);
}

template <typename Config, int idx>
size_t rank(const Config& config) {
    return config.descs.at(idx)->getShape().getRank();
}

template <typename Config>
size_t srcRank(const Config& config) {
    return rank<Config, ARG_SRC>(config);
}

template <typename Config>
size_t weiRank(const Config& config) {
    return rank<Config, ARG_WEI>(config);
}

template <typename Config, int idx>
size_t memSize(const Config& config) {
    return config.descs.at(idx)->getCurrentMemSize();
}

template <typename Config>
size_t srcMemSize(const Config& config) {
    return memSize<Config, ARG_SRC>(config);
}

template <typename Config>
size_t weiMemSize(const Config& config) {
    return memSize<Config, ARG_WEI>(config);
}

template <typename Config>
size_t postOpsNumbers(const Config& config) {
    return config.postOps.size();
}

template <typename Attrs>
struct RequiredNoFallback {
    std::optional<executor::Config<Attrs>> operator()(const executor::Config<Attrs>&) const {
        return {};
    }
};

template <typename Attrs>
struct SupportsAnyConfig {
    bool operator()(const executor::Config<Attrs>&) const {
        return true;
    }
};

template <typename Attrs>
struct AcceptsAnyShape {
    bool operator()(const Attrs&, const PostOps&, const MemoryArgs&) const {
        return true;
    }
};

template <typename Primitive, typename Attrs>
struct CreateDefault {
    ExecutorPtr operator()(const Attrs& attrs,
                           const PostOps& postOps,
                           const MemoryArgs& memory,
                           const ExecutorContext::CPtr& context) const {
        return std::make_shared<Primitive>(attrs, postOps, memory, context);
    }
};

template <typename Primitive,
          typename Attrs,
          typename ShapeAgnosticData = DnnlShapeAgnosticData,
          typename Instantiator = DefaultInstantiator<Primitive, Attrs, ShapeAgnosticData>>
struct CreateDnnlDefault {
    ExecutorPtr operator()(const Attrs& attrs,
                           const PostOps& postOps,
                           const MemoryArgs& memory,
                           const ExecutorContext::CPtr& context) const {
        return std::make_shared<DnnlFCExecutor<Primitive, Attrs, DnnlShapeAgnosticData, Instantiator>>(attrs,
                                                                                                       postOps,
                                                                                                       memory,
                                                                                                       context,
                                                                                                       false);
    }
};

template <typename Attrs>
std::optional<executor::Config<Attrs>> requiresFallbackCommon(const executor::Config<Attrs>& config,
                                                              const TypeMapping& typeMapping,
                                                              const std::vector<LayoutType>& layoutConfig,
                                                              const MappingNotation& notation) {
    // @todo lambdas inside a template function can potentially increase binary size
    auto fullyMatchConfiguration = [](const MemoryDescArgs& currentDescriptors,
                                      const InOutTypes& typeConfig,
                                      const std::vector<LayoutType>& layoutConfig,
                                      const MappingNotation& notation) {
        for (size_t i = 0; i < typeConfig.size(); i++) {
            const auto& type = typeConfig[i];
            const auto& desc = currentDescriptors.at(notation[i]);

            if (desc->empty()) {
                continue;
            }

            if (desc->getPrecision() != type) {
                return false;  // type mismatch
            }

            if (desc->getShape().getRank() > 2 && !desc->hasLayoutType(layoutConfig[i])) {
                return false;  // layout mismatch
            }
        }

        return true;
    };

    auto createOptimalDescriptors = [](const MemoryDescArgs& currentDescriptors,
                                       const InOutTypes& typeConfig,
                                       const std::vector<LayoutType>& layoutConfig,
                                       const MappingNotation& notation) {
        MemoryDescArgs descs = currentDescriptors;

        const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
        for (size_t i = 0; i < typeConfig.size(); i++) {
            const auto& desc = currentDescriptors.at(notation[i]);
            const auto& descType = desc->getPrecision();
            const auto& type = typeConfig[i];
            const auto& layout = layoutConfig[i];

            if (desc->empty()) {
                continue;
            }

            if (descType == type && desc->hasLayoutType(layout)) {
                continue;
            }

            if (desc->getShape().getRank() < 2) {
                descs[notation[i]] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(type, desc->getShape());
                continue;
            }

            descs[notation[i]] = creatorsMap.at(layout)->createSharedDesc(type, desc->getShape());
        }

        return descs;
    };

    const auto typeConfig = getTypeConfiguration(config.descs, typeMapping, notation);

    if (fullyMatchConfiguration(config.descs, typeConfig, layoutConfig, notation)) {
        return {};
    }

    const auto optimalDescriptors = createOptimalDescriptors(config.descs, typeConfig, layoutConfig, notation);

    return std::optional<executor::Config<Attrs>>(
        executor::Config<Attrs>{optimalDescriptors, config.attrs, config.postOps});
}

}  // namespace ov::intel_cpu
