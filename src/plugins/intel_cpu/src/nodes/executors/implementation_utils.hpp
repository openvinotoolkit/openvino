// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <optional>
#include <vector>

#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_format_filter.hpp"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/dnnl/dnnl_fullyconnected.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"

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
size_t dstRank(const Config& config) {
    return rank<Config, ARG_DST>(config);
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
    return config.attrs.postOps.size();
}

template <typename PostOpType>
[[maybe_unused]] static inline bool hasPostOp(const PostOps& postOps) {
    return any_of(postOps.begin(), postOps.end(), [](const auto& postOp) {
        return postOp.type() == typeid(PostOpType);
    });
}

template <typename Attrs>
struct HasNoOptimalConfig {
    std::optional<executor::Config<Attrs>> operator()([[maybe_unused]] const executor::Config<Attrs>& attrs) const {
        return {};
    }
};

template <typename Attrs>
struct SupportsAnyConfig {
    bool operator()([[maybe_unused]] const executor::Config<Attrs>& attrs) const {
        return true;
    }
};

// Allows to express 'shape agnostic' intention in a more verbose way
template <typename Attrs>
static const inline typename ExecutorImplementation<Attrs>::AcceptsShapePredicate AcceptsAnyShape{};

template <typename Primitive, typename Attrs>
struct CreateDefault {
    ExecutorPtr operator()(const Attrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) const {
        return std::make_shared<Primitive>(attrs, memory, context);
    }
};

template <typename ExecutorT, typename Attrs, typename ShapeAgnosticData>
class DefaultInstantiator {
public:
    std::shared_ptr<ExecutorT> operator()(const MemoryArgs& memory,
                                          const Attrs& attrs,
                                          const ExecutorContext::CPtr context,
                                          const std::shared_ptr<ShapeAgnosticData>& shapeAgnosticData) {
        return ExecutorT::create(memory, attrs, context, shapeAgnosticData);
    }
};

template <typename Primitive,
          typename Attrs,
          typename ShapeAgnosticData = DnnlShapeAgnosticData,
          typename Instantiator = DefaultInstantiator<Primitive, Attrs, ShapeAgnosticData>>
struct CreateDnnlDefault {
    CreateDnnlDefault(bool cacheWeights, bool fc3Das2D) : m_cacheWeights(cacheWeights), m_fc3Das2D(fc3Das2D) {}
    CreateDnnlDefault() = default;

    ExecutorPtr operator()(const Attrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) const {
        return std::make_shared<DnnlExecutor<Primitive, Attrs, DnnlShapeAgnosticData, Instantiator>>(attrs,
                                                                                                     memory,
                                                                                                     context,
                                                                                                     m_cacheWeights,
                                                                                                     m_fc3Das2D);
    }

private:
    bool m_cacheWeights = false;
    // WA for dnnl fullyconnected primitive
    bool m_fc3Das2D = false;
};

template <typename Attrs>
std::optional<executor::Config<Attrs>> createOptimalConfigCommon(const executor::Config<Attrs>& config,
                                                                 const TypeMapping& typeMapping,
                                                                 const std::vector<LayoutType>& layoutConfig,
                                                                 const MappingNotation& notation) {
    // @todo lambdas inside a template function can potentially increase binary size
    auto fullyMatchConfiguration = [](const MemoryDescArgs& currentDescriptors,
                                      const TypeOfArg& typeConfig,
                                      const std::vector<LayoutType>& layoutConfig,
                                      const MappingNotation& notation) {
        return std::all_of(currentDescriptors.begin(), currentDescriptors.end(), [&](const auto& entry) {
            const auto& [argId, desc] = entry;

            if (!notation.count(argId)) {
                return true;
            }

            const int i = notation.at(argId);
            const auto type = typeConfig.at(argId);

            if (desc->empty()) {
                return true;  // empty descriptor is considered as a match
            }

            if (desc->getPrecision() != type) {
                return false;  // type mismatch
            }

            if (desc->getShape().getRank() == 1) {
                return true;  // rank 1 tensors are always ncsp
            }

            return desc->hasLayoutType(layoutConfig[i]);
        });
    };

    auto createOptimalDescriptors = [](const MemoryDescArgs& currentDescriptors,
                                       const TypeOfArg& typeConfig,
                                       const std::vector<LayoutType>& layoutConfig,
                                       const MappingNotation& notation) {
        MemoryDescArgs descs = currentDescriptors;

        const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
        for (const auto& [argId, desc] : currentDescriptors) {
            if (desc->empty()) {
                continue;  // skip empty descriptors
            }

            if (!notation.count(argId)) {
                continue;  // skip if argId is not in notation
            }

            const size_t i = notation.at(argId);
            const auto& type = typeConfig.at(argId);
            const auto& layout = layoutConfig[i];

            if (desc->getPrecision() == type && desc->hasLayoutType(layout)) {
                continue;  // already optimal
            }

            if (desc->getShape().getRank() < 2) {  // rank 1 tensors are always ncsp
                descs[argId] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(type, desc->getShape());
                continue;
            }

            descs[argId] = creatorsMap.at(layout)->createSharedDesc(type, desc->getShape());
        }

        return descs;
    };

    const auto typeConfig = getTypeConfiguration(config.descs, typeMapping, notation);

    if (fullyMatchConfiguration(config.descs, typeConfig, layoutConfig, notation)) {
        return {};
    }

    const auto optimalDescriptors = createOptimalDescriptors(config.descs, typeConfig, layoutConfig, notation);

    return std::optional<executor::Config<Attrs>>(executor::Config<Attrs>{optimalDescriptors, config.attrs});
}

inline MemoryDescArgs memoryDescsFromMemory(const MemoryArgs& memory) {
    MemoryDescArgs memoryDescs;
    memoryDescs.reserve(memory.size());

    for (const auto& mem : memory) {
        memoryDescs[mem.first] = mem.second->getDescPtr();
    }

    return memoryDescs;
}

template <typename Attrs>
executor::Config<Attrs> createConfig(const MemoryArgs& memory, const Attrs& attrs) {
    return executor::Config<Attrs>{memoryDescsFromMemory(memory), attrs};
}

inline bool MatchesMemoryFormatFilter(const MemoryDescArgs& descs,
                                      const std::vector<LayoutType>& layoutConfig,
                                      const MemoryFormatFilter& filter,
                                      const MappingNotation& notation) {
    for (const auto& [argId, desc] : descs) {
        if (argId == ARG_DST) {
            continue;  // skip output
        }

        if (desc->empty()) {
            continue;
        }

        const size_t filterId = argId - ARG_SRC;
        if (filterId >= filter.input.size()) {
            continue;  // no filter for this input
        }

        if (desc->getShape().getRank() > 1) {
            continue;  // rank 1 tensors are always ncsp
        }

        if (desc->getShape().getMinDims()[1] <= 1) {
            continue;
        }

        const auto dnnlDesc =
            DnnlBlockedMemoryDesc(desc->getShape(), dnnl::memory::data_type::f32, filter.input[filterId]);

        if (!notation.count(argId)) {
            continue;  // skip if argId is not in notation
        }

        const size_t i = notation.at(argId);

        if (!dnnlDesc.hasLayoutType(layoutConfig[i])) {
            return false;
        }
    }

    if (filter.output.empty()) {
        return true;
    }

    const auto desc =
        DnnlBlockedMemoryDesc(descs.at(ARG_DST)->getShape(), dnnl::memory::data_type::f32, filter.output.front());
    return desc.hasLayoutType(layoutConfig.back());
}

}  // namespace ov::intel_cpu
