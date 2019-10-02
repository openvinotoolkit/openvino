// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <queue>
#include <stack>

#include <vpu/utils/extra.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/handle.hpp>
#include <vpu/utils/attributes_map.hpp>
#include <vpu/utils/range.hpp>
#include <vpu/utils/containers.hpp>

namespace vpu {

//
// VPU_DEFINE_MODEL_TYPES
//

#define VPU_DEFINE_MODEL_TYPES(type, postfix)                                                       \
    using type = Handle<VPU_COMBINE(type, postfix)>;                                                \
    \
    using VPU_COMBINE(type, Vector) = SmallVector<type>;                                            \
    \
    using VPU_COMBINE(type, ListNode) = IntrusiveHandleListNode<VPU_COMBINE(type, postfix)>;        \
    using VPU_COMBINE(type, List) = IntrusiveHandleList<VPU_COMBINE(type, postfix)>;                \
    \
    using VPU_COMBINE(type, Set) = HandleSet<VPU_COMBINE(type, postfix)>;                           \
    \
    template <typename Val>                                                                         \
    using VPU_COMBINE(type, Map) = HandleMap<VPU_COMBINE(type, postfix), Val>;                      \
    \
    using VPU_COMBINE(type, Ptr) = std::shared_ptr<VPU_COMBINE(type, postfix)>;                     \
    \
    using VPU_COMBINE(type, PtrList) = std::list<VPU_COMBINE(type, Ptr)>;

//
// VPU_MODEL_ATTRIBUTE
//

#define VPU_MODEL_ATTRIBUTE(type, name, defVal)                                 \
    private:                                                                    \
        type VPU_COMBINE(_, name) = defVal;                                     \
    public:                                                                     \
        inline const type& name() const {                                       \
            return VPU_COMBINE(_, name);                                        \
        }

#define VPU_MODEL_ATTRIBUTE_PTR_RANGE(type, name)                                   \
    private:                                                                      \
        type VPU_COMBINE(_, name);                                                  \
    public:                                                                         \
        inline auto name() const -> decltype(VPU_COMBINE(_, name) | asRange()) {    \
            return VPU_COMBINE(_, name) | asRange();                                \
        }

//
// Forward declaration
//

class GraphTransformerImpl;

class Model;
using ModelPtr = std::shared_ptr<Model>;

class DataNode;
VPU_DEFINE_MODEL_TYPES(Data, Node)

class StageNode;
VPU_DEFINE_MODEL_TYPES(Stage, Node)

class StageInputEdge;
VPU_DEFINE_MODEL_TYPES(StageInput, Edge)

class StageOutputEdge;
VPU_DEFINE_MODEL_TYPES(StageOutput, Edge)

class StageTempBufferEdge;
VPU_DEFINE_MODEL_TYPES(StageTempBuffer, Edge)

class SharedAllocationEdge;
VPU_DEFINE_MODEL_TYPES(SharedAllocation, Edge)

class InjectedStageEdge;
VPU_DEFINE_MODEL_TYPES(InjectedStage, Edge)

}  // namespace vpu
