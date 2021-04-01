// Copyright (C) 2018-2020 Intel Corporation
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
#include <vpu/utils/error.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/handle.hpp>
#include <vpu/utils/attributes_map.hpp>
#include <vpu/utils/range.hpp>
#include <vpu/utils/small_vector.hpp>
#include <vpu/utils/intrusive_handle_list.hpp>

namespace vpu {

//
// VPU_DEFINE_HANDLE_TYPES
//

//
// <BaseName><Postfix> - actual object, stored inside the Model Graph container, accessed only via Handle(s)
//

#define VPU_DEFINE_HANDLE_TYPES(BaseName, Postfix)                                                      \
    class VPU_COMBINE(BaseName, Postfix);                                                               \
    \
    using BaseName = Handle<VPU_COMBINE(BaseName, Postfix)>;                                            \
    \
    using VPU_COMBINE(BaseName, Vector) = SmallVector<BaseName>;                                        \
    \
    using VPU_COMBINE(BaseName, ListNode) = IntrusiveHandleListNode<VPU_COMBINE(BaseName, Postfix)>;    \
    using VPU_COMBINE(BaseName, List) = IntrusiveHandleList<VPU_COMBINE(BaseName, Postfix)>;            \
    \
    using VPU_COMBINE(BaseName, Set) = HandleSet<VPU_COMBINE(BaseName, Postfix)>;                       \
    \
    template <typename Val>                                                                             \
    using VPU_COMBINE(BaseName, Map) = HandleMap<VPU_COMBINE(BaseName, Postfix), Val>;

//
// VPU_DEFINE_*_PTR_TYPES
//

#define VPU_DEFINE_UNIQUE_PTR_TYPES(BaseName, Postfix)                                              \
    using VPU_COMBINE(BaseName, Ptr) = std::unique_ptr<VPU_COMBINE(BaseName, Postfix)>;             \
    \
    using VPU_COMBINE(BaseName, PtrList) = std::list<VPU_COMBINE(BaseName, Ptr)>;

#define VPU_DEFINE_SHARED_PTR_TYPES(BaseName, Postfix)                                              \
    using VPU_COMBINE(BaseName, Ptr) = std::shared_ptr<VPU_COMBINE(BaseName, Postfix)>;             \
    \
    using VPU_COMBINE(BaseName, PtrList) = std::list<VPU_COMBINE(BaseName, Ptr)>;

//
// VPU_MODEL_ATTRIBUTE
//

#define VPU_MODEL_ATTRIBUTE(AttrType, name, defVal)                             \
    private:                                                                    \
        AttrType VPU_COMBINE(_, name) = defVal;                                 \
    public:                                                                     \
        inline const AttrType& name() const {                                   \
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

VPU_DEFINE_HANDLE_TYPES(Model, Obj)
VPU_DEFINE_SHARED_PTR_TYPES(Model, Obj)

VPU_DEFINE_HANDLE_TYPES(Data, Node)
VPU_DEFINE_SHARED_PTR_TYPES(Data, Node)

VPU_DEFINE_HANDLE_TYPES(Stage, Node)
VPU_DEFINE_SHARED_PTR_TYPES(Stage, Node)

VPU_DEFINE_HANDLE_TYPES(StageInput, Edge)
VPU_DEFINE_SHARED_PTR_TYPES(StageInput, Edge)

VPU_DEFINE_HANDLE_TYPES(StageOutput, Edge)
VPU_DEFINE_SHARED_PTR_TYPES(StageOutput, Edge)

VPU_DEFINE_HANDLE_TYPES(StageDependency, Edge)
VPU_DEFINE_SHARED_PTR_TYPES(StageDependency, Edge)

VPU_DEFINE_HANDLE_TYPES(StageTempBuffer, Edge)
VPU_DEFINE_SHARED_PTR_TYPES(StageTempBuffer, Edge)

VPU_DEFINE_HANDLE_TYPES(DataToDataAllocation, Edge)
VPU_DEFINE_SHARED_PTR_TYPES(DataToDataAllocation, Edge)

VPU_DEFINE_HANDLE_TYPES(DataToShapeAllocation, Edge)
VPU_DEFINE_SHARED_PTR_TYPES(DataToShapeAllocation, Edge)

VPU_DEFINE_HANDLE_TYPES(Injection, Edge)
VPU_DEFINE_SHARED_PTR_TYPES(Injection, Edge)

}  // namespace vpu
