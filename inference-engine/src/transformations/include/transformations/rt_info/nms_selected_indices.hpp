// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>

namespace ov {

class TRANSFORMATIONS_API NmsSelectedIndices {
public:
    NmsSelectedIndices() = default;
};

TRANSFORMATIONS_API bool has_nms_selected_indices(const Node * node);

TRANSFORMATIONS_API void set_nms_selected_indices(Node * node);

extern template class TRANSFORMATIONS_API VariantImpl<NmsSelectedIndices>;

template<>
class TRANSFORMATIONS_API VariantWrapper<NmsSelectedIndices> : public VariantImpl<NmsSelectedIndices> {
public:
    static constexpr VariantTypeInfo type_info{"NMS_SELECTED_INDICES", 0};

    const VariantTypeInfo &get_type_info() const override {
        return type_info;
    }

    VariantWrapper() = default;

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}

    bool is_copyable() const override { return false; }
};

}  // namespace ov
