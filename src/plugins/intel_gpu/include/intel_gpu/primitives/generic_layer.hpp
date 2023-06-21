// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include <vector>

namespace cldnn {

struct WeightsReorderParams {
    WeightsReorderParams(layout in_layout, layout out_layout) : _in_layout(in_layout), _out_layout(out_layout) {}

    virtual size_t hash() const {
        return hash_combine(_in_layout.hash(), _out_layout.hash());
    }

    virtual bool operator==(const WeightsReorderParams& rhs) const {
        if (typeid(*this) != typeid(rhs))
            return false;

        return _in_layout == rhs._in_layout &&
               _out_layout == rhs._out_layout;
    }

    layout get_input_layout() const { return _in_layout; }
    layout get_output_layout() const { return _out_layout; }

    virtual ~WeightsReorderParams() = default;

protected:
    layout _in_layout;
    layout _out_layout;
};

/// @brief Changes how data is ordered in memory. Value type is not changed & all information is preserved.
/// @details Corresponding values are bitwise equal before/after reorder.
struct generic_layer : public primitive_base<generic_layer> {
    CLDNN_DECLARE_PRIMITIVE(generic_layer)

    generic_layer() : primitive_base("", {}) {}

    DECLARE_OBJECT_TYPE_SERIALIZATION

    /// @brief Constructs generic_layer primitive which takes mean subtract values from another primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param mean Primitive id to get mean subtract values.
    generic_layer(const primitive_id& id,
                  const primitive_id& input,
                  std::shared_ptr<WeightsReorderParams> params,
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}), params(params) {}

    std::shared_ptr<WeightsReorderParams> params;

    size_t hash() const override {
        size_t seed = primitive::hash();

        if (params)
            seed = hash_combine(seed, params->hash());

        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const generic_layer>(rhs);

        if ((params == nullptr) != (rhs_casted.params == nullptr))
            return false;

        if (params != nullptr)
            return *params == *rhs_casted.params;

        return true;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<generic_layer>::save(ob);
        ob << params->get_input_layout();
        ob << params->get_output_layout();
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<generic_layer>::load(ib);
        layout input_layout, output_layout;
        ib >> input_layout;
        ib >> output_layout;
        params = std::make_shared<WeightsReorderParams>(input_layout, output_layout);
    }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override { return {}; }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
