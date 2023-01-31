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
    virtual ~WeightsReorderParams() = default;
    virtual size_t hash() const { return hash_combine(_in_layout.hash(), _out_layout.hash()); }
    layout get_input_layout() const { return _in_layout; }
    layout get_output_layout() const { return _out_layout; }

protected:
    layout _in_layout;
    layout _out_layout;
};

/// @brief Changes how data is ordered in memory. Value type is not changed & all information is preserved.
/// @details Corresponding values are bitwise equal before/after reorder.
struct generic_layer : public primitive_base<generic_layer> {
    CLDNN_DECLARE_PRIMITIVE(generic_layer)

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
        seed = hash_combine(seed, generic_params.engine);

        if (generic_params.cpuKernel != nullptr) {
            auto& cpuKernel = generic_params.cpuKernel;
            seed = hash_combine(seed, cpuKernel->GetExpectedInputLayout());
            seed = hash_combine(seed, cpuKernel->GetExpectedInputType());
        }

        if (generic_params.clKernel != nullptr) {
            auto& clKernel = generic_params.clKernel;
            seed = hash_combine(seed, clKernel->skip_execution);

            auto& gws = clKernel->params.workGroups.global;
            seed = hash_range(seed, gws.begin(), gws.end());

            auto& lws = clKernel->params.workGroups.local;
            seed = hash_range(seed, lws.begin(), lws.end());

            auto& arguments = clKernel->params.arguments;
            for (auto& args : arguments) {
                seed = hash_combine(seed, args.index);
                seed = hash_combine(seed, args.t);
            }

            auto& scalars = clKernel->params.scalars;
            for (auto& s : scalars) {
                seed = hash_combine(seed, s.t);
            }

            seed = hash_combine(seed, clKernel->code.kernelString->get_hash());
        }
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const generic_layer>(rhs);

        if (generic_params.engine != rhs_casted.generic_params.engine)
            return false;

        if (generic_params.cpuKernel != nullptr) {
            if (generic_params.cpuKernel->GetExpectedInputLayout() != rhs_casted.generic_params.cpuKernel->GetExpectedInputLayout())
                return false;

            if (generic_params.cpuKernel->GetExpectedInputType() != rhs_casted.generic_params.cpuKernel->GetExpectedInputType())
                return false;
        }

        if (generic_params.clKernel != nullptr) {
            auto& clKernel = generic_params.clKernel;
            auto& clKernel_rhs = rhs_casted.generic_params.clKernel;
            if (clKernel->skip_execution != clKernel_rhs->skip_execution)
                return false;

            auto& gws       = clKernel->params.workGroups.global;
            auto& gws_rhs   = clKernel_rhs->params.workGroups.global;
            if (gws != gws_rhs)
                return false;

            auto& lws       = clKernel->params.workGroups.local;
            auto& lws_rhs   = clKernel_rhs->params.workGroups.local;
            if (lws != lws_rhs)
                return false;

            auto& arguments     = clKernel->params.arguments;
            auto& arguments_rhs = clKernel_rhs->params.arguments;
            if (arguments.size() != arguments_rhs.size())
                return false;

            for (size_t idx = 0; idx < arguments.size(); idx++) {
                if (arguments[idx].index != arguments_rhs[idx].index)
                    return false;

                if (arguments[idx].t != arguments_rhs[idx].t)
                    return false;
            }

            auto& scalars     = clKernel->params.scalars;
            auto& scalars_rhs = clKernel_rhs->params.scalars;
            if (scalars.size() != scalars_rhs.size())
                return false;

            for (size_t idx = 0; idx < scalars.size(); idx++) {
                if (scalars[idx].t != scalars_rhs[idx].t)
                    return false;
            }

            if (clKernel->code.kernelString->get_str() != clKernel_rhs->code.kernelString->get_str())
                return false;
        }
        return true;
    }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override { return {}; }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
