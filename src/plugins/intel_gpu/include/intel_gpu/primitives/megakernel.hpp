// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/op/megakernel.hpp"
#include "primitive.hpp"

namespace cldnn {

struct megakernel : public primitive_base<megakernel> {
    CLDNN_DECLARE_PRIMITIVE(megakernel)

    megakernel() : primitive_base("", {}) {}

    megakernel(const primitive_id& id, const std::vector<input_info>& inputs, const ov::intel_gpu::op::MegaKernelAttrs& attrs)
        : primitive_base(id, inputs, attrs.outputs.size()),
          attrs(attrs) {}

    ov::intel_gpu::op::MegaKernelAttrs attrs;

    size_t hash() const override {
        size_t seed = primitive::hash();
        for (const auto& out : attrs.outputs) {
            seed = hash_combine(seed, out.type.hash());
            for (const auto& d : out.shape) {
                seed = hash_combine(seed, d.input);
                seed = hash_combine(seed, d.axis);
                seed = hash_combine(seed, d.value);
            }
        }
        return hash_range(seed, attrs.params.begin(), attrs.params.end());
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        return attrs == downcast<const megakernel>(rhs).attrs;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<megakernel>::save(ob);
        ob << attrs.outputs.size();
        for (const auto& out : attrs.outputs) {
            const ov::element::Type_t type = out.type;
            ob << make_data(&type, sizeof(type));
            ob << out.shape.size();
            for (const auto& d : out.shape)
                ob << d.input << d.axis << d.value;
        }
        ob << attrs.params;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<megakernel>::load(ib);
        size_t num_outputs = 0;
        ib >> num_outputs;
        attrs.outputs.resize(num_outputs);
        for (auto& out : attrs.outputs) {
            ov::element::Type_t type = ov::element::Type_t::dynamic;
            ib >> make_data(&type, sizeof(type));
            out.type = type;
            size_t rank = 0;
            ib >> rank;
            out.shape.resize(rank);
            for (auto& d : out.shape)
                ib >> d.input >> d.axis >> d.value;
        }
        ib >> attrs.params;
    }
};

}  // namespace cldnn
