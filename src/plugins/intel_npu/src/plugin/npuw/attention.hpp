// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "util.hpp" // fill_tensor
#include "logging.hpp"  // NPUW_ASSERT
#include "openvino/openvino.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"  // get_tensor_impl

namespace ov {
namespace npuw {

namespace function {

// Partition-time attention information. So far assume dynamic execution in 1 dimension only
// Defined at this level to be aligned with other partitioning entities (but needs to be moved?)
struct Attention {
    using PPtr = std::shared_ptr<ov::op::v0::Parameter>;
    struct Param {
        PPtr param;
        std::size_t dim;
    };
    std::vector<Param> _inputs;

    // FIXME: This may be way too specific for a generic dynamic block,
    // as it reflects the attention input here.
    PPtr _mask;
    ov::Shape _mask_shape;

    std::size_t query_len() const {
        // Put the mask's innermost dimension dynamic
        NPUW_ASSERT(_mask_shape.size() == 4);
        return _mask_shape.at(2);
    }

    std::size_t context_len() const {
        // Put the mask's innermost dimension dynamic
        NPUW_ASSERT(_mask_shape.size() == 4);
        return _mask_shape.at(3);
    }
};

}  // namespace function

namespace compiled {

// Compile-time attention information. Not much different from the above
struct Attention {
    std::size_t query_size = 0u;
    std::size_t context_size = 0u;

    struct Param {
        std::size_t idx;  // function input index for this spatial parameter
        std::size_t dim;
    };
    std::vector<Param> params;
    std::size_t mask_idx = 0u;

    ov::Tensor attend_all;

    Attention() = default;
    Attention(const function::Attention& d, const std::shared_ptr<ov::Model>& m)
        : query_size(d.query_len())
        , context_size(d.context_len()) {
        for (auto&& input : d._inputs) {
            std::size_t p_idx = m->get_parameter_index(input.param);
            params.push_back(Param{p_idx, input.dim});
        }
        mask_idx = m->get_parameter_index(d._mask);

        // Create a mask data tensor here. Technically it is a runtime parameter,
        // but in fact it doesn't change in runtime (we ignore what user passes us!)
        // FIXME: Probably it is wrong
        const auto mask_type = d._mask->get_element_type();
        attend_all = ov::Tensor(mask_type, d._mask_shape);

        switch (mask_type) {
        case ov::element::f16:
            ov::npuw::util::fill_tensor<ov::float16>(ov::get_tensor_impl(attend_all), 0);
            break;
        case ov::element::f32:
            ov::npuw::util::fill_tensor<float>(ov::get_tensor_impl(attend_all), 0);
            break;
        default:
            OPENVINO_THROW("Dynamic attenion mask type ", mask_type, " is not supported yet");
        }
    }
};

}  // namespace compiled

namespace runtime {
namespace attention {

// A base class to decide the work-scope from some feature
class Selector {
public:
    enum class Case { PREFILL, GENERATE, UNKNOWN };

    using Ptr = std::shared_ptr<Selector>;
    virtual ~Selector() = default;
    virtual void prepare() = 0;
    virtual int64_t length() const = 0;
    virtual int64_t past_length() const = 0;

    Case this_case() const {
        return m_case;
    }

protected:
    Case m_case = Case::UNKNOWN;
};

// No dynamic dispatch - just run over the whole range
class All final : public Selector {
    void prepare() override {}
    int64_t length() const override {
        return -1;
    }
    int64_t past_length() const override {
        OPENVINO_NOT_IMPLEMENTED; // And shouldn't be here
    }
};

// Define work scope based on position ids
class PositionIDs final : public Selector {
    std::size_t m_position_ids_idx = 0u;
    int64_t m_current_length = 0;
    int64_t m_past_length = 0;

    const compiled::Attention& m_d;
    const ov::ISyncInferRequest& m_rq;

    PositionIDs(std::size_t param_idx, const compiled::Attention& d, const ov::ISyncInferRequest& rq);
    void prepare() override;
    int64_t length() const override;
    int64_t past_length() const override;

public:
    static Selector::Ptr find(const compiled::Attention& d, const ov::ISyncInferRequest& rq);
};

}  // namespace attention
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
