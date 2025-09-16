// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "logging.hpp"  // NPUW_ASSERT
#include "openvino/openvino.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"  // get_tensor_impl

namespace ov {
namespace npuw {

namespace function {

// Partition-time dynamic information. So far assume dynamic execution in 1 dimension only
// Defined at this level to be aligned with other partitioning entities (but needs to be moved?)
struct Dynamic {
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
};

}  // namespace function

namespace compiled {

// FIXME: Stolen from LLMInferRequest, should be shared across utils
namespace {
template <typename T>
void fill_tensor(ov::SoPtr<ov::ITensor> tensor, T fill_val, size_t offset = 0u) {
    T* tensor_data = tensor->data<T>();
    std::fill(tensor_data + offset, tensor_data + tensor->get_size(), fill_val);
}

template <typename T>
void prepare_mask(ov::SoPtr<ov::ITensor> tensor) {
    NPUW_ASSERT(tensor->is_continuous());
    fill_tensor<T>(tensor, 0);

    // Initialize the top-right triangle with -inf
    const auto shape = tensor->get_shape();
    NPUW_ASSERT(shape.size() == 4);
    NPUW_ASSERT(shape[0] == 1);
    NPUW_ASSERT(shape[1] == 1);

    const auto q_size = shape[2];
    const auto ctx_size = shape[3];
    for (std::size_t i = 0; i < q_size - 1; i++) {
        T* pRow = tensor->data<T>() + i * ctx_size;
        for (std::size_t j = ctx_size - q_size + i; j < ctx_size; j++) {
            pRow[j] = std::numeric_limits<T>::lowest();
        }
    }

    // for a matrix like below where i = -inf (of type T)
    // <-past -> < q >
    // 000000000 0iiii
    // 000000000 00iii
    // 000000000 000ii
    // 000000000 0000i
    // 000000000 00000
}

}  // anonymous namespace

// Compile-time dynamic information. Not much different from the above
struct Dynamic {
    struct Param {
        std::size_t idx;  // function input index for this spatial parameter
        std::size_t dim;
    };
    std::vector<Param> params;
    std::size_t mask_idx = 0u;

    ov::Tensor mask_data;

    Dynamic() = default;
    Dynamic(const function::Dynamic& s, const std::shared_ptr<ov::Model>& m) {
        for (auto&& input : s._inputs) {
            std::size_t p_idx = m->get_parameter_index(input.param);
            params.push_back(Param{p_idx, input.dim});
        }
        mask_idx = m->get_parameter_index(s._mask);

        // Create a mask data tensor here. Technically it is a runtime parameter,
        // but in fact it doesn't change in runtime (we ignore what user passes us!)
        // FIXME: Probably it is wrong
        const auto mask_type = s._mask->get_element_type();
        mask_data = ov::Tensor(mask_type, s._mask_shape);
        switch (mask_type) {
        case ov::element::f16:
            prepare_mask<ov::float16>(ov::get_tensor_impl(mask_data));
            break;
        case ov::element::f32:
            prepare_mask<float>(ov::get_tensor_impl(mask_data));
            break;
        default:
            OPENVINO_THROW("Dynamic attenion mask type ", mask_type, " is not supported yet");
        }
    }
};

}  // namespace compiled

namespace runtime {
namespace dynamic {

// A base class to decide the work-scope from some feature
class Selector {
public:
    using Ptr = std::shared_ptr<Selector>;
    virtual ~Selector() = default;
    virtual void prepare() = 0;
    virtual int64_t length() = 0;
};

// No dynamic dispatch - just run over the whole range
class All final : public Selector {
    void prepare() override {}
    int64_t length() override {
        return -1;
    }
};

// Define work scope based on position ids
class PositionIDs final : public Selector {
    std::size_t m_position_ids_idx = 0u;
    int64_t m_current_length = 0;

    const ov::ISyncInferRequest& m_rq;

    PositionIDs(std::size_t param_idx, const ov::ISyncInferRequest& rq);
    void prepare() override;
    int64_t length() override;

public:
    static Selector::Ptr find(const ov::ISyncInferRequest& rq);
};

}  // namespace dynamic
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
