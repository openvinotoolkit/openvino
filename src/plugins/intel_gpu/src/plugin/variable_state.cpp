// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <intel_gpu/plugin/variable_state.hpp>
#include <blob_factory.hpp>

namespace ov {
namespace intel_gpu {

VariableState::VariableState(const std::string &name,
    const std::vector<cldnn::network::VariableState::Ptr> &states,
    std::shared_ptr<cldnn::engine> engine, int currentBatch) :
    InferenceEngine::IVariableStateInternal {name},
    currentBatch_ {currentBatch},
    states_ {states},
    desc_{
        PrecisionFromDataType(states.front()->memory->get_layout().data_type),
        AggregateShape(states.front()->memory->get_layout()),
        InferenceEngine::Layout::ANY
    },
    engine_ {std::move(engine)} {
}

void VariableState::Reset() {
    IterateOverStates([this](cldnn::network::VariableState &state) {
        state.is_set = false;
    });
}

void VariableState::SetState(const InferenceEngine::Blob::Ptr &newState) {
    auto lock = std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(newState)->rmap();
    auto data = lock.as<char*>();
    IterateOverStates([&data, this](cldnn::network::VariableState &state) {
        state.memory->copy_from(engine_->get_program_stream(), data);
        data += state.memory->get_layout().bytes_count();
        state.is_set = true;
    });
    engine_->get_program_stream().enqueue_barrier();
}

InferenceEngine::Blob::CPtr VariableState::GetState() const {
    auto blob = make_blob_with_precision(desc_, InferenceEngine::CreateDefaultAllocator());
    blob->allocate();
    auto blobLock = std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(blob)->wmap();
    auto data = blobLock.as<char*>();
    IterateOverStates([&data, this](cldnn::network::VariableState &state) {
        cldnn::mem_lock<char, cldnn::mem_lock_type::read> lock { state.memory, engine_->get_program_stream() };
        std::copy(lock.begin(), lock.end(), data);
        data += state.memory->get_layout().bytes_count();
    });
    return blob;
}

InferenceEngine::SizeVector VariableState::AggregateShape(const cldnn::layout &layout) {
    const auto& dims = layout.get_dims();
    InferenceEngine::SizeVector shape {dims.begin(), dims.end()};
    if (currentBatch_ != -1)
        shape.front() = currentBatch_;
    return shape;
}

void VariableState::IterateOverStates(std::function<void(cldnn::network::VariableState&)> f) const {
    for (int i = 0; i < states_.size(); i++) {
        auto batch = 1 << i;
        if (batch & currentBatch_)
            f(*states_[i]);
    }
}

}  // namespace intel_gpu
}  // namespace ov
