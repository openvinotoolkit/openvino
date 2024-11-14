// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_infer_request.hpp"

#include "llm_compiled_model.hpp"

template <typename T>
void print_tensor(ov::SoPtr<ov::ITensor> t) {
    auto* ptr = t->data<T>();
    std::cout << "[ ";
    for (int i = 0; i < t->get_size(); ++i) {
        std::cout << ptr[i] << " ";
    }
    std::cout << " ]" << std::endl;
}

ov::npuw::LLMInferRequest::LLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    std::cout << "[LOG_DEBUG] ov::npuw::LLMInferRequest::LLMInferRequest() " << std::endl;
    m_kvcache_request = compiled_model->kvcache_compiled->create_infer_request();
    m_prefill_request = compiled_model->prefill_compiled->create_infer_request();
}

void ov::npuw::LLMInferRequest::infer() {
    std::cout << "[LOG_DEBUG] LLMInferRequest::infer()" << std::endl;
    const auto& inputs = this->get_inputs();
    std::cout << "inputs size: " << inputs.size() << std::endl;

    std::cout << inputs[0].get_partial_shape() << std::endl;
    std::cout << "print input" << std::endl;
    std::cout << inputs[0] << std::endl;
    std::cout << "print input - done" << std::endl;
    auto input_ids = get_tensor(inputs[0]);
    //auto attention_mask = this->get_tensor(inputs[1]);
    //auto position_ids   = this->get_tensor(inputs[2]);

    std::cout << "print tensor: " << std::endl;
    print_tensor<int64_t>(input_ids);
    //print_tensor<int64_t>(attention_mask);
    //print_tensor<int64_t>(position_ids);
}
