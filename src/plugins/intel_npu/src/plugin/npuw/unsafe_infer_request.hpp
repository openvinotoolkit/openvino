#pragma once

#include "openvino/runtime/iasync_infer_request.hpp"

namespace ov :: npuw {
class UnsafeAsyncInferRequest : public IAsyncInferRequest {

    std::shared_ptr<IInferRequest> m_base;
    std::shared_ptr<IAsyncInferRequest> m_async_wrap;

    public:
    UnsafeAsyncInferRequest(const std::shared_ptr<IInferRequest>& base,
                            const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                            const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor) 
                            : IAsyncInferRequest(nullptr, nullptr, nullptr), m_base(base) {
        m_base = base;
        m_async_wrap = std::make_shared<ov::IAsyncInferRequest>(m_base, task_executor, callback_executor);
    }
 
    // IInferRequest dispatch
    void infer() override {
        m_async_wrap->infer();
    }
 
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return m_async_wrap->get_profiling_info();
     }

    // here is stateless functions that are potentially unsafe, so make sure check state(query_state) before calling them
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const {
        return m_base->get_tensor(port);
    }
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const {
        return m_base->get_tensors(port);
    }

     void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
        m_async_wrap->set_tensor(port, tensor);
     }
 
     void set_tensors(const ov::Output<const ov::Node>& port,
                              const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
        m_async_wrap->set_tensors(port, tensors);
    }
 
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const {
        return m_async_wrap->query_state();
    }
 
    const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const {
        return m_async_wrap->get_compiled_model();
    }
 
     const std::vector<ov::Output<const ov::Node>>& get_inputs() const {
        return m_async_wrap->get_inputs();
     }
 
     const std::vector<ov::Output<const ov::Node>>& get_outputs() const {
        return m_async_wrap->get_outputs();
     }
 
 protected:
    void check_tensors() const override {
    }

public:
    //// IAsyncInferRequest:

   
    void start_async() override{
        m_async_wrap->start_async();   
    }
    void wait() override {
        m_async_wrap->wait();   
    }
    bool wait_for(const std::chrono::milliseconds& timeout) override {
        return m_async_wrap->wait_for(timeout);   
    }
    void cancel() override {
        m_async_wrap->cancel();   
    }

    void set_callback(std::function<void(std::exception_ptr)> callback) {
        m_async_wrap->set_callback(callback);   
    }
   

};
 

}  // namespace ov :: npuw