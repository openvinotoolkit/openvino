#pragma once


#include "base_sync_infer_request.hpp"

namespace ov :: npuw {

// decoration of inferrequest with prefill model specific KV-cache copying
class LLMPrefillInferRequest final : 
    public IInferRequestSubmissionListener {


  std::function<void (std::size_t, Completed cb)> m_subscribe_dispatch;
  std::function<void ( std::size_t)> m_complete_dispatch;
  std::function<void ( std::size_t idx, std::string , ov::SoPtr<ITensor>)> m_output_ready_dispatch;

public:
    LLMPrefillInferRequest() = delete;
    template <class CB1, class CB2, class CB3>
    LLMPrefillInferRequest(CB1 cb1, CB2 cb2, CB3 cb3) 
        : m_subscribe_dispatch(cb1), m_complete_dispatch(cb2), m_output_ready_dispatch(cb3) { }

    void on_output_ready(std::size_t idx, std::string name, ov::SoPtr<ITensor> tensor) override {
        m_output_ready_dispatch(idx, name, tensor);
    }
    void subscribe_subrequest(std::size_t idx, Completed cb) override {
        m_subscribe_dispatch (idx, cb);
    }
    void complete_subrequest(std::size_t idx) override {
        m_complete_dispatch(idx);
    }
  };

}  // namespace ov::npuw