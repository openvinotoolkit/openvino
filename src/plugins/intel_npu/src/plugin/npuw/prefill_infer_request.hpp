#pragma once


#include "base_sync_infer_request.hpp"

namespace ov :: npuw {

// decoration of inferrequest with prefill model specific KV-cache copying
class LLMPrefillInferRequest final : 
//    public std::enable_shared_from_this<LLMPrefillInferRequest>, 
//    public ov::ISyncInferRequest, 
    public IInferRequestSubmissionListener {


  std::function<void (std::size_t, Completed cb)> m_subscribe_dispatch;
  std::function<void (ov::SoPtr<ov::IAsyncInferRequest>, std::size_t)> m_complete_dispatch;

public:
    LLMPrefillInferRequest() = delete;
    template <class CB1, class CB2>
    LLMPrefillInferRequest(CB1 cb1, CB2 cb2) : m_subscribe_dispatch(cb1), m_complete_dispatch(cb2) { }

    // 
    // void subscribe_itself(std::shared_ptr<IBaseInferRequest> base_request) {
    //     this->base = std::move(base_request);
    //     base->add_infer_requests_listener(shared_from_this());
    // }
    void subscribe_subrequest(std::size_t sz, Completed cb) override {
      m_subscribe_dispatch (sz, cb);
    }
    void complete_subrequest(ov::SoPtr<ov::IAsyncInferRequest> req, std::size_t sz) override {
      m_complete_dispatch(req, sz);
    }
  };

}  // namespace ov::npuw