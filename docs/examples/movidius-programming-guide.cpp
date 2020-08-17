#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
struct Request {

    InferenceEngine::InferRequest::Ptr inferRequest;

    int frameidx;

};

//! [part0]

//! [part1]
// numRequests is the number of frames (max size, equal to the number of VPUs in use)

vector<Request> request(numRequests);

//! [part1]

//! [part2]
// initialize infer request pointer – Consult IE API for more detail.

request[i].inferRequest = executable_network.CreateInferRequestPtr();

//! [part2]

//! [part3]
// Run inference

request[i].inferRequest->StartAsync();

//! [part3]

//! [part4]
request[i].inferRequest->SetCompletionCallback(InferenceEngine::IInferRequest::Ptr context)

//! [part4]

return 0;
}
