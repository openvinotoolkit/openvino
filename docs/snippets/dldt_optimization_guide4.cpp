#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part4]
InferenceEngine::SizeVector dims_src = {
       1     /* batch, N*/,
       3     /*Channels,*/,
       (size_t) frame_in->Info.Height  /* Height */,
       (size_t) frame_in->Info.Width    /* Width */,
       };
TensorDesc desc(InferenceEngine::Precision::U8, dims_src, InferenceEngine::NCHW);
/* wrapping the RGBP surface data*/
InferenceEngine::TBlob<uint8_t>::Ptr p = InferenceEngine::make_shared_blob<uint8_t>( desc, (uint8_t*) frame_in->Data.R);
inferRequest.SetBlob("input", p);
// …
//! [part4]

return 0;
}
