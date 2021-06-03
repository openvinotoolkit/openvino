#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part3]
InferenceEngine::SizeVector dims_src = {
    1         /* batch, N*/,
    (size_t) frame_in->Info.Height  /* Height */,
    (size_t) frame_in->Info.Width    /* Width */,
    3 /*Channels,*/,
    };
InferenceEngine::TensorDesc desc(InferenceEngine::Precision::U8, dims_src, InferenceEngine::NHWC);
/* wrapping the surface data, as RGB is interleaved, need to pass only ptr to the R, notice that this wouldn’t work with planar formats as these are 3 separate planes/pointers*/
InferenceEngine::TBlob<uint8_t>::Ptr p = InferenceEngine::make_shared_blob<uint8_t>( desc, (uint8_t*) frame_in->Data.R);
inferRequest.SetBlob("input", p);
inferRequest.Infer();
//Make sure to unlock the surface upon inference completion, to return the ownership back to the Intel MSS
pAlloc->Unlock(pAlloc->pthis, frame_in->Data.MemId, &frame_in->Data);
//! [part3]

return 0;
}
