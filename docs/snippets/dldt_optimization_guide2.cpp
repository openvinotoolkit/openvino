#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
//! [part2]
//Lock Intel MSS surface  
mfxFrameSurface1 *frame_in;   //Input MSS surface.
mfxFrameAllocator* pAlloc = &m_mfxCore.FrameAllocator();    
pAlloc->Lock(pAlloc->pthis, frame_in->Data.MemId, &frame_in->Data);
//Inference Engine code
//! [part2]

return 0;
}
