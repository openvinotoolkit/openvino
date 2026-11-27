#pragma once

#include <memory>

namespace cldnn {
struct network;
}

namespace ov::intel_gpu::ocl::moe {
class LookaheadContext;

class LookaheadRegistry {
public:
    static std::shared_ptr<LookaheadContext> acquire(cldnn::network& network);
};
}