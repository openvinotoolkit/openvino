#include "megakernelImpl.h"

#include "impl/test_runtime.h"

extern "C" mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime() {
    return new mk::TestRuntime();
}

extern "C" void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime) {
    delete runtime;
}