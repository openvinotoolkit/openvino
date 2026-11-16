#include "megakernelImpl.h"

#include "impl/gemma4POCRuntime.h"

extern "C" mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime() {
  return new mk::Gemma4POCRuntime();
}

extern "C" void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime) {
  if (runtime) delete runtime;
}
