#pragma once
#include "../iMegakernelRuntime.h"
#include "exportApi.h"
#include "gemma4POCParams.h"

extern "C" EXPORT_API mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime();

extern "C" EXPORT_API void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime);
