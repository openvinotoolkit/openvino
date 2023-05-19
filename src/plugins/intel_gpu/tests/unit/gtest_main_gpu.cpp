// Copyright 2006, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cstdio>
#include <string>
#include <stdlib.h>

#include "intel_gpu/runtime/device_query.hpp"
#include "gtest/gtest.h"
#include "test_utils/test_utils.h"
#include "gflags/gflags.h"

DEFINE_int32(device_suffix, -1, "GPU Device ID (a number starts from 0)");

GTEST_API_ int main(int argc, char** argv) {
    printf("Running main() from %s\n", __FILE__);

    //gflags
    gflags::AllowCommandLineReparsing();
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_device_suffix != -1 && cldnn::device_query::device_id == -1)
        cldnn::device_query::device_id = FLAGS_device_suffix;
    //restore cmdline arg for gtest
    auto varg=gflags::GetArgvs();
    int new_argc = static_cast<int>(varg.size());
    char** new_argv=new char*[new_argc];
    for(int i=0;i<new_argc;i++)
        new_argv[i]=&varg[i][0];

    if (const auto env_var = std::getenv("cl_cache_dir"))
        printf("Env variable cl_cache_dir: %s\n", env_var);
    else
        printf("WARNING: cl_cache_dir is not set. Test will take longer than expected\n");

    //gtest
    testing::InitGoogleTest(&new_argc, new_argv);
    auto retcode = RUN_ALL_TESTS();
    delete[] new_argv;
    return retcode;
}
