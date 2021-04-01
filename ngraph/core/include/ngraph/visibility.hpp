//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

// https://gcc.gnu.org/wiki/Visibility
// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
#define NGRAPH_HELPER_DLL_IMPORT __declspec(dllimport)
#define NGRAPH_HELPER_DLL_EXPORT __declspec(dllexport)
#define NGRAPH_HELPER_DLL_LOCAL
#elif defined(__GNUC__) && __GNUC__ >= 4
#define NGRAPH_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define NGRAPH_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#define NGRAPH_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define NGRAPH_HELPER_DLL_IMPORT
#define NGRAPH_HELPER_DLL_EXPORT
#define NGRAPH_HELPER_DLL_LOCAL
#endif
