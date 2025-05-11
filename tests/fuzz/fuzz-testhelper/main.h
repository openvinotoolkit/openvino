// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/*!
\file
\brief Replacement of libFuzzer main entrypoint for fuzz tests.

Include this file into code so that Microsoft linker can resolve
main entrypoint from static library.
Microsoft linker only resolve libary functions which are referenced,
`main` is not.
*/
#ifndef TESTS_FUZZ_TESTHELPER_MAIN_H_
#define TESTS_FUZZ_TESTHELPER_MAIN_H_

#if !defined(WITH_LIBFUZZER)
extern "C" int main(int argc, char* argv[]);
// make a reference to main so linker resolve it from static library
void* main_ptr_ = (void*)main;
#endif  // !defined(WITH_LIBFUZZER)

#endif  // TESTS_FUZZ_TESTHELPER_MAIN_H_
