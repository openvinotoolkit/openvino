// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

// A sample program demonstrating using Google C++ testing framework.
//
// Author: wan@google.com (Zhanyong Wan)

#ifndef GTEST_SAMPLES_SAMPLE1_H_
#define GTEST_SAMPLES_SAMPLE1_H_

// Returns n! (the factorial of n).  For negative n, n! is defined to be 1.
int Factorial(int n);

// Returns true iff n is a prime number.
bool IsPrime(int n);

#endif  // GTEST_SAMPLES_SAMPLE1_H_
