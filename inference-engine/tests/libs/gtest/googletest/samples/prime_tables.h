// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

// This provides interface PrimeTable that determines whether a number is a
// prime and determines a next prime number. This interface is used
// in Google Test samples demonstrating use of parameterized tests.

#ifndef GTEST_SAMPLES_PRIME_TABLES_H_
#define GTEST_SAMPLES_PRIME_TABLES_H_

#include <algorithm>

// The prime table interface.
class PrimeTable {
 public:
  virtual ~PrimeTable() {}

  // Returns true iff n is a prime number.
  virtual bool IsPrime(int n) const = 0;

  // Returns the smallest prime number greater than p; or returns -1
  // if the next prime is beyond the capacity of the table.
  virtual int GetNextPrime(int p) const = 0;
};

// Implementation #1 calculates the primes on-the-fly.
class OnTheFlyPrimeTable : public PrimeTable {
 public:
  virtual bool IsPrime(int n) const {
    if (n <= 1) return false;

    for (int i = 2; i*i <= n; i++) {
      // n is divisible by an integer other than 1 and itself.
      if ((n % i) == 0) return false;
    }

    return true;
  }

  virtual int GetNextPrime(int p) const {
    for (int n = p + 1; n > 0; n++) {
      if (IsPrime(n)) return n;
    }

    return -1;
  }
};

// Implementation #2 pre-calculates the primes and stores the result
// in an array.
class PreCalculatedPrimeTable : public PrimeTable {
 public:
  // 'max' specifies the maximum number the prime table holds.
  explicit PreCalculatedPrimeTable(int max)
      : is_prime_size_(max + 1), is_prime_(new bool[max + 1]) {
    CalculatePrimesUpTo(max);
  }
  virtual ~PreCalculatedPrimeTable() { delete[] is_prime_; }

  virtual bool IsPrime(int n) const {
    return 0 <= n && n < is_prime_size_ && is_prime_[n];
  }

  virtual int GetNextPrime(int p) const {
    for (int n = p + 1; n < is_prime_size_; n++) {
      if (is_prime_[n]) return n;
    }

    return -1;
  }

 private:
  void CalculatePrimesUpTo(int max) {
    ::std::fill(is_prime_, is_prime_ + is_prime_size_, true);
    is_prime_[0] = is_prime_[1] = false;

    for (int i = 2; i <= max; i++) {
      if (!is_prime_[i]) continue;

      // Marks all multiples of i (except i itself) as non-prime.
      for (int j = 2*i; j <= max; j += i) {
        is_prime_[j] = false;
      }
    }
  }

  const int is_prime_size_;
  bool* const is_prime_;

  // Disables compiler warning "assignment operator could not be generated."
  void operator=(const PreCalculatedPrimeTable& rhs);
};

#endif  // GTEST_SAMPLES_PRIME_TABLES_H_
