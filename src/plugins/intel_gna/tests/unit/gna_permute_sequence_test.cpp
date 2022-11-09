// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include <layers/gna_permute.hpp>

using namespace GNAPluginNS;

class PermuteSequenceTest : public ::testing::Test {
};

TEST_F(PermuteSequenceTest, testImpossiblePermute_neg_value) {
    ASSERT_ANY_THROW(genPermutations({-1}));
}

TEST_F(PermuteSequenceTest, testImpossiblePermute_big_value) {
    ASSERT_ANY_THROW(genPermutations({1}));
}

TEST_F(PermuteSequenceTest, testImpossiblePermute_big_value_2) {
    ASSERT_ANY_THROW(genPermutations({0, 2}));
}

TEST_F(PermuteSequenceTest, testImpossiblePermute_same_value) {
    ASSERT_ANY_THROW(genPermutations({0, 1, 0}));
}

TEST_F(PermuteSequenceTest, testIdentity1d) {
    ASSERT_EQ(0, genPermutations({0}).size());
}

TEST_F(PermuteSequenceTest, testIdentity2d) {
    ASSERT_EQ(0, genPermutations({0, 1}).size());
}

TEST_F(PermuteSequenceTest, testIdentity3d) {
    ASSERT_EQ(0, genPermutations({0, 1, 2}).size());
}

TEST_F(PermuteSequenceTest, testIdentity4d) {
    ASSERT_EQ(0, genPermutations({0, 1, 2, 3}).size());
}

TEST_F(PermuteSequenceTest, test2d) {
    ASSERT_EQ(1, genPermutations({1, 0}).size());
    ASSERT_EQ(std::make_pair(0, 1), genPermutations({1, 0}).front());
}

TEST_F(PermuteSequenceTest, test3d_1_Permutation) {
    ASSERT_EQ(1, genPermutations({1, 0, 2}).size());
    ASSERT_EQ(std::make_pair(0, 1), genPermutations({1, 0, 2})[0]);
}

TEST_F(PermuteSequenceTest, test3d_1_Permutation_2) {
    ASSERT_EQ(1, genPermutations({2, 1, 0}).size());
    ASSERT_EQ(std::make_pair(0, 2), genPermutations({2, 1, 0})[0]);
}

TEST_F(PermuteSequenceTest, test3d_2_Permutations) {
    ASSERT_EQ(2, genPermutations({2, 0, 1}).size());
    ASSERT_EQ(std::make_pair(0, 2), genPermutations({2, 0, 1})[0]);
    ASSERT_EQ(std::make_pair(2, 1), genPermutations({2, 0, 1})[1]);
}

TEST_F(PermuteSequenceTest, test3d_4_Permutation) {
    auto permutation = {2, 1, 4, 5, 6, 3, 0, 7};
    ASSERT_EQ(4, genPermutations(permutation).size());
    ASSERT_EQ(std::make_pair(0, 2), genPermutations(permutation)[0]);
    ASSERT_EQ(std::make_pair(2, 4), genPermutations(permutation)[1]);
    ASSERT_EQ(std::make_pair(4, 6), genPermutations(permutation)[2]);
    ASSERT_EQ(std::make_pair(3, 5), genPermutations(permutation)[3]);
}
