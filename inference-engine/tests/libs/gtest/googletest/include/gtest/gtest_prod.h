// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef GTEST_INCLUDE_GTEST_GTEST_PROD_H_
#define GTEST_INCLUDE_GTEST_GTEST_PROD_H_

// When you need to test the private or protected members of a class,
// use the FRIEND_TEST macro to declare your tests as friends of the
// class.  For example:
//
// class MyClass {
//  private:
//   void MyMethod();
//   FRIEND_TEST(MyClassTest, MyMethod);
// };
//
// class MyClassTest : public testing::Test {
//   // ...
// };
//
// TEST_F(MyClassTest, MyMethod) {
//   // Can call MyClass::MyMethod() here.
// }

#define FRIEND_TEST(test_case_name, test_name)\
friend class test_case_name##_##test_name##_Test

#endif  // GTEST_INCLUDE_GTEST_GTEST_PROD_H_
