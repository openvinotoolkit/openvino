#ifndef _TEST_UTILS_HPP_
 #define _TEST_UTILS_HPP_

#include <iostream>

inline void test_assert(bool condition, const char* msg = "")
{
    if (!condition)
        {
            std::cout << "Condition failed with msg: " << msg << std::endl;
            throw std::runtime_error(msg);
        }
}

#endif // _TEST_UTILS_HPP_