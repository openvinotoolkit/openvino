#include <iostream>

int main () {
    const char *patchGenerator = R"V0G0N(
[
    {
        "str": "std::cout << \"prefix\\Perplexity: 100.0% [FAILED:  abs error = 9.118 | relative error = 0.3144]\\n\";",
        "comment": "success_1"
    },
    {
        "str": "std::cout << \"prefix\\Perplexity: 100.0% [FAILED:  abs error = 9.118 | relative error = 0.3144]\\n\";",
        "comment": "success_2"
    },
    {
        "str": "std::cout << \"prefix\\Perplexity: 50.0% [FAILED:  abs error = 9.118 | relative error = 0.3144]\\n\";",
        "comment": "error_1",
        "state": "BREAK"
    },
    {
        "str": "std::cout << \"prefix\\Perplexity: 50.0% [FAILED:  abs error = 9.118 | relative error = 0.3144]\\n\";",
        "comment": "error_2"
    },
    {
        "str": "std::cout << \"prefix\\Perplexity: 50.0% [FAILED:  abs error = 9.118 | relative error = 0.3144]\\n\";",
        "comment": "error_1"
    },
    {
        "str": "std::cout << \"prefix\\Perplexity: 50.0% [FAILED:  abs error = 9.118 | relative error = 0.3144]\\n\";",
        "comment": "error_2"
    },
    {
        "str": "std::cout << \"prefix\\Perplexity: 50.0% [FAILED:  abs error = 9.118 | relative error = 0.3144]\\n\";",
        "comment": "error_1"
    },
    {
        "str": "std::cout << \"prefix\\Perplexity: 50.0% [FAILED:  abs error = 9.118 | relative error = 0.3144]\\n\";",
        "comment": "error_2"
    }
]
)V0G0N";
    return 0;
}