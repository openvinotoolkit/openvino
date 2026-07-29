#include <iostream>

int main () {
    const char *patchGenerator = R"V0G0N(
[
    {
        "str": "std::cout << \"Latency:\\nAverage: 700.0 ms\\n\";",
        "comment": "success_1"
    },
    {
        "str": "std::cout << \"Latency:\\nAverage: 700.0 ms\\n\";",
        "comment": "success_2"
    },
    {
        "str": "std::cout << \"Latency:\\nAverage: 500.0 ms\\n\";",
        "comment": "error_1",
        "state": "BREAK"
    },
    {
        "str": "std::cout << \"Latency:\\nAverage: 500.0 ms\\n\";",
        "comment": "error_2"
    },
    {
        "str": "std::cout << \"Latency:\\nAverage: 500.0 ms\\n\";",
        "comment": "error_1"
    },
    {
        "str": "std::cout << \"Latency:\\nAverage: 500.0 ms\\n\";",
        "comment": "error_2"
    },
    {
        "str": "std::cout << \"Latency:\\nAverage: 500.0 ms\\n\";",
        "comment": "error_1"
    },
    {
        "str": "std::cout << \"Latency:\\nAverage: 500.0 ms\\n\";",
        "comment": "error_2"
    }
]
)V0G0N";
    return 0;
}