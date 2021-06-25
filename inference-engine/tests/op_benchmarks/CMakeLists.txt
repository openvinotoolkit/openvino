add_executable(op_benchmarks convolution.cpp)
target_link_libraries(op_benchmarks PRIVATE benchmark::benchmark
                                        benchmark::benchmark_main
                                        ngraph
                                        ngraph_backend
                                        inference_engine)
