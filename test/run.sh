TEST_USE_SDPA_OCL=1 OV_KV_CACHE_PRECISION=i8 python ~/work/openvino.genai/tools/llm_bench/benchmark.py -d GPU.1 -lc "{\"ATTENTION_BACKEND\" : \"SDPA\"}" -m /mnt/models/ov-share-13.sclab.intel.com/cv_bench_cache/WW29_llm-optimum_2026.3.0-22446-RC2/llama-2-7b-chat-hf/pytorch/ov/OV_FP16-4BIT_DEFAULT/ -n 3 -ic 256 -pf /home/shingyuk/work/frameworks.ai.openvino.llm.prompts/4096/llama-2-7b-chat.jsonl

TEST_USE_SDPA_OCL=1 OV_KV_CACHE_PRECISION=f16 python ~/work/openvino.genai/tools/llm_bench/benchmark.py -d GPU.1 -m /mnt/models/ov-share-13.sclab.intel.com/cv_bench_cache/WW29_llm-optimum_2026.3.0-22446-RC2/llama-2-7b-chat-hf/pytorch/ov/OV_FP16-4BIT_DEFAULT/ -n 3 -ic 256 -pf /home/shingyuk/work/frameworks.ai.openvino.llm.prompts/4096/llama-2-7b-chat.jsonl




TEST_USE_SDPA_OCL=1 OV_KV_CACHE_PRECISION=i8 python ~/work/openvino.genai/tools/llm_bench/benchmark.py -d GPU.1 -lc "{\"ATTENTION_BACKEND\" : \"SDPA\"}" -m /mnt/models/ov-share-13.sclab.intel.com/cv_bench_cache/WW25_llm-optimum_2026.3.0-22208/llama-3.1-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT -n 3 -ic 256 -pf /home/shingyuk/work/frameworks.ai.openvino.llm.prompts/4096/llama-3-8b.jsonl

TEST_USE_SDPA_OCL=1 OV_KV_CACHE_PRECISION=f16 python ~/work/openvino.genai/tools/llm_bench/benchmark.py -d GPU.1 -m /mnt/models/ov-share-13.sclab.intel.com/cv_bench_cache/WW25_llm-optimum_2026.3.0-22208/llama-3.1-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT -n 3 -ic 256 -pf /home/shingyuk/work/frameworks.ai.openvino.llm.prompts/4096/llama-3-8b.jsonl


TEST_USE_SDPA_OCL=1 OV_KV_CACHE_PRECISION=i8 python ~/work/openvino.genai/tools/llm_bench/benchmark.py -d GPU.1 -lc "{\"ATTENTION_BACKEND\" : \"SDPA\"}" -m /mnt/models/ov-share-13.sclab.intel.com/cv_bench_cache/WW29_llm-optimum_2026.3.0-22446-RC2/gemma-2-9b-it/pytorch/ov/OV_FP16-4BIT_DEFAULT/ -n 3 -ic 256 -pf /home/shingyuk/work/frameworks.ai.openvino.llm.prompts/4096/gemma-2-9b-it.jsonl

TEST_USE_SDPA_OCL=1 OV_KV_CACHE_PRECISION=f16 python ~/work/openvino.genai/tools/llm_bench/benchmark.py -d GPU.1 -m /mnt/models/ov-share-13.sclab.intel.com/cv_bench_cache/WW29_llm-optimum_2026.3.0-22446-RC2/gemma-2-9b-it/pytorch/ov/OV_FP16-4BIT_DEFAULT/ -n 3 -ic 256 -pf /home/shingyuk/work/frameworks.ai.openvino.llm.prompts/4096/gemma-2-9b-it.jsonl


TEST_USE_SDPA_OCL=1 OV_KV_CACHE_PRECISION=i8 python ~/work/openvino.genai/tools/llm_bench/benchmark.py -d GPU.1 -lc "{\"ATTENTION_BACKEND\" : \"SDPA\"}" -m /mnt/models/ov-share-13.sclab.intel.com/cv_bench_cache/WW29_llm-optimum_2026.3.0-22446-RC2/gemma-3-4b-it/pytorch/ov/OV_FP16-4BIT_DEFAULT/ -n 3 -ic 256 -pf /home/shingyuk/work/frameworks.ai.openvino.llm.prompts/4096/gemma-3-4b-it.jsonl


TEST_USE_SDPA_OCL=1 OV_KV_CACHE_PRECISION=i8 python ~/work/openvino.genai/tools/llm_bench/benchmark.py -d GPU.1 -lc "{\"ATTENTION_BACKEND\" : \"SDPA\"}" -m /mnt/models/ov-share-13.sclab.intel.com/cv_bench_cache/WW29_llm-optimum_2026.3.0-22446-RC2/gemma-4-e2b-it/pytorch/ov/OV_FP16-4BIT_DEFAULT/ -n 3 -ic 256 -pf /home/shingyuk/work/frameworks.ai.openvino.llm.prompts/4096/gemma-4-e2b-it.jsonl


