from vllm import LLM, SamplingParams
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM offline generation with configurable options.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2, help="max_num_batched_tokens")
    parser.add_argument(
        "--compilation",
        type=str,
        choices=["mirage", "vllm", "none"],
        default="mirage",
        help="Compilation backend: mirage / vllm / none (enforce_eager: True)",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--async-scheduling", action="store_true", help="Enable asynchronous scheduling")
    args = parser.parse_args()

    start_time = time.time()
    llm_kwargs = {
        "model": args.model,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "block_size": 64,
    }
    
    if args.compilation == "mirage":
        compilation_config = {"mode": 2, "backend": "mirage_byname"}
        llm_kwargs["compilation_config"] = compilation_config
    elif args.compilation == "vllm":
        compilation_config = None
        llm_kwargs["compilation_config"] = compilation_config
    else:
        llm_kwargs["enforce_eager"] = True
        
    if args.async_scheduling:
        llm_kwargs["async_scheduling"] = True
    # if compilation_config is not None:
    #     llm_kwargs["compilation_config"] = compilation_config
    # else:
    #     llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)
    time_after_llm = time.time()
    prompts = ["""<|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Give me a short introduction to large language model.<|im_end|>
    <|im_start|>assistant"""]
    # sampling_params = SamplingParams(max_tokens=128)
    sampling_params = SamplingParams(max_tokens=args.max_tokens)
    llm.start_profile() 
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()
    end_time = time.time()
    print("Total Tokens generated: ", len(outputs[0].outputs[0].token_ids))
    print("Total Time taken: ", end_time - start_time)
    print("Time taken to initialize LLM: ", time_after_llm - start_time)
    print("Time taken to generate outputs: ", end_time - time_after_llm)
    print(len(outputs), " outputs generated")
    # print("prompt: ", outputs[0].prompt)
    # print("output: ", outputs[0].outputs[0].text)
    for index, output in enumerate(outputs):
        print(f"prompt {index}: ", output.prompt)
        print(f"output {index}: ", output.outputs[0].text)
        print("-" * 100)
