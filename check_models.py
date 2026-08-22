"""Quick script to check which models are available on Tinker.

Set TINKER_API_KEY env var before running.
"""

import asyncio
import tinker


async def main():
    client = tinker.ServiceClient()
    caps = await client.get_server_capabilities_async()

    print("=== Available Models on Tinker ===\n")
    moe_models = []
    dense_models = []
    for m in caps.supported_models:
        name = m.model_name
        if "-A" in name.split("/")[-1]:
            moe_models.append(name)
        else:
            dense_models.append(name)

    print("MoE models:")
    for m in sorted(moe_models):
        print(f"  {m}")

    print("\nDense models:")
    for m in sorted(dense_models):
        print(f"  {m}")

    print("\n=== Qwen 3.5/3.6 availability ===")
    qwen_new = [m.model_name for m in caps.supported_models if "3.5" in m.model_name or "3.6" in m.model_name]
    if qwen_new:
        for m in qwen_new:
            print(f"  ✓ {m}")
    else:
        print("  ✗ No Qwen 3.5/3.6 models yet")
        qwen3 = [m.model_name for m in caps.supported_models if "Qwen3" in m.model_name]
        print(f"  Qwen3 models available: {qwen3}")


if __name__ == "__main__":
    asyncio.run(main())
