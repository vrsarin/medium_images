# Tokens Are Not Free: How Every Token Generated Drives Real Operating Cost

## Introduction

When organizations deploy large language models — whether through cloud APIs or locally on GPU hardware — the conversation around cost almost always gravitates toward infrastructure. How many GPUs? What instance type? What cloud provider? These are legitimate questions, but they frame the problem incorrectly. The true atomic unit of cost in any LLM deployment is not the server, not the API call, and not the hour of compute. It is the **token**.

Understanding why requires stepping inside how a transformer-based language model actually works — because the economics of token generation are a direct consequence of the mathematics of the architecture.

---

## What Is a Token?

A token is the smallest unit of text that a language model processes. It is neither a character nor a word — it sits somewhere between the two. In English, the average token corresponds to roughly four characters or three-quarters of a word. The word "infrastructure" is two tokens. "MikroTik" is three. A typical sentence of twenty words is approximately twenty-five to thirty tokens.

Every interaction with an LLM involves two categories of tokens:

**Input tokens** — everything the model reads before generating a single word. This includes the system prompt (the standing instructions that define the model's behavior), the conversation history (every prior turn in the session), any retrieved documents injected via RAG, and the user's current message. In production systems with rich system prompts and multi-turn histories, input tokens routinely reach five thousand to twenty thousand tokens per call — before the model has said a word.

**Output tokens** — everything the model generates in response. These are produced one at a time, sequentially, and they are the most computationally expensive tokens in the system.

The distinction matters enormously for cost accounting, and the reason lies in how the neural network processes each category differently.

---

## How a Transformer Generates Tokens

A modern language model — GPT, Gemma, Mistral, Claude — is built on the transformer architecture. To understand why tokens drive cost, you need a working mental model of what happens inside the network each time it produces a single token.

### The Forward Pass

Every token generation requires one complete **forward pass** through the model. A forward pass is the mathematical journey a representation of text takes from the input layer to the output layer. For a 7-billion-parameter model like Mistral 7B, this means passing through approximately thirty-two transformer layers in sequence. For a 70-billion-parameter model, that number rises to eighty. Each layer performs two expensive operations: **multi-head self-attention** and a **feed-forward network (FFN)** transformation.

The attention mechanism is where the model decides which parts of the input are relevant to what it is currently generating. Mathematically, every token attends to every other token in the context window — computing a weighted relationship between them. The weight of that relationship (the attention score) is determined by three learned matrices: queries, keys, and values. The computational cost of this operation scales with the **square of the context length**. Double the context, quadruple the attention computation.

The FFN then transforms the attended representation through two large matrix multiplications before passing it to the next layer. Across thirty-two or eighty layers, these matrix multiplications account for the majority of GPU FLOP consumption per token.

### Two Phases: Prefill and Decode

Token generation is not a single uniform process. It has two mechanically distinct phases.

**Prefill** is the processing of all input tokens. During prefill, the GPU can process every token in the input simultaneously in parallel — it is a highly parallelized operation that leverages the GPU's massive matrix multiplication throughput. A twenty-thousand-token input context is processed in a single, relatively brief burst. The output of prefill is the KV cache.

**Decode** is the generation of output tokens. Here the model generates exactly one token per forward pass, and it cannot generate the next token until the current one is complete. This is inherently sequential — there is no parallelism to exploit across output tokens. Each decode step requires a full forward pass through all layers of the model. A five-hundred-token response requires five hundred sequential forward passes.

This asymmetry is the fundamental reason output tokens are more expensive than input tokens on cloud APIs — and why output token volume is the primary driver of GPU-time on local hardware.

### The KV Cache

During prefill, the model computes key and value matrices for every input token and stores them in GPU memory — the KV cache. During decode, each new token can attend to these cached values without recomputing them, which dramatically reduces the cost of each decode step compared to recomputing attention from scratch.

However, the KV cache is not free. Its memory footprint scales with `layers × attention_heads × context_length × data_type_bytes`. A 70B model with a 128,000-token context window requires tens of gigabytes of VRAM just for the KV cache — leaving less room for model weights, forcing quantization or smaller batch sizes, and compressing throughput.

---
(./images/diagram_03_12_2025.png)


## How Tokens Become OPEX

On a cloud API, the cost model is explicit: you are billed per million input tokens and per million output tokens, with output carrying a rate two to five times higher than input. The invoice is visible, line-itemed, and attributable.

On local GPU hardware, the cost model is identical in structure but expressed in different units. There is no invoice from a vendor. Instead, the cost manifests as electricity consumption, cooling load, and hardware wear — all driven by the same underlying variable: how many tokens the GPU must process, and for how long.

### Electricity

GPU power draw is not constant. Under idle conditions, a high-end inference GPU like a Tesla P40 draws approximately thirty to fifty watts. Under full inference load, it draws two hundred to two hundred and fifty watts. The transition from idle to full load is triggered by token processing.

During prefill, the GPU is maximally utilized — all cores active, full power draw. During decode, each sequential forward pass sustains full power draw for its duration. The total energy consumed per response is therefore a direct function of the number of output tokens, the tokens-per-second throughput, and the GPU's wattage profile.

```
Electricity cost per call = (output_tokens / tokens_per_sec) × (GPU_watts / 1000) × price_per_kWh
```

At fifty tokens per second on a 250-watt GPU drawing current at $0.12 per kilowatt-hour, a five-hundred-token response costs approximately $0.0003 in electricity. That figure appears negligible in isolation. At ten thousand calls per day, it becomes $3.00 per day, $90 per month — from electricity alone, on a single card, for a modest workload.

### Cooling

Every watt the GPU dissipates becomes heat that must be removed. In a rack environment, cooling overhead typically adds twenty to thirty percent to the electrical cost of the compute it supports. This is not a fixed cost — it is proportional to GPU utilization, which is proportional to token throughput. Higher token volume means more heat, more cooling, more electricity.

### Hardware Wear and Lifespan

GPUs are not indefinitely durable. Sustained high utilization — high temperature, thermal cycling, memory pressure — accelerates electromigration in the silicon and degrades solder joints over time. A GPU driven at sustained high utilization through high token volume will reach end of life faster than the same GPU at moderate utilization. The amortization schedule for hardware is therefore not purely a function of calendar time — it is a function of workload intensity, which is a function of token volume.

### Memory Bandwidth and Throughput Collapse

The P40 uses GDDR5 memory — not HBM2 or HBM3 as found in newer data-center GPUs. GDDR5 has significantly lower memory bandwidth, which becomes a bottleneck during decode. Each decode step requires loading model weights from VRAM into GPU core registers — a memory-bandwidth-bound operation. At high token generation rates or large context windows where the KV cache competes with model weights for VRAM, throughput degrades, decode slows, and effective tokens-per-second drops. This means each token takes longer to produce, extending the high-power-draw period per response and increasing per-token electrical cost.

### Reasoning Models and the Hidden Token Problem

Some models — including extended thinking variants and openly reasoning models like DeepSeek-R1 — generate internal reasoning tokens before producing the final response. These tokens are often not surfaced to the user but are fully computed by the model. A prompt that elicits two thousand tokens of internal reasoning followed by a two-hundred-token answer does not cost two hundred output tokens of compute. It costs two thousand two hundred. On local hardware, that is two thousand two hundred sequential decode passes — each consuming a full forward pass through all model layers. The electrical cost scales accordingly, invisibly.

---

## Why Token Accounting Must Be Part of OPEX

The argument occasionally made against including token generation in local OPEX accounting rests on a single premise: the hardware has already been paid for. This confuses capital expenditure with operating expenditure.

Capital expenditure is the one-time cost of acquiring the GPU. It is sunk at the moment of purchase. Operating expenditure is the recurring cost of running the GPU — electricity, cooling, maintenance, and replacement. These costs occur every month, every day, every call. They do not vanish because the hardware was bought outright.

Token volume is the only variable that drives the variable component of local OPEX. Hardware depreciation is fixed on a calendar schedule. Staff costs are fixed. Rack space is fixed. But electricity consumption, cooling load, and hardware wear rate all vary — and the single variable that causes them to vary is how hard the GPU is working, which is directly determined by how many tokens it is processing.

Without per-token cost accounting on local infrastructure, the electricity and cooling line items in the operating budget become unexplained constants. Finance cannot attribute them to workloads. Engineering cannot optimize them. Leadership cannot compare the true cost of a local inference call against a cloud API call to make rational build-versus-buy decisions.

The unified cost formula that closes the TCO model requires token measurement at every layer:

```
Cloud OPEX  = (input_tokens / 1M × input_rate) + (output_tokens / 1M × output_rate)

Local OPEX  = (output_tokens / tokens_per_sec) × (GPU_watts / 1000) × $/kWh
            + cooling_multiplier (× 1.25–1.30)
            + hardware_amortization (proportional to utilization)

Total TCO   = CapEx (hardware) + ∑ Local OPEX + ∑ Cloud OPEX + Staff + Facilities
```

Every term in the local OPEX row is either directly token-driven or proportional to something that is. Removing token measurement from this model does not simplify the accounting — it introduces an unquantified variable at the center of the equation.

---

## Conclusion

Tokens are not an abstraction imposed by vendors for billing convenience. They are the fundamental unit of computation in transformer-based language models, and the computational cost of processing each one is real, measurable, and cumulative. Every token in the system prompt is processed on every single call. Every output token requires a full sequential forward pass through every layer of the model. Every forward pass consumes GPU cycles, draws electrical power, generates heat, and contributes to hardware wear.

On cloud infrastructure, this cost is visible on the invoice. On local GPU infrastructure, it is embedded in the electricity bill, the cooling system load, and the hardware replacement cycle. The mechanism is identical. Only the ledger entry differs.

Organizations that measure token consumption only on cloud calls and treat local inference as a fixed-cost resource are operating with an incomplete cost model. They cannot optimize what they do not measure, and they cannot make sound infrastructure decisions without a complete picture of where cost actually originates. Token accounting on local inference is not an engineering metric — it is a financial control instrument, and it belongs in every TCO analysis that includes local LLM deployment.
