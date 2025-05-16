Decoding CPU vs. GPU: A Detailed Exploration of NVIDIA and AMD GPU Architectures

![CPU vs GPU Overview](cpuvsgpu.png)  
*Figure: High-level comparison of CPU and GPU architectures.*

![CPU vs GPU Table](cpuvsgputable.png)  
*Figure: Summary of architectural strengths.*

---

## Key Metrics

| Metric      | Definition                                                                                   |
|-------------|----------------------------------------------------------------------------------------------|
| **Bandwidth** (VRAM) | Amount of data transferred per second (e.g., GB/s). Critical for moving large model weights and activations efficiently. |
| **Latency** (cache)  | Delay to begin a data transfer (e.g., ns). Important for accessing small, frequently reused data (attention caches, etc.). |
| **Throughput**       | Useful work completed per second (e.g., tokens/sec or FLOPs). Determined by both bandwidth and latency, plus compute and pipelining efficiency. |

---

## Memory Hierarchy

- **VRAM (GDDR6/HBM)**  
  - **High bandwidth**, higher latency  
  - Used to store entire model weights and large activations (MB → GB scale)  
  - If bandwidth is insufficient, compute units stall waiting for data.

- **Cache (L1/L2)**  
  - **Low latency**, low capacity (KB → MB scale)  
  - Holds “hot” data (e.g., key-value caches in transformers)  
  - Storing all weights here is infeasible due to size and cost constraints.

---

## Memory Bandwidth of Selected GPUs

| GPU Model                   | Memory Type | Bandwidth    | Capacity |
|-----------------------------|-------------|--------------|----------|
| **NVIDIA RTX 5090**         | GDDR7       | 1.79 TB/s     | 32 GB    |
| **AMD Radeon RX 7900 XTX**  | GDDR6       | 960 GB/s      | 24 GB    |
| **NVIDIA Blackwell B200**   | HBM3e       | 8.0 TB/s      | 192 GB   |
| **AMD Instinct MI325X**     | HBM3e       | 6.0 TB/s      | 288 GB   |

---

## Measuring Performance

- **Tokens/sec**  
  How many tokens a model processes or generates per second. Reflects real-world LLM throughput.

- **FLOPs (Floating-Point Operations/sec)**  
  Theoretical compute capacity. Doesn’t always map directly to LLM speed, since memory bandwidth and latency play major roles.

---

## GPU Architecture Highlights

1. **Thousands of Simple Cores**  
   Optimized for parallel math (FP32, FP16, INT8).

2. **SIMD vs. SIMT**  
   - **SIMD (Single Instruction, Multiple Data)**  
     One instruction operates on multiple data elements in lockstep.  
   - **SIMT (Single Instruction, Multiple Threads)**  
     Programmers write many independent threads; the GPU groups them (warps/wavefronts) and executes each group via SIMD hardware.

3. **Typical Core Counts**  
   - **NVIDIA Blackwell B200**  
     - ~208 SMs (Streaming Multiprocessors)  
     - ~96 FP32 cores per SM → 208 × 96 = 19,968 cores (often cited as ~20,480)  
   - **AMD Instinct MI325X**  
     - ~240 CUs (Compute Units)  
     - 64 FP32 ALUs per CU → 240 × 64 = 15,360 cores

---

## CUDA Cores and Execution Model

- **CUDA core** (NVIDIA) / **Stream processor** (AMD) = FP32 ALU  
- **SM** (NVIDIA) / **CU** (AMD) = cluster of cores + shared resources  
- **Warp** (NVIDIA) = 32 threads; **Wavefront** (AMD) = 64 threads  
- **SIMT** groups threads into warps/wavefronts; **SIMD** executes one instruction across all lanes in the group.

---

## GPU Programming Concepts

- **Kernel**  
  A GPU function that runs in parallel on many threads. Written in CUDA (NVIDIA), HIP (AMD), or OpenCL.

- **Tensor**  
  A multi-dimensional array (0D = scalar, 1D = vector, 2D = matrix, etc.).

---

## Example: Matrix Multiplication on GPU

    When I want to multiply two matrices (say A[1024×1024] × B[1024×1024]),
    I write a kernel that describes how to compute one element of the output matrix C.

    At runtime, the GPU launches one thread per output element (or per tile), meaning 1024×1024 = ~1 million threads. each output element (C[i][j]) is computed by one SIMT thread.

    These threads are grouped into warps of 32. Each warp is scheduled onto an SM.

    Inside the SM, the same kernel instruction is executed across threads via SIMD hardware, each using its own piece of input data.

    This model is called SIMT — Single Instruction, Multiple Threads — which sits on top of SIMD — Single Instruction, Multiple Data.

    SIMT is the programming model: one thread per output value, written like it's independent.
    SIMD is how the GPU really runs it: one instruction across 32 values at once.

    Each thread is assigned to one SIMD lane (One FP32 ALU — handles one thread’s instruction) during that instruction
    All threads in a warp share the SIMD unit (A group of FP32 ALUs that runs 1 instruction on many data), executing in lockstep

This represents a 50–100 × speedup for the GPU, demonstrating why GPUs have become essential for deep learning and scientific computing than CPUs - which process sequentially.
