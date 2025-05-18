# NVIDIA CUDA on Google Colab

## Setting Up

- Colab → New notebook → Runtime → Change runtime type → Select T4 GPU

### Check GPU
```bash
!nvidia-smi
```
- Invokes Nvidia's System Management Interface (SMI)
- `!` means “run this as a shell command”
- CUDA Version: 12.4 — the highest CUDA toolkit version the driver supports
- GPU 0: Tesla T4 — common on Colab’s free tier
- Memory: 15360 MiB ≈ 15 GiB

### MB vs MiB
| Unit    | Stands for | Bytes     | Based on      |
| ------- | ---------- | --------- | ------------- |
| **MB**  | Megabyte   | 1,000,000 | Decimal (10³) |
| **MiB** | Mebibyte   | 1,048,576 | Binary (2²⁰)  |

## Check CUDA Compiler
```bash
!nvcc --version
```
- `nvcc`: NVIDIA CUDA compiler driver

## What is CUDA?

CUDA (Compute Unified Device Architecture) is a framework from NVIDIA to write C/C++ code that runs on the GPU.

GPUs are good at running **thousands of tiny threads in parallel**. CUDA gives tools to:
- Write GPU functions (kernels)
- Launch hundreds or thousands of threads
- Move data between CPU and GPU

### CUDA C++ feels low-level:
- Manually allocate GPU memory
- Launch GPU threads
- Define and launch GPU kernel functions
- Synchronize with `cudaDeviceSynchronize()`
- Free GPU memory

## CPU vs GPU in LLMs
| Component                  | Language | Description                                                                |
| -------------------------- | -------- | -------------------------------------------------------------------------- |
| **Model code (e.g., GPT)** | Python   | Uses PyTorch or TensorFlow                                                 |
| **Accelerator backend**    | C++/CUDA | Deep inside frameworks, matrix ops are in CUDA                             |

## Minimal CUDA Program (pure kernel printf — may not work in Colab)
```cpp
#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    hello_from_gpu<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

## CUDA Keywords and Concepts
- CUDA extensions like `__global__`, `threadIdx`, `blockIdx` are built-in (no need to import)
- `nvcc` adds headers like `<cuda_runtime.h>`

### `__global__` Keyword
- Originally related to global memory access
- Now used to mark GPU kernels callable from the host (CPU)

| CUDA Keyword | Meaning                              |
| ------------ | ------------------------------------ |
| `__global__` | CPU can call this; runs on GPU       |
| `__device__` | Callable from GPU; runs on GPU       |
| `__host__`   | Runs on CPU (optional)               |

## Built-in CUDA Thread Variables

| Variable       | Meaning                                   |
| -------------- | ----------------------------------------- |
| `threadIdx.x`  | ID of thread within block (0 to N-1)      |
| `blockIdx.x`   | ID of block within grid                   |
| `blockDim.x`   | Number of threads per block               |

## Launch Syntax
```cpp
kernel<<<numBlocks, threadsPerBlock>>>(arguments);
```

Example:
```cpp
<<<2, 64>>>
```
- 2 blocks, each with 64 threads = 128 total threads
- Each block forms 2 warps (64 / 32), total 4 warps

## Why `cudaDeviceSynchronize()`?
- Waits for GPU to finish
- Ensures proper memory visibility
- Where runtime errors appear

## Pure printf doesn’t work well in Colab (output swallowed). Use write-to-buffer pattern:

```cpp
%%bash
cat > hello2_debug.cu << 'EOF'
#include <cstdio>

__global__ void hello_write(int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = tid;
}

int main() {
    const int N = 8;
    int *out;

    cudaMallocManaged(&out, N * sizeof(int));

    hello_write<<<1, N>>>(out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    for (int i = 0; i < N; i++) {
        printf("out[%d] = %d\n", i, out[i]);
    }

    cudaFree(out);
    return 0;
}
EOF
```

## Why This Works:
- Threads write their ID (`tid`) into `out[tid]`
- Host (CPU) prints values later

## Breakdown:
- `blockIdx.x * blockDim.x + threadIdx.x` → unique thread ID across all blocks
- `cudaMallocManaged` → allocates unified memory usable by both CPU & GPU
- `cudaMalloc` → GPU-only memory (needs `cudaMemcpy`)
- `out[tid] = tid` → each thread writes to a different index of array `out`

## PTX and Native Execution

### What is PTX?
- PTX = Parallel Thread Execution
- An intermediate assembly-like language for NVIDIA GPUs

### JIT and Native:
- Drivers JIT PTX → real GPU machine code
- You can **precompile for your GPU** with:
```bash
!nvcc -arch=sm_75 hello2_debug.cu -o hello2_debug
```

### SM Versions
| SM Version | Architecture | GPU Example        |
| ---------- | ------------ | ------------------ |
| `sm_70`    | Volta        | V100               |
| `sm_75`    | Turing       | Tesla T4 (Colab)   |
| `sm_90`    | Hopper       | H100               |

- Use `-arch=sm_75` to ensure native code generation for Tesla T4

---

# 🚀 1-D Vector Addition using CUDA

## ✅ Overview

We will implement vector addition using CUDA by:
- ❌ Not using `cudaMallocManaged` (no unified memory)
- ✅ Using `cudaMalloc`, `cudaMemcpy`
- ✅ Writing a **parallel kernel** where **each thread adds one element**
- ✅ Measuring execution time using `cudaEvent_t`
- ✅ Verifying results on the CPU

---

## 🔢 What is Vector Addition?

- A **vector** is just a 1D array of numbers.
- Vector addition:
  ```text
  A[i] + B[i] = C[i]
  ```

---

## 🧵 One Thread per Element

Instead of using a loop, we assign:
> 🧠 "One thread to handle one index."

This replaces a loop like:
```cpp
for (int i = 0; i < N; ++i)
    C[i] = A[i] + B[i];
```

With a **GPU kernel**:
```cpp
C[thread_id] = A[thread_id] + B[thread_id];
```

---

## 🧩 Grid Sizing

CUDA launches threads in groups:

- `kernel<<<blocks, threads_per_block>>>(...)`
- `threads_per_block` → number of threads per block
- `blocks` → number of blocks
- `Grid: all the blocks together`

### 🔁 Safe formula to compute blocks:

```cpp
int blocks = (N + threads - 1) / threads;
```

### ✅ Example:

- N = 1000, threads = 256  
- blocks = (1000 + 255) / 256 = 5  
- Total threads = 5 × 256 = 1280  
- Threads with `tid >= N` do nothing (`if (tid < N)`)

---

## 🧠 Bitwise Left Shift

```cpp
1 << 20
```

- Means `2^20 = 1,048,576`
- It shifts `1` left by 20 bits. 
- general rule of thumb: 1 << n  ==  2^n
- Used to represent 1 million elements

---

## 💻 Full Code

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  const int N = 1 << 20; // 2^20 = 1,048,576
  size_t size = N * sizeof(float);

  // Allocate CPU Memory
  float *h_a = (float*)malloc(size);
  float *h_b = (float*)malloc(size);
  float *h_c = (float*)malloc(size);

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  // Allocate GPU Memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy inputs to GPU
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // Launch kernel
  int threads = 256;
  int blocks = (N + threads -1) / threads;
  vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);

  // Copy output back to host
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  // Verify result
  bool pass = true;
  for (int i = 0; i < N; ++i) {
      if (fabs(h_c[i] - 3.0f) > 1e-5f) {
          printf("Error at %d: %f\n", i, h_c[i]);
          pass = false;
          break;
      }
  }
  printf("Vector addition: %s\n", pass ? "PASS" : "FAIL");

  // Cleanup
  free(h_a); free(h_b); free(h_c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}
```

---