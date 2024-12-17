# Parallel Computing for Sentiment Analysis Model
SNU 확장형고성능컴퓨팅 Final Project

## Model Structure
![result](./sentiment_analysis_model_structure.png)

## GPU Information
- Name: NVIDIA TITAN RTX   
- CUDA Runtime Version: 11.8
- Compute Capability: 7.5
- Total number of SMs: 72   
- Max threads per block: 1024 (Less than 1024)   
- Max threads per multiprocessor: 1024
- Threads per warp: 32
- Max regs per block: 65536
- Max regs per multiprocessor: 65536
- Total global mem: 24220 MB
- Max shared mem per block: 48 KB
- Shared mem per multiprocessor: 64 KB
- Max warps per multiprocessor: 32

## Performance Experiments
|ID|Kernel Optimization|Communication Optimization|Throughput(sentences/sec)|Commit Number|
|:-:|:-|:-|:-:|:-:|
|1|Naive Conv1D|Sequential Conv1Ds|686|43562e5|
|2|Conv1D to Input Spread and Matrix WMMA (Tensor core)|Sequential Conv1Ds|4710|9635014|
|3|Add a & b SMEM tiling to Matrix WMMA Kernel|Sequential Conv1Ds|9622|07acaef|
|4|ID3|Use 4 nodes (MPI Scatter/Gather)|29484|afd88a5|
|4|ID3|Logically Concurrent CONV1Ds|29437||

### Performance Experiments ID3 Parameter Tuning
|ID|WMMA_BLOCKDIM|TSKA|TSKB|Throughput(sentences/sec)|
|:-:|:-:|:-:|:-:|:-:|
|0|512|32|256|9622|
|1|512|32|512|9702|