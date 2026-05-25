FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip cmake ninja-build bc \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch jax[cuda12]

WORKDIR /cuda_swap
COPY . .

RUN cmake -S . -B build -DCUDA_INCLUDE_DIR=/usr/local/cuda/include && \
    cmake --build build -j$(nproc)

CMD ["bash"]
