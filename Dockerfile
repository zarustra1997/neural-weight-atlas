# Neural Weight Atlas fora do Mac: refaz o corte do gpt-oss-20b e roda o modelo no Ollama, na GPU quando houver.
# Versões fixas para sair o mesmo arquivo do Mac: Ollama 0.40.1 e o llama.cpp que ele traz (commit 631109b3).
# Use pelo nwa.py, que escolhe o perfil (cpu, nvidia ou amd) do compose.yaml.
ARG OLLAMA_IMAGE=ollama/ollama:0.40.1

FROM ${OLLAMA_IMAGE} AS quantize
ARG LLAMA_CPP_COMMIT=631109b34da437a3c4a5ebd75091d677671392e3
# Mais tarefas em paralelo estouram a memória de um Docker Desktop de 8 GB.
ARG BUILD_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_CPP_COMMIT}.tar.gz" | tar -xz -C /tmp \
    && cmake -S "/tmp/llama.cpp-${LLAMA_CPP_COMMIT}" -B /build -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
    && cmake --build /build --target llama-quantize -j "${BUILD_JOBS}"

FROM ${OLLAMA_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends python3 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=quantize /build/bin/llama-quantize /usr/local/bin/llama-quantize
WORKDIR /nwa
COPY *.py ./
COPY results/experts/usage.json results/experts/
COPY results/bench120/questions.json results/bench120/
COPY results/holdout/holdout_H01_H30.json results/holdout/
ENV LLAMA_QUANTIZE=/usr/local/bin/llama-quantize OLLAMA_MODELS=/root/.ollama/models NWA_IN_DOCKER=1
ENTRYPOINT ["python3", "-B", "nwa.py"]
CMD ["build"]
