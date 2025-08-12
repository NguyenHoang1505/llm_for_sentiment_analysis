docker build \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    -t sa .

docker run -it --rm --gpus all -v $(pwd)/models:/app/models -v ~/.cache:/root/.cache -v $(pwd)/output:/app/output sa /bin/bash

