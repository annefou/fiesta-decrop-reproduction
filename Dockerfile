FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        p7zip-full \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "tensorflow==2.19.0" \
        "protobuf>=3.20,<6.0" \
        "numpy>=1.26,<2.2" \
        "opencv-python>=4.5" \
        "albumentations~=2.0.5" \
        scikit-learn \
        pandas \
        pyyaml \
        pillow \
        requests \
        tqdm \
        matplotlib \
        jupytext \
        nbclient \
        ipykernel \
        zenodo-get \
        "planktonclas==0.2.0"

WORKDIR /app
COPY . /app

CMD ["sh", "-c", "jupytext --to notebook 01_reproduce_decrop.py && jupyter execute --inplace 01_reproduce_decrop.ipynb"]
