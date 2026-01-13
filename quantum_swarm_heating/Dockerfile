FROM python:3.12-slim

# Install dependencies
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir networkx requests

# Copy script
COPY qsh_script.py /qsh_script.py

# Run script
CMD ["python", "/qsh_script.py"]
