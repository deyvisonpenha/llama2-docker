FROM ubuntu:latest

ENV ROOT_PASSWORD='P@ssW0rd##'

# Install OpenSSH, Nginx and supervisor
RUN apt-get update && apt-get install -y \
    openssh-server \
    supervisor \
    python3 \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Setup SSH server
RUN mkdir /var/run/sshd
RUN echo "root:$ROOT_PASSWORD" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# to avoid issue with key exchange on macos ssh clients
RUN echo "KexAlgorithms=ecdh-sha2-nistp521" >> /etc/ssh/sshd_config

# SSH login fix. Otherwise, the user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# Copy supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create the path for LLM app
RUN mkdir app
WORKDIR /app
COPY . /app

# Initialize a virtualenv
RUN python3 -m venv venv
RUN /bin/bash -c "source venv/bin/activate"

# Install Python dependencies
RUN pip install langchain fastapi uvicorn

# Base ctransformers with ** no ** GPU acceleration
# RUN pip install ctransformers>=0.2.24
# Base ctransformers with no GPU acceleration
RUN pip install ctransformers[cuda]>=0.2.24
# Or with ROCm GPU acceleration
# RUN CT_HIPBLAS=1 pip install ctransformers>=0.2.24 --no-binary ctransformers

# Expose the SSH port and the HTTP port for API endpoint
EXPOSE 22 80

# Start supervisord
CMD ["/usr/bin/supervisord"]
# CMD ["sh", "-c", "uvicorn generativeAIStream:app --reload --port=80 --host=0.0.0.0"]
