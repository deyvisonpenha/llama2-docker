FROM python

RUN mkdir app
WORKDIR /app

COPY . /app

# Initialize a virtualenv
RUN python3 -m venv venv
RUN /bin/bash -c "source venv/bin/activate"

# Install Python dependencies
RUN pip install langchain llama-cpp-python fastapi uvicorn

