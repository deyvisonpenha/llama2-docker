# Cloud GPU Environment with LLAMA2-7b-Chat

This image provides an easily adaptable environment for running on cloud servers equipped with GPUs. Developed in Python, this project utilizes the following stack:

- [Langchain](https://github.com/langchain)
- [LLAMA2-7b-Chat](https://github.com/LLAMA2-7b-Chat)
- [Uvicorn](https://www.uvicorn.org/)
- [CTransformers](https://github.com/ctransformers)
- [FastAPI](https://fastapi.tiangolo.com/)

Additionally, this image comes with an SSH service, allowing for easy modification of the model. Ports 22 for SSH and 80 for FastAPI are exposed.

## Usage

1. **Pull the Image**:

    ```bash
    docker pull your_image_name:tag
    ```

2. **Run the Container**:

    ```bash
    docker run -d -p 22:22 -p 80:80 your_image_name:tag
    ```

    Replace `your_image_name:tag` with the name and tag of your Docker image.

3. **Access the Services**:

    - SSH: Connect to port 22 to access SSH.
    - FastAPI: Access your FastAPI service via port 80.

## Customization

To customize the root password, pass the ROOT_PASSWORD environment variable when running the container. If not provided, the default password is 'P@ssW0rd##'.Also, to modify the model or any other aspect of the environment, access the container via SSH and make the necessary changes.

## License

This project is licensed under the [MIT License](LICENSE).
