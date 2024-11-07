## Prerequisites

1. **Docker**: Ensure Docker is installed and running on your system.
2. **VSCode**: Install Visual Studio Code.

## Step-by-Step Guide

### 1. Clone the Repository

Clone the repository to your local machine:
```sh
git clone https://github.com/safqore/ai-knowledge-search.git
cd ai-knowledge-search
```

### 2. Start the Docker Container
From Git Bash or the terminal, navigate to the root directory of the cloned repository and run the following command to start the Docker container using Docker Compose:

```sh
docker-compose up --build
```

### 3. Open in VSCode
Ensure you have Visual Studio Code installed. Open knowledge-search-environment folder in Visual Studio Code. The .devcontainer folder has been added to the project. Once the container is running, you will see an option in VSCode to "Reopen in Container." Click this option to attach VSCode to the running container.

### 4. Start Developing
Once everything is set up, you can start developing in Python. Open any Python script (e.g., environment-check.py) and run it to ensure everything is configured correctly:

```sh
python environment-check.py
```

### Additional Information
If you need to see all running services or stop the Docker container, you can use the following commands:

```sh
docker ps -a 
docker-compose logs
docker-compose down
```

if wanting to use JupyterLab instead of VSCode, simply nativate to: http://localhost:8888/