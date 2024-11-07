FROM continuumio/anaconda3

# Create conda environment based on software specified in environment.yml
WORKDIR /knowledge-search
COPY environment.yml /knowledge-search/environment.yml
RUN conda env create -f environment.yml

# Initialize conda
RUN conda init bash

# Shell override the default shell used in subsequent RUN, CMD, and ENTRYPOINT instructions in the Dockerfile.
SHELL ["conda", "run", "-n", "knowledge-search-env", "/bin/bash", "-c"]

# Copy all code to work directory in conda environment
COPY . /knowledge-search

# Setup environment for any future interactive shell sessions to start in conda environment. 
RUN echo "conda activate knowledge-search-env" >> ~/.bashrc

# Set environment variables for the conda environment
ENV PATH=/opt/conda/envs/knowledge-search-env/bin:$PATH

# Expose port to Jupyter Lab
EXPOSE 8888

# Set the default command to run Jupyter Lab in the conda environment and keep container running
CMD ["bash", "-c", "conda run -n knowledge-search-env jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]