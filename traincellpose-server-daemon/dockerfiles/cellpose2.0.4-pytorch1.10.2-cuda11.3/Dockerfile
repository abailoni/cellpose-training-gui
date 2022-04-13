FROM anibali/pytorch:1.10.2-cuda11.3-ubuntu20.04

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Create a working directory
WORKDIR /training_data_dir

#TODO: test with copying from dev branch and istalling locally...?
# Install cellpose and streamlit
RUN pip install --no-cache-dir -e "vcs+protocol://github.com/abailoni/cellpose-training-gui#egg=traincellposeserver&subdirectory=traincellpose-server-daemon"

#conda env update -n base -f /app/environment.yml \
# && rm /app/environment.yml \
# && conda clean -ya

# Trigger initial Cellpose model download to cache models.
RUN python -c "import cellpose.models"

# Expose the port to be used to run the application:
EXPOSE 8501

## Copy the streamlit app:
#COPY training_server_app.py ./training_server_app.py

# Create an entry point to make our image executable:
ENTRYPOINT ["streamlit", "run"]
CMD ["python -m traincellposeserver -d /training_data_dir"]
