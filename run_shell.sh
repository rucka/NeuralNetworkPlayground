if [ ! "$(docker ps -q -f name=tfshell)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=tfshell)" ]; then
        docker rm tfshell
    fi
    docker run --name tfshell -it -v $(pwd)/data:/data -v $(pwd)/models:/models -v $(pwd)/code:/code -v $(pwd)/logs:/logs -v $(pwd)/notebook:/book -p 8888:8888 -w /code tensorflow/tensorflow /bin/bash
fi
