if [ ! "$(docker ps -q -f name=tensor)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=tensor)" ]; then
        docker rm tensor
    fi
    docker run --name tensor -it -v $(pwd)/logs:/logs -v $(pwd)/notebook:/book -p 8888:8888 -w /book tensorflow/tensorflow sh -c "/run_jupyter.sh --allow-root"
fi
