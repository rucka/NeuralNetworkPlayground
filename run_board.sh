if [ ! "$(docker ps -q -f name=board)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=board)" ]; then
        docker rm board
    fi
  docker run --name board -it -v $(pwd)/logs:/logs -p 6006:6006 -w /book tensorflow/tensorflow sh -c "tensorboard --logdir=/logs/$1"
fi
