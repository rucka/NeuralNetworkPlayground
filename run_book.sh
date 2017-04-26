if [ ! "$(docker ps -q -f name=tensor)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=tensor)" ]; then
        # cleanup
        docker rm tensor
    fi
    # run your container
  docker run --name tensor -it -v $HOME/Projects/NeuralNetworkPlayground/notebook:/book:rw -p 8888:8888 -p 6006:6006 -w /book tensorflow/tensorflow sh -c "/run_jupyter.sh --allow-root & tensorboard --logdir=/tmp/tensorflow/mnist/logs/mnist_with_summaries"
fi
#docker run --name tensor -it -v $HOME/Projects/NeuralNetworkPlayground/notebook:/book:rw -p 8888:8888 -p 6006:6006 -w /book tensorflow/tensorflow sh -c "/run_jupyter.sh --allow-root & tensorboard --logdir=`pwd`"

#docker run --name tensor -it -v $HOME/Projects/NeuralNetworkPlayground/notebook:/book:rw -p 8888:8888 -p 6006:6006 -w /book tensorflow/tensorflow sh -c "/run_jupyter.sh --allow-root & tensorboard --logdir=/tmp/tensorflow/mnist/logs/mnist_with_summaries"
