# global config

# rabbitmq meta
HOST          = 'localhost'
DATAQUEUE     = 'data_queue'
MODELEXCHANGE = 'model_exchange'

# message types for receiving
MSG_CONFIG  = 0
MSG_WEIGHTS = 1
MSG_DONE    = 2

# supported environments
cp = 'CartPole-v0'
ll = 'LunarLander-v2'
