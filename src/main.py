import gym
import os

from .generate import generate_data
from .model import train_model, test_model
from .protocol import Protocol

cp = 'CartPole-v0'
ll = 'LunarLander-v2'

# the neural net
model        = None
# input_dim: {{{
#
# represents the input dimension our model uses.
# Gets set when initializing the data_set to
# len(data_set[0][0]), the size our observation has.
#
# }}}
input_dim    = None
# global reference to our environment
env          = None
# meta data for each environment {{{
# {{{
#
#   goal_score:
#       highscore
#
#   score_req:
#       minimum score the first
#       data in our data set has
#       to reach
#
#   steps:
#       is used to loop the actions
#       performed in model.train_model
#       and generate.generate_data.
#
#   eps:
#       episodes that are performed
#       every time in
#       generate.generate_data
#       (eps * steps actions are
#       performed).
#
#   goal_cons:
#       how often the agent should
#       reach the goal_score con-
#       secutively before we can
#       say our agent has solved the
#       environment
#
#   action_space:
#       represents the set of actions our
#       agent can perform to this environment.
#       Actually it is just an integer n and
#       the actions the agent can perform are
#       0, 1, ..., n-1
#
# }}}
goal_score   = None
score_req    = None
steps        = None
eps          = None
goal_cons    = None
action_space = None
# }}}

# def main {{{
def main(

    # visual: {{{
    #
    # defines if the gym environment should
    # be rendered (output on screen) or not
    # when the agent is tested (model.test_model).
    # if gym does not have to render it can
    # perform the testing much faster.
    #
    # }}}
    visual = False,

    # env_name: {{{
    #
    # defines the gym environment. Right now
    # cp und ll can be used (others may as
    # well, but aren't jet tested)
    #
    # }}}
    env_name = cp,

    # _model: {{{
    #
    # defines which neural net is imported from
    # $WPFVS_HOME/models. Every module in models
    # has a model-function which returns a compiled
    # keras Sequential. _model is passed to
    # model.train_model as imp parameter when
    # initializing the model (first call of model.
    # train_model
    #
    # }}}
    _model = '64x64',

    # procs: {{{
    #
    # CPython's threads only run on a single core,
    # which is not what we need, instead we need
    # to generate our data using multiple cores,
    # which is only possible with Processes in
    # CPython.
    #
    # }}}
    procs = 1

):
    global model
    global env
    global goal_score
    global score_req
    global steps
    global eps
    global goal_cons
    global action_space
    global input_dim

    # $WPFVS_HOME has to be set when running
    # this program (protocol.Protocol needs it
    # for writing the protocolled meta data to
    # file)
    if not 'WPFVS_HOME' in os.environ:
        raise Exception('ENVIRONMENT VARIABLE WPFVS_HOME is not set')

    # build environment env_name
    #
    # raises error when gym does
    # not know env_name
    env = gym.make(str(env_name))

    if env_name == cp:
        goal_score=200
        score_req=50
        steps=500
        eps = 2000
        goal_cons = 10
    elif env_name == ll:
        goal_score=1000
        score_req=-500
        steps=1000
        eps = 1000
        goal_cons=100

    # mandatory first call of env.reset(),
    # without first calling env.reset(),
    # env.step(action) throws an error
    env.reset()

    # save which environment is played
    Protocol.info['env'] = str(env_name)
    Protocol.info['model'] = _model

    action_space=env.action_space.n

    # initialize model: {{{
    #
    # initialize the model with randomly
    # generated training data

    # collect the training data set
    data_set = generate_data()

    # provide the model which is trained
    # with our random data set
    model = train_model(data=data_set,imp=_model)
    # }}}

    # counts the times the net reaches goal_cons
    # consecutive
    cons = 0

    # training loop: {{{
    #
    # continue training until, when the model
    # is tested, it reaches goal_cons consecutive
    # time goal_score
    #
    while cons < goal_cons:

        score = test_model(visual=visual)

        # if the previous test equals our
        # goal_score, we want to count the
        # times, the net reaches the goal_score
        # consecutively, without having to
        # train it again
        while score == goal_score:

            cons += 1
            print('Consecutive: ', cons)

            # we reached our goal. Our net
            # has solved the environment
            if cons == goal_cons:
                Protocol.dump()
                return

            score = test_model(visual = visual)

        #data_set = generate_data(data_set=data_set)
        data_set = generate_data()

        model = train_model(data=data_set,model=model)

        # our agent has failed. We need to
        # train it again and try to reach
        # the goal_score once again goal_cons
        # time
        cons = 0

    # }}}
# }}}

if __name__ == '__main__':
    main()
