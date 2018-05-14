import gym
import os

from .generate import generate_data
from .model import train_model, test_model
from .bench import Bench

cp = 'CartPole-v0'
ll = 'LunarLander-v2'

# def main {{{
def main(

    # test: {{{
    # defines which neural net is used.
    # if test is True, a smaller net will
    # be used, so testing changes
    # made to this program other than the
    # actual agent can be done faster
    # }}}
    test = False,

    # visual: {{{
    # defines if the gym environment should
    # be rendered (output on screen) or not
    # when the agent is tested (model.test_model).
    # if gym does not have to render it can
    # perform the testing much faster.
    # }}}
    visual = False,

    # env_name: {{{
    # defines the gym environment. Right now
    # cp und ll can be used (others may as
    # well, but aren't jet tested)
    # }}}
    env_name = cp,

    # bench: {{{
    # defines if we want to benchmark the
    # process or not
    # }}}
    bench = False
):

    # $WPFVS_HOME has to be set when running
    # this program (bench.Bench needs it for
    # writing the benchmark to file)
    if not 'WPFVS_HOME' in os.environ:
        raise Exception('ENVIRONMENT VARIABLE WPFVS_HOME is not set')

    # build environment env_name
    #
    # raises error when gym does
    # not know env_name
    env = gym.make(str(env_name))

    # meta data for each environment {{{
    #
    #   goal_score:
    #       highscore
    #
    #   score_req:
    #       minimum score the first
    #       data in our data set has
    #       to reach
    #
    #   goal_steps:
    #       is used to loop the actions
    #       performed in model.train_model
    #       and generate.generate_data.
    #
    #   eps:
    #       episodes that are performed
    #       every time in
    #       generate.generate_data
    #       (eps * goal_steps actions are
    #       performed).
    #
    #   goal_cons:
    #       how often the agent should
    #       reach the goal_score con-
    #       secutively before we can
    #       say our agent has solved the
    #       environment
    #
    # }}}
    if env_name == cp:
        goal_score=200
        score_req=50
        goal_steps=500
        eps = 1000
        goal_cons = 10
    elif env_name == ll:
        goal_score=1000
        score_req=-500
        goal_steps=1000
        eps = 1000
        goal_cons=100

    # mandatory first call of env.reset(),
    # without first calling env.reset(),
    # env.step(action) throws an error
    env.reset()

    # save the descision if we will benchmark
    # or not this time
    Bench.bench = bench
    # save which environment is played
    Bench.info['env'] =str(env_name)


    # action_space: {{{
    # represents the set of actions our
    # agent can perform to this environment.
    # Actually it is just an integer n and
    # the actions the agent can perform are
    # 0, 1, ..., n-1
    # }}}
    action_space=env.action_space.n

    # initialize model: {{{
    #
    # initialize the model with randomly
    # generated training data

    # collect the training data set and
    # the input dimension dim.
    # dim should only be collected when
    # this function gets called the first
    # time, after that dim does not change,
    # but still gets returned.
    data_set, dim = generate_data(
        env=env,
        score_req=score_req,
        action_space=action_space,
        eps=eps,
        steps=goal_steps
    )

    # provide the model which is trained
    # with our random data set
    model = train_model(data=data_set,test=test)

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

        score = test_model(
            env=env,
            model=model,
            dim=dim,
            steps=goal_steps,
            visual=visual
        )

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
                Bench.dump()
                return

            score = test_model(
                env=env,
                model=model,
                dim=dim,
                steps=goal_steps,
                visual = visual
            )

        data_set, _ = generate_data(
            env=env,
            _model=(model,dim),
            data_set=data_set,

            # set the new score_req to the average
            # of our tests.
            score_req= (
                (score + goal_score * cons) / ( cons + 1 )
            ),

            action_space=action_space,
            eps=eps,
            steps=goal_steps
        )

        model = train_model(
            data=data_set,
            model=model,
        )

        # our agent has failed. We need to
        # train it again and try to reach
        # the goal_score once again goal_cons
        # time
        cons = 0

    # }}}
# }}}

if __name__ == '__main__':
    main()
