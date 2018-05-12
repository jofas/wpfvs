import gym

from .generate import generate_training_data
from .model import train_model, test_model

env = gym.make('CartPole-v0')

goal_score=500
score_req=50
goal_steps=1000
eps = 1000

# def main {{{
def main():

    # mandatory first call of env.reset(),
    # without first calling env.reset(),
    # env.step(action) throws an error
    env.reset()

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
    t, dim = generate_training_data(
        env=env,
        score_req=score_req,
        action_space=action_space,
        eps=eps,
        steps=goal_steps
    )

    # provide the model which is trained
    # with t
    model = train_model(data=t)

    # test the model
    best_run = test_model(
        env=env,
        model=model,
        dim=dim,
        steps=goal_steps
    )
    # }}}

    # training loop: {{{
    # continue training until, when the model
    # is tested, goal_score is reached
    while best_run < goal_score:
        t, _ = generate_training_data(
            env=env,
            _model=(model,dim),
            score_req=score_req,
            action_space=action_space,
            eps=eps,
            steps=goal_steps
        )
        model = train_model(
            data=t,
            model=model
        )
        run = test_model(
            env=env,
            model=model,
            dim=dim,
            steps=goal_steps
        )
        if run > best_run:
            best_run = run
            print(best_run)
    # }}}
# }}}

if __name__ == '__main__':
    main()
