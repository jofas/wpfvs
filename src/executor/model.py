import random
import importlib
import numpy as np

from concurrent import futures

# def train_model {{{
# {{{
#
# data:
#   the data set used for training
#   our model
#
# }}}
def train_model(X, y, model=None):

    from .main import import_

    if import_ == None:
        raise Exception(
            'Model to import is not defined'
        )

    if model == None:
        # initialize model
        print('initializing model...')
        model = importlib.import_module(
            'models.'+import_
        ).model(
            input_size = len(X[0]),
            output_size = len(y[0])
        )

    print('beginning actual training...')
    model.fit(X,y,epochs=5,batch_size=1024)

    return model
    # }}}
# }}}

# def test_model {{{
# {{{
#
# visual:
#   boolean whether gym should
#   render the environment or
#   not
#
# }}}
def test_model(visual, model):

    # imports {{{
    from .main    import action_space, env
    from ..config import goal_score, steps, goal_cons
    # }}}

    cons = 0

    # run tests while the goal_score is reached {{{
    while True:

        score = 0
        prev_obs = []

        # run test {{{
        for _ in range(steps):
            if visual:
                env.render()

            if prev_obs == []:
                action = random.randrange(0,action_space)
            else:
                # predict action
                action = np.argmax(
                    model.predict(
                        np.array([prev_obs])
                    )[0]
                )

            new_observation, reward, done, info = \
                env.step(action)

            prev_obs = new_observation
            score+=reward

            if done:
                env.reset()
                print('Score: ', score)
                break
        # }}}

        print('Consecutive: ', cons)

        if score == goal_score:
            cons += 1
        else:
            return ( False, score, cons )

        if cons == goal_cons:
            return ( True, score, cons )
    # }}}
# }}}
