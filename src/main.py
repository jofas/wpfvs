import gym

from .model import train_model, test_model

env = gym.make('CartPole-v0')

goal_score=500
goal_steps=750

# def main {{{
def main():

    # mandatory first call of env.reset(),
    # without first calling env.reset(),
    # env.step(action) throws an error
    env.reset()

    #t = generate_training_data()

    #model = train_model(t)

    model = train_model(env=env)
    best_run = 0

    while best_run < goal_score:
        model = train_model(
            env=env,
            model=model,
            best_test_run=best_run
        )
        run = test_model(env=env,model=model)
        if run > best_run:
            best_run = run
            print(best_run)
# }}}

if __name__ == '__main__':
    main()
