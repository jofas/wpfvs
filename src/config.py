# global config

# rabbitmq
HOST          = 'localhost'
DATAQUEUE     = 'data_queue'
MODELEXCHANGE = 'model_exchange'
METAEXCHANGE  = 'meta_exchange'
METAQUEUE     = 'meta_queue'

# channel types
T_QUEUE    = 0
T_EXCHANGE = 1

# message types for receiving
MSG_CONFIG  = 0
MSG_WEIGHTS = 1
# (!) DEPRECATED
MSG_DONE    = 2

# supported environments
cp = 'CartPole-v0'
ll = 'LunarLander-v2'

# cofing for the environment {{{
# {{{
#
# env_name:
#   string representation of our gym environment.
#
# model_name:
#   name of our model (e.g. 64x64).
#
# goal_score:
#   highscore
#
# steps:
#   is used to loop the actions performed in
#   model.train_model and generate.generate_data.
#
# goal_cons:
#   how often the agent should reach the goal_score
#   consecutively before we can say our agent has
#   solved the environment
#
# eps:
#   episodes that are performed every time in
#   generate.generate_data (eps * steps actions are
#   performed).
#
# rand_eps:
#   episodes that are performed by every random generator.
#
# gen_rand:
#   to every proc (generator using our model, not random)
#   we add some random generators (adds new aspects to our
#   training data). We add gen_rand many random generators.
#
# r_take_eps:
#   DATA-SANITATION: we take every generated episodes where
#   the episode's score devided by goal_score  is greater
#   equal r_takes_eps.
#
# r_clean_eps:
#   DATA-SANITATION: if an episode's score devided by
#   goal_score is greater equal r_clean_eps we sanitize the
#   episode's data and take it (if the episode generated a
#   score lower than r_clean_eps we throw it away).
#
# r_clean_cut:
#   DATA-SANITATION: if the episode's score devided by
#   goal_score is between r_clean_eps and r_take_eps we
#   normalize the data of the episode and take only the
#   steps that generated a single score greater equal
#   r_clean_cut.
#
# }}}
env_name     = None
model_name   = None
goal_score   = None
steps        = None
goal_cons    = None
eps          = None
rand_eps     = None
gen_rand     = None
r_take_eps   = None
r_clean_eps  = None
r_clean_cut  = None
# }}}

# def init_conf {{{
def init_conf(env, model=None):
    global cp
    global ll
    global env_name
    global goal_score
    global steps
    global goal_cons
    global goal_score
    global eps
    global rand_eps
    global gen_rand
    global r_take_eps
    global r_clean_eps
    global r_clean_cut
    global model_name

    env_name = env

    if env == cp:
        goal_score   = 200
        steps        = 1000
        goal_cons    = 10
        eps          = 1000
        rand_eps     = 2000
        gen_rand     = 3
        r_take_eps   = 0.95
        r_clean_eps  = 0.2
        r_clean_cut  = -1
    elif env == ll:
        goal_score   = 200
        steps        = 1000
        goal_cons    = 100
        eps          = 100
        rand_eps     = 100
        gen_rand     = 5
        r_take_eps   = 0
        r_clean_eps  = -3
        r_clean_cut  = 0.4
    else:
        raise Exception('INVALID ENVIRONMENT')

    if model != None: model_name = model
# }}}
