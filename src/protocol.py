import os
import json

from time import time, gmtime

class Protocol:

    info = {
        'model'      : None,
        'env'        : None,
        'procs'      : None,
        'visual'     : None,
        'goal_score' : None,
        'steps'      : None,
        'eps'        : None,
        'goal_cons'  : None,
        'loop'       : [

            # example
            '''
            {
            'loop_number'    : None,
            'data_set'       : None,
            'cons'           : None,
            'score'          : None,
            'avg_score'      : None,
            },
            '''

        ],
        'time'       : {
            'single' : {},
            'total'  : {},
        },
    }

    exec_number = 0
    loop_number = 0
    start = time()

    def __init__(self,f):
        self.f = f

    # def __call__ {{{
    def __call__(self,*args,**kwargs):

        start = time()
        r = self.f(*args, **kwargs)
        end = time()

        # collect total execution time we have
        # spend in a function
        if not str(self.f.__name__) in Protocol \
            .info['time']['total']:

            Protocol.info['time']['total'][
                str(self.f.__name__)
            ] = end - start

        else:
            Protocol.info['time']['total'][
                str(self.f.__name__)
            ] += end - start

        # Push time to Protocol.info['single']
        Protocol.exec_number += 1

        Protocol.info['time']['single'][
            str(Protocol.exec_number) + '. ' + \
            str(self.f.__name__)
        ] = end - start

        return r
    # }}}

    # def save_loop {{{
    @staticmethod
    def save_loop(score, cons, data_set):

        avg_score = (
            Protocol.info['goal_score'] * cons + score
        ) / ( cons + 1)

        Protocol.loop_number += 1

        Protocol.info['loop'].append({
            'loop_number'  : Protocol.loop_number,
            'data_set'     : [{
                'observation': [ i for i in x[0] ],
                'action':x[1],
                'score':x[2]
            } for x in data_set ],
            'cons'         : cons,
            'score'        : score,
            'avg_score'    : avg_score,
        })
    # }}}

    # def dump {{{
    @staticmethod
    def dump():
        Protocol.info['time']['total']['total'] \
            = time() - Protocol.start


        gm_time = gmtime()

        file = open(
            "{}/protocols/{}_{}_{}_{}_{}_{}_{}.json" \
                .format(
                    os.environ['WPFVS_HOME'],
                    Protocol.info['env'],
                    Protocol.info['model'],
                    gm_time.tm_year,
                    gm_time.tm_mon,
                    gm_time.tm_mday,
                    gm_time.tm_hour,
                    gm_time.tm_min
                ),
            'w+'
        )
        file.write(
            json.dumps(
                Protocol.info, sort_keys=True, indent=2
            )
        )
    # }}}
