import os
import json

from time import time, gmtime

class Protocol:

    info = {
        'model'  : None,
        'env'    : None,
        'time'   : {
            'single' : {},
            'total'  : {},
        },
    }

    count = 0
    start = time()

    def __init__(self,f):
        self.f = f

    def __call__(self,*args,**kwargs):

        start = time()
        r = self.f(*args, **kwargs)
        end = time()

        # collect total execution time we have spend in a function
        if not str(self.f.__name__) in Protocol.info['time']['total']:
            Protocol.info['time']['total'][
                str(self.f.__name__)
            ] = end - start
        else:
            Protocol.info['time']['total'][
                str(self.f.__name__)
            ] += end - start

        # Push time to Protocol.info['single']
        Protocol.count += 1

        Protocol.info['time']['single'][
            str(Protocol.count) + '. ' + str(self.f.__name__)
        ] = end - start

        return r

    @staticmethod
    def dump():
        Protocol.info['time']['total']['total'] = time() - Protocol.start

        gm_time = gmtime()

        file = open(
            "{}/protocols/{}_{}_STE_{}_{}_{}_{}_{}.json".format(
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
