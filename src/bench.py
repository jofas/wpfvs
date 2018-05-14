import os
import json

from time import time, gmtime

class Bench:

    info = {
        'env'    : None,
        'single' : {},
        'total'  : {}
    }

    count = 0
    start = time()
    bench = False

    def __init__(self,f):
        self.f = f

    def __call__(self,*args,**kwargs):

        if Bench.bench:

            start = time()
            r = self.f(*args, **kwargs)
            end = time()

            # collect total execution time we have spend in a function
            if not str(self.f.__name__) in Bench.info['total']:
                Bench.info['total'][
                    str(self.f.__name__)
                ] = end - start
            else:
                Bench.info['total'][
                    str(self.f.__name__)
                ] += end - start

            # Push time to Bench.info['single']
            Bench.count += 1

            Bench.info['single'][
                str(Bench.count) + '. ' + str(self.f.__name__)
            ] = end - start

        else:
            r = self.f(*args, **kwargs)

        return r

    @staticmethod
    def dump():
        if Bench.bench:
            Bench.info['total']['total'] = time() - Bench.start

            file = open(
                os.environ['WPFVS_HOME'] + \
                    '/benchmarks/single_thread_execution_' + \
                    str(gmtime())+'.json',
                'w+'
            )
            file.write(
                json.dumps(
                    {str(gmtime()):Bench.info}, sort_keys=True, indent=2
                )+'\n'
            )
