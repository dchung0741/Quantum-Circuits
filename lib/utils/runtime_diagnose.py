import cProfile
import pstats
from typing import Callable, Any

def diagnose(f: Callable[[Any], Any]):

    def wrapper(*args, **kwargs):

        with cProfile.Profile() as pr:
            res = f(*args, **kwargs)
        
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
    
        return res
    
    return wrapper


if __name__ == '__main__':

    @diagnose
    def tmpF(x):
        return 1
    

    y = tmpF(3)
    print(y)