import numpy as np
import pandas as pd
from Miss_rate import Miss_rate 

class Miss_rate_col(Miss_rate):
    def get_t0_type():
        return 'col'