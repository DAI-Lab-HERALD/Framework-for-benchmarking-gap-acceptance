import numpy as np
import pandas as pd
from ROC_curve import ROC_curve 

class ROC_curve_col(ROC_curve):
    def get_t0_type():
        return 'col'
