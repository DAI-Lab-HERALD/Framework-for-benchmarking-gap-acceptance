import numpy as np
import pandas as pd
from scipy import interpolate, optimize, stats

import ddm
from model_template import model_template


class OverlayNonDecisionGaussian(ddm.Overlay):
    """ Courtesy of the pyddm cookbook """
    name = "Add a Gaussian-distributed non-decision time"
    required_parameters = ["ndt_location", "ndt_scale"]

    def apply(self, solution):
        # Extract components of the solution object for convenience
        corr = solution.corr
        err = solution.err
        dt = solution.model.dt
        # Create the weights for different timepoints
        times = np.asarray(list(range(-len(corr), len(corr)))) * dt
        weights = stats.norm(scale=self.ndt_scale, loc=self.ndt_location).pdf(times)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)  # Ensure it integrates to 1
        newcorr = np.convolve(weights, corr, mode="full")[len(corr):(2 * len(corr))]
        newerr = np.convolve(weights, err, mode="full")[len(corr):(2 * len(corr))]
        return ddm.Solution(newcorr, newerr, solution.model,
                            solution.conditions, solution.undec)

class Bound_Implementation(ddm.models.Bound):
    name = "Bounds implemented for this model"
    required_parameters = ["required_conditions", "required_parameters"]
    required_conditions = []
    
    def __init__(self, required_conditions, **kwargs):
        self.required_conditions = required_conditions
        self.required_parameters = ["b_0", "k", "tau_crit"]
        super().__init__(**kwargs)
        
    def get_bound(self, t, conditions, **kwargs):
        if 'tta_condition' in conditions:
            tau = conditions["tta_condition"] - t
        else:
            tau = - t
        return self.b_0 / (1 + np.exp(-self.k * (tau - self.tau_crit)))  

    
class Drift_Implementation_tuple(ddm.models.Drift):
    name = "Drift implemented for this model"
    # pyDDM only allows to overwrite the attributes inside required_parameters, 
    # so the following is a pretty big hack to allow for dynamic conditions
    required_parameters = ["required_conditions", "required_parameters", "dt", "min_i"]
    required_conditions = []
    
    def __init__(self, required_conditions, Input, dt):
        self.required_conditions = required_conditions
        self.dt = dt
        num_conditions = len(required_conditions)
        
        
        
        I = Input.to_numpy()
        self.min_i = np.array([[len(I[i,j]) for j in range(len(I[i]))] for i in range(len(I))]).min() - 1
        min_inputs = np.array([[min(I[i,j]) for j in range(len(I[i]))] for i in range(len(I))]).min(0)
        max_inputs = np.array([[max(I[i,j]) for j in range(len(I[i]))] for i in range(len(I))]).max(0)
        ran_inputs = max_inputs - min_inputs
        abs_inputs = np.maximum(max_inputs, -min_inputs)
        
        required_parameters = ["alpha"]
        parameters = {'alpha': ddm.Fittable(minval = - 5, maxval = 5)}
        ran_theta = 0
        for i in range(1, num_conditions):
            if num_conditions < 3:
                required_parameters.append("beta")
            else:
                required_parameters.append("beta"+str(i)) 
            # 25 is random value which should more or less be enough
            val_b = 25 * ran_inputs[0] / ran_inputs[i] 
            ran_theta = ran_theta + abs_inputs[i] * val_b
            parameters[required_parameters[-1]] = ddm.Fittable(minval = - val_b, maxval = val_b)

        required_parameters.append("theta")
        min_t = 2 * min_inputs[0] - ran_theta * 0.4 
        max_t = 2 * max_inputs[0] + ran_theta * 0.4
        parameters["theta"] = ddm.Fittable(minval = min_t, maxval = max_t)
        
        self.required_parameters = required_parameters
        super().__init__(**parameters)
        
    def get_drift(self, t, conditions, **kwargs):
        # Implement the required formulation for driftrate here
        i = min(int(t / self.dt), self.min_i)
        beta = [getattr(self, key) for key in self.required_parameters[1:-1]]
        conditions_i = [conditions.get(key)[i] for key in self.required_conditions]
        Beta = sum([x1*x2 for x1,x2 in zip(beta,conditions_i[1:])])
        return self.alpha * (conditions_i[0] + Beta - self.theta)      


class LossWLSVincent(ddm.LossFunction):
    name = """Weighted least squares as described in Ratcliff & Tuerlinckx 2002, 
                fitting to the quantile function vincent-averaged per subject (Ratcliff 1979)"""
    rt_quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    rt_q_weights = [2, 2, 1, 1, 0.5]

    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur

    def get_rt_quantiles(self, x, t_domain, exp=False):
        if exp:
            vincentized_quantiles = (self.comb_rts.groupby("subj_id")
                                     .apply(lambda group: np.quantile(a=group.RT, q=self.rt_quantiles))).mean()
            return vincentized_quantiles
        else:
            cdf = x.cdf_corr()
            cdf_interp = interpolate.interp1d(t_domain, cdf / cdf[-1])
            # If the model produces very fast RTs, interpolated cdf(0) can be >0.1, then we cannot find root like usual
            # In this case, the corresponding rt quantile is half of the time step of cdf
            rt_quantile_values = [optimize.root_scalar(lambda x: cdf_interp(x) - quantile, bracket=(0, t_domain[-1])).root
                                  if (cdf_interp(0) < quantile) else self.dt / 2
                                  for quantile in self.rt_quantiles]
            return np.array(rt_quantile_values)

    def loss(self, model):
        solutions = self.cache_by_conditions(model)
        WLS = 0
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            c = frozenset(comb.items())
            #            print(c)
            comb_sample = self.sample.subset(**comb)
            WLS += 4 * (solutions[c].prob_correct() - comb_sample.prob_correct()) ** 2
            a = [item for item in comb_sample.items(correct=True)]
            
            self.comb_rts = pd.DataFrame([[item[0], item[1]["subj_id"]] for item in comb_sample.items(correct=True)],
                                         columns=["RT", "subj_id"])

            # Sometimes model p_correct is very close to 0, then RT distribution is weird, in this case ignore RT error
            if ((solutions[c].prob_correct() > 0.001) & (comb_sample.prob_correct() > 0)):
                model_rt_q = self.get_rt_quantiles(solutions[c], model.t_domain(), exp=False)
                exp_rt_q = self.get_rt_quantiles(comb_sample, model.t_domain(), exp=True)
                WLS += np.dot((model_rt_q - exp_rt_q) ** 2, self.rt_q_weights) * comb_sample.prob_correct()
        return WLS


class ddm_zgonnikov(model_template):
    
    def setup_method(self):
        self.timesteps = max([len(T) for T in self.Input_T_train])
        
        self.Output_A_train = self.Output_A_train.astype('float32')
        self.training_data = self.Input_train.copy()
        self.training_data['RT'] = self.RT_train
        self.training_data['is_go_decision'] = self.Output_train 
        self.training_data['subj_id'] = self.Subj_id
        self.training_sample = ddm.Sample.from_pandas_dataframe(df=self.training_data,
                                                                rt_column_name="RT",
                                                                correct_column_name="is_go_decision")
        
        # Define models 
        self.T_dur = 2.5
        
        self.overlay = OverlayNonDecisionGaussian(ndt_location=ddm.Fittable(minval=0, maxval=1.0),
                                                  ndt_scale=ddm.Fittable(minval=0.001, maxval=0.3))
        

        self.drift = Drift_Implementation_tuple(list(self.Input_names), self.Input_train, self.dt)
        
        self.bound = Bound_Implementation(list(self.Input_names),
                                          b_0=ddm.Fittable(minval = 0.5, maxval = 5),
                                          k=ddm.Fittable(minval = 0.0, maxval = 2),
                                          tau_crit=ddm.Fittable(minval = 0, maxval = 2 * self.T_dur))
        
        self.loss = LossWLSVincent
        
        self.unfitted_model = ddm.Model(name="Static drift defined by initial TTA and d, timedependend bounds",
                               drift=self.drift, noise=ddm.NoiseConstant(noise=1), bound=self.bound,
                               overlay=self.overlay, T_dur=self.T_dur, dt = self.dt)
        
        self.trained = False
        

    def train_method(self, l2_regulization = 0.01):
        # Classify the usable cluster centers one could collapse to 
        
        
        
        
        
        # Multiple timesteps have to be flattened
        fitparams = {'disp': True, 'maxiter': num_iter, 'popsize': 1, 'polish': False, 'tol': 0.0001}
        self.fitted_model = ddm.fit_adjust_model(sample=self.training_sample, model=self.unfitted_model, 
                                               lossfunction=self.loss, verbose=False, fitparams=fitparams)
        
        self.weights_saved = [self.mean, self.xmin, self.xmax, self.structure, DBN_weights]
        
        
    def load_method(self, l2_regulization = 0):
        [self.mean, self.xmin, self.xmax, self.structure, DBN_weights] = self.weights_saved
        
        self.DBN = deep_belief_network(self.structure) 
        
        Y = np.zeros((len(self.Output_A_train), 2))
        
        self.DBN.train(self.mean, Y[[0]], [], int(len(self.mean)/10), 0, 0)
        
        self.DBN.DBN.set_weights(DBN_weights)
        
        # set parameters
        
    def predict_method(self):
        def predict_single_case(model, input_test):
            instance = input_test
            prediction = model.solve(input_test.to_dict())
            return pd.DataFrame([[prediction.prob_correct(), prediction.mean_decision_time()]],columns=["is_go_decision", "RT"])
        
        sim_result = [predict_single_case(self.fitted_model, input_test) for idx,input_test in Input_test.iterrows()]
        sim_result = pd.concat(sim_result)
        
        
        return [Prob]

    def check_input_names_method(self, names, train = True):
        if all(names == self.input_names_train):
            return True
        else:
            return False
     
    
    def get_output_type_class():
        # Logit model only produces binary outputs
        return 'binary'
    
    def get_name(self):
        return 'ddm'
        

        
        
        
        
        
    
        
        
        