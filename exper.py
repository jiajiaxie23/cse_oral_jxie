import yaml
import pandas as pd
import numpy as np
from mixedNegativeBinomial import MixNegBinomials
#from mixedNegativeBinomial import MixNegBinomials
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

def experiment(st, xi= 2, relax= 0.35):


    reject = True

    _dict = dict()

    with open('configs.yaml', 'r') as file:
        configs = yaml.safe_load(file)


    df = pd.read_csv(configs['data_path'], sep=',')
    if len(st) ==0:
        pass
    else: 
        df = df[df['state_x'].isin(st)]
    X = df[configs['variables'][1:]].values
    y = df[configs['variables'][0:1]].values.squeeze()


    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    N,_ = X.shape
    z = np.ones((N,1))
    X = np.append(X, z, axis=1)



    # X_train, X_test, y_train, y_test = train_test_split(
    #       X, y, test_size=0.33, random_state=42)


    X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.33)



    model = MixNegBinomials(
                alpha=configs['alpha'],
                xi = xi,
                max_iter = configs['max_iter'],
                E_max_iter=configs['E_max_iter'],
                tol=configs['tol'],
                mix_num = configs['mix_num'],
                random_init = configs['random_init'],
                _vars = np.array(configs['variables']),
                lr= configs['lr'],
                vars_tp1 = np.array(configs['solar_right']),
                vars_tp2 =  np.array(configs['solar_topology']),
                verbose=configs['verbose'],
                verbose_interval= configs['verbose_interval'],
                lsq_solver = configs['lsq_solver']
    )

    obj_lower_bound = - 10.4
    #-5.4


    # model_adv = MixNegBinomials(
    #             alpha=configs['alpha'],
    #             xi = xi,
    #             max_iter = configs['max_iter'],
    #             E_max_iter=configs['E_max_iter'],
    #             tol=configs['tol'],
    #             mix_num = 1,
    #             random_init = configs['random_init'],
    #             _vars = np.array(configs['variables']),
    #             vars_tp1 = np.array(configs['solar_right']),
    #             vars_tp2 =  np.array(configs['solar_topology']),
    #             verbose=configs['verbose'],
    #             verbose_interval= configs['verbose_interval'],
    #             lsq_solver = configs['lsq_solver']
    # )

    # model_adv.fit(X_train,y_train, X_test= X_test, y_test = y_test, evaluate = True)
    model.fit(X_train,y_train, X_test= X_test, y_test = y_test, evaluate = True)
    #print(model.lower_bound)
    #if model.converged_  and model_adv.converged_:
    if model.converged_ :
        # print(np.abs(model.lower_bound - model_adv.lower_bound) )
        # print((1+relax) *model_adv.lower_bound)
        # if np.abs(model.lower_bound - model_adv.lower_bound) < np.abs((relax) *model_adv.lower_bound):
        if np.abs(model.lower_bound - obj_lower_bound) < np.abs((relax) *obj_lower_bound):
            _dict['Beta'] = model.Beta.tolist()
            _dict['delta'] =model.delta.tolist()
            _dict['lower_bound'] = model.lower_bound
            _dict['xi'] =xi
            reject = False 
            print("accept")
            return reject, _dict

        else:
            return reject, _dict

    else:
        return reject, _dict






