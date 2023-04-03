import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy import linalg
from scipy.stats import nbinom
#from scipy.special import expit as _sigmoid
from scipy.special import logsumexp
from scipy.special import gamma as _gamma
from scipy.special import comb
import math
from sklearn.metrics import mean_squared_error
from scipy.optimize import check_grad


def _log_scale(x, tol = 1e-6):
    #return np.log(x + tol)
    return np.log(x)


def _logit(vals, xi):
    mu = np.exp(vals)
    p = mu/(xi + mu)
    #print(np.isfinite(p).all())
    return p
    

def _pdf_NB(val, y, xi, tol =  1e-6):
    pp = _logit(val, xi)  # N x J
    combina = comb(y+xi-1, xi - 1) # N x  1
    #combina = _gamma(y+xi)/_gamma(xi)/_gamma(y+1) # N x  1
    probs = np.power(1- pp, xi) *  np.power(pp, y)  * combina # N x J
    #return np.where(probs == 0, tol, probs)
    return probs
    

# def _func(beta, X, y, xi, delta, alpha, tol =1e-10):
#     Vals = np.dot(X, beta.T)
#     Z_probs = _pdf_NB(Vals, y, xi)
#     obj = -np.mean(np.log(Z_probs *delta + tol)) + alpha* np.dot(beta, beta)
#     return obj

def _func(beta, X, y, z, xi, alpha):
    xbta = np.dot(X, beta.T)
    first_term = -np.dot(z*y, xbta)
    lnn = np.log(np.exp(xbta)+ xi)
    second_term = np.dot(z*(y+xi), lnn)
    third_term = alpha*np.linalg.norm(beta)**2
    #third_term =0
    obj = first_term+ second_term + third_term
    return obj



# def _gradient(beta, X, y, z, xi, alpha):
#     ell = np.dot(X, beta.T)
#     ell = np.where(ell > 707, 707, ell)
#     Vals = np.exp(ell)
#     denom = Vals + xi
#     M = z*((y+xi) * Vals/denom  - y )    # N x  1
#     grad = np.dot(X.T, M)
#     #grad = np.dot(X.T, M) + 2*alpha* beta
#     return grad


def _gradient(beta, X, y, z,xi,alpha):
    ell =  np.dot(X, beta.T)
    uu = np.exp(ell)
    grad = np.dot(X.T,xi*z*(uu-y)/(uu + xi))
    return grad + 2* alpha*beta


def grad_check(X, y, z, xi, alpha, beta, epsilon= 1e-16):
    beta_plus = beta + epsilon
    beta_minus = beta - epsilon

    f1 = _func(beta_plus, X, y, z, xi, alpha)
    f2 = _func(beta_minus, X, y, z, xi, alpha)

    grad_approx = (f1-f2)/(2*epsilon)

    grad = _gradient(beta, X, y, z, xi, alpha)
    numerator = np.linalg.norm(grad - grad_approx)
    denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approx)

    diff = numerator/denominator

    if diff < 1e-7:
        print("the gradient is correct")
    else:
        print('The gradient is wrong %.5f' % diff)
    return diff


class MixNegBinomials:
    """Parameters
    ---------- 
    _lambda : float, default=1
    Constant that multiplies the penalty term and thus determines the
    regularization strength. ``_lambda = 0`` is equivalent to unpenalized
    NB. In this case, the design matrix `X` must have full column rank
    (no collinearities).
    Values must be in the range `[0.0, inf)`.
    """
    def __init__(
            self,
            alpha=1.0,
            xi = 10,
            max_iter = 100,
            E_max_iter=100,
            tol=1e-4,
            mix_num = 3,
            verbose=0,
            lb = None,
            ub = None,
            _vars = np.array([]),
            vars_tp1 = np.array([]),
            vars_tp2 = np.array([]),
            random_init = False,
            numerical_esp = 0.0000000001,
            verbose_interval = 1,
            lsq_solver = 'exact',
            optim_method = 'CG'  , #conjugate gradident by default
            lr= 0.0001,
            step_decay = 0.8
            ):
        self.alpha = alpha
        self.xi = xi
        self.max_iter = max_iter
        self.E_max_iter = E_max_iter
        self.random_init = random_init
        self.tol = tol
        self.mix_num = mix_num
        self.verbose = verbose
        self.lb = lb
        self.ub = ub
        self.verbose_interval = verbose_interval
        self.lsq_solver = lsq_solver
        self.numerical_eps = numerical_esp
        self.optim_method = optim_method
        self.lr = lr
        self.step_decay = step_decay


        self.vars = _vars
        self.vars_tp1 = vars_tp1
        self.vars_tp2 = vars_tp2
        if self.mix_num ==1:
            self.inv_mask1 = []
            self.inv_mask2 = []
        else:
            self.inv_mask1 = np.in1d(self.vars[1:], self.vars_tp1, invert = True).nonzero()[0]
            self.inv_mask2 = np.in1d(self.vars[1:], self.vars_tp2, invert = True).nonzero()[0]
        # self.inv_mask1 = []
        # self.inv_mask2 = []
        # self.mask1 = np.in1d(self.vars[1:], self.vars_tp1).nonzero()[0]
        # self.mask2 = np.in1d(self.vars[1:], self.vars_tp2).nonzero()[0]
        # print(self.mask2)
        # print(self.mask1)
        # print(self.inv_mask1)

  
        self.inv_mask = [self.inv_mask1, self.inv_mask2]

        self.lower_bound = -np.inf

    def fit(self, X, y, X_test = None, y_test= None, evaluate= False, sample_weight=None):
        """Fit a mixed Negative Binomial Models.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (N, d)
        Training data.
        y : array-like of shape (N,)
        Target values.
        sample_weight : array-like of shape (N,), default=None
        Sample weights.
        Returns
        -------
        self : object
        Fitted model.
        """

        N, d  = X.shape
        max_lower_bound = -np.inf
        self.converged_ = False

        self._init_params(X,y)
        # print(self.Beta[0,:])
        # print(self.Beta[1,:])

        lower_bound = -np.inf

        for n_iter in range(1,self.max_iter+1): 
            
            #self.step_decay = 1/np.sqrt(n_iter)
            prev_lower_bound = lower_bound
            log_prob_norm, resp_Z = self._e_step(X,y)
            
            self._m_step(X,y, resp_Z) 
            # print("Training set lower bound: %f" % lower_bound)
            # print(lower_bound)
            #change = lower_bound - prev_lower_bound
            if evaluate:
                test_lb, y_hat = self.predict(X_test, y_test)
                lower_bound = test_lb
                #print("Testing set lower bound: %f" % test_lb)
                change = test_lb  - prev_lower_bound
                # print(
                #     "  First component weight: {%.5f}, Second component weight: {%.5f}"
                #     % (self.delta[0],self.delta[1])
                # )
            else:
                lower_bound = self._compute_lower_bound(log_prob_norm)
                change = lower_bound - prev_lower_bound
                
            
            #self._print_verbose_msg_iter_end(n_iter, change)
            # print(lower_bound)
            # print(self.delta)
            if abs(change) < self.tol or change < 0:
                self.converged_ = True
                self.lower_bound = test_lb
                break

        #self._print_verbose_msg_init_end(lower_bound)

        if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
            max_lower_bound = lower_bound
            best_parames = self._get_parameters()
            best_n_iter = n_iter




        if not self.converged_ and self.max_iter > 0:
            print(
                "Initialization did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." )


        return self

    def predict(self, X, y):
        """Fit a mixed Negative Binomial Models.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (N, d)
        Training data.
        Returns
        -------
        y_pred: array-like of shape (N,)
        """

        log_prob_norm, resp_Z = self._e_step(X,y)
        lower_bound = self._compute_lower_bound( log_prob_norm)
        pred_m = np.exp(np.dot(X, self.Beta.T)) * self.delta.squeeze()

        return lower_bound, pred_m.sum(axis = 1)

    def _init_params(self, X,y):



        _, self.d = X.shape 
        # ii_1 = np.ones(self.d) 
        # ii_2 = np.ones(self.d) 
        # ii_1[self.inv_mask1] = 0
        # ii_2[self.inv_mask2] = 0 
        # self.proj_mat1 = np.diag(ii_1)
        # self.proj_mat2 = np.diag(ii_2)
        # self.proj_mats = [self.proj_mat1,self.proj_mat2]


        # self.mask1= np.append(self.mask1, [self.d -1])
        # self.mask2 = np.append(self.mask2, [self.d -1])
        # self.mask = [self.mask1, self.mask2]


        #initialize paratemeters
        #self.Beta = np.ones((self.mix_num, self.d)) 
        # init__ = [0.14200029, -0.02211038,  0.61406927,  0.52149264 , 0.34268369,  0.39455948,
        #             0.34750198,  0.43677533,  0.32923305,  0.28899321,  0.56115418,  0.60765282,
        #             0.15199311,  0.66703328,  0.70629813 , 0.11957884, -0.15091862]
        # self.Beta = np.array([init__, init__])
        #self.delta = np.ones(self.mix_num)/self.mix_num
        if self.random_init == False:
            self.delta = np.ones(self.mix_num)/self.mix_num
            #self.Beta = np.ones((self.mix_num, self.d)) 
            
        else:
            self.delta = np.random.dirichlet(np.ones(self.mix_num),size=1).squeeze()
            self.Beta = np.random.randn(self.mix_num, self.d)

        
        self._force_zeros()

        return None


    def _resp_check(self,X,y):
        self._init_params(X,y)
        _, z = self._e_step(X,y)
        return z

    def _grad_check(self,X,y,z,beta):

        grad_check(X, y, z, self.xi, self.alpha, beta)

        return None

    def _force_zeros(self):

        for j in range(self.mix_num):
            self.Beta[j,self.inv_mask[j]] = 0
        return None


    def _e_step(self, X, y):
        """E step.
        Parameters
        ----------
        X : array-like of shape (N, d)
        y : array-like of shape (N, )
        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X
            1/N log( p(y|X) )
        log_responsibility_Z : array, shape (N, J)
            Logarithm of the posterior probabilities of Z (or responsibilities) of
            the point of each sample in X.
        """

        #compute the posterior probabilities of Z
        log_prob_norm_Z, resp_Z = self._estimate_log_prob_resp(X,y)
        # print('by log scale')
        # print(np.isfinite(log_resp_Z).all())


        #return np.mean(log_prob_norm_Z), log_resp_Z, W_bar 
        return log_prob_norm_Z, resp_Z

    def _m_step(self, X, y, resp):

        """ M step.
        Parameters
        ----------
        X : array-like of shape (N, d)
        y : array-like of shape (N, )
        log_prob_norm : float
                Mean of the logarithms of the probabilities of each sample in X
                1/N log( p(y|X) )
        responsibility_Z : array, shape (N, J)
            Logarithm of the posterior probabilities of Z (or responsibilities) of
            the point of each sample in X.
        """

        N,d  = X.shape
        #update parameters delta
        #resp = np.exp(log_resp_Z)
        nk = resp.sum(axis=0)  
        
        nw_delta = nk
        nw_delta /= nw_delta.sum()

        #self.delta = self.delta*self.step_decay + nw_delta*(1 - self.step_decay)
        #self.delta = self.delta*self.lr + nw_delta*(1 - self.lr)

        self.delta = nw_delta
        #update parameters Beta
        for j in range(self.mix_num):

            #self._grad_check(X[:,self.mask[j]],y,resp[:,j],self.Beta[j,self.mask[j]])
            # error = check_grad(_func, _gradient, self.Beta[j,:], X, y, resp[:,j],self.xi,self.alpha) 
            # print(error)
            if self.mix_num ==1:
                res = minimize(_func,  
                    x0 = self.Beta[j,:],
                    jac = _gradient, 
                    args = (X, y, resp[:,j], self.xi, self.alpha),
                    method=self.optim_method,
                    #method = 'L-BFGS-B',
                    tol = self.numerical_eps,
                    options = {'maxiter':self.E_max_iter}
                )



                self.Beta[j,:] = res.x
            else:
                for tt in range(2):
                    update = self.Beta[j,:] - self.lr * _gradient(self.Beta[j,:], X, y, resp[:,j], self.xi,self.alpha)
                
                    update[self.inv_mask[j]] =0
                    self.Beta[j,:] = update

            #print(_gradient(res.x, X, y, resp[:,j],self.xi,self.alpha))
            # for kk in range(1):
            #      # sln  = self.Beta[j,:] - self.lr * _gradient(self.Beta[j,:], X, y, resp[:,j], self.xi,self.alpha)
            #      # self.Beta[j,:] = np.linalg.lstsq(self.proj_mats[j], sln, rcond=None)[0]
            #     self.Beta[j,:] -= self.lr * _gradient(self.Beta[j,:], X, y, resp[:,j], self.xi,self.alpha)
        #self._force_zeros()
        #print(self.Beta)





    def _estimate_weighted_resp(self,X, y):
        """Estimate the responsibilities of negative binomial probability.
        Parameters
        ----------
        X : array-like of shape (N, d)
        y : array-like of shape (N, )
        -------
        resp : array, shape (N, J)
        """
        Vals = np.dot(X, self.Beta.T)
        Y = np.tile(y, (self.mix_num, 1)).T      
        Z_prob = _pdf_NB(Vals,Y, self.xi, tol = self.numerical_eps)

        resp = Z_prob * self.delta.squeeze()

        # vals1 = np.dot(X[:,self.mask[0]], self.Beta[:,self.mask[0]].T)
        # vals2 = np.dot(X[:,self.mask[1]], self.Beta[:,self.mask[1]].T)

        # z_1 = _pdf_NB(vals1, Y, self.xi, tol = self.numerical_eps)
        # z_2 = _pdf_NB(vals2, Y, self.xi, tol = self.numerical_eps)
        # resp = self.delta[0]*z_1 + self.delta[1]* z_2
        # print('resp shape')
        # print(resp.shape)
        return resp


    def _estimate_log_prob_resp(self, X, y):
        """Estimate log probabilities and responsibilities for each sample.
        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.
        Parameters
        ----------
        X : array-like of shape (N, d)
        y : array-like of shape (N, )
        Returns
        -------
        log_prob_norm : array, shape (N,)
            log p(y|X)
        responsibilities of Z: array, shape (N, J)
            logarithm of the responsibilities
        """
        #weighted_log_prob = self._estimate_weighted_log_prob(X, y)
        weighted_prob = self._estimate_weighted_resp(X,y)
        weighted_log_prob = _log_scale(weighted_prob + self.numerical_eps)
        #log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_resp_norm = weighted_log_prob.mean()
        resp = weighted_prob/weighted_prob.sum()
        # with np.errstate(under="ignore"):
        #     # ignore underflow
        #     log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        #return log_prob_norm, log_resp
        return log_resp_norm, resp






    def _compute_lower_bound(self, log_prob_norm):
        return log_prob_norm 


    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print(
                    "  Iteration %d\t time lapse %.5fs\t ll change %.5f"
                    % (n_iter, cur_time - self._iter_prev_time, diff_ll)
                )
                self._iter_prev_time = cur_time


    def _print_verbose_msg_init_end(self, ll):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.verbose >= 2:
            print(
                "Initialization converged: %s\t time lapse %.5fs\t ll %.5f"
                % (self.converged_, time() - self._init_prev_time, ll)
            )



    def _get_parameters(self):
        return (
            self.delta,
            self.Beta
        )


