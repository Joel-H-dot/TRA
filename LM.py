import numpy as np
import math as mth
import time

class Levenberg_Marquart():

    def __init__(self, initial_prediction, f0, compute_hessian, compute_gradient, forward_model, d_param=1e-50,
                 lower_constraint=-np.inf, upper_constraint=np.inf, num_iterations=5):



        self.initial_prediction = np.copy(initial_prediction.reshape((len(initial_prediction), 1)))
        self.current_estimate = np.copy(initial_prediction.reshape((len(initial_prediction), 1)))

        self.num_iterations = num_iterations

        self.gamma = d_param

        self.compute_hessian = compute_hessian
        self.compute_gradient = compute_gradient
        self.forward_model = forward_model

        self.lower_constraint = lower_constraint
        self.upper_constraint = upper_constraint
        self.flag_bisection = True

        self.verbose = True
        self.f0 = f0
        self.rho = np.inf

        self.parameter_update = 'Nielson'
        self.damp_type = 'Jacobian'

    def direction(self):

        flag_bisection = True
        if self.verbose:
            print('Finding the damping parameter')
        L = 1

        mult = 2


        while 1:
            hess = np.copy(self.hess)
            grad = np.copy(self.grad)

            if self.damp_type == 'Jacobian':
                DM_square = np.diag(np.diag(hess) / np.max(np.diag(hess)))
            elif self.damp_type == 'Identity':
                DM_square = np.diag(np.ones((len(hess[0, :]), 1)))

            damped_hessian = hess + self.gamma * DM_square

            hessinv = np.linalg.inv(damped_hessian)

            dir = -np.matmul(hessinv, grad)
            step = np.copy(dir)

            param_violations_lower = ((self.current_estimate + dir) <= self.lower_constraint).squeeze()
            param_violations_upper = ((self.current_estimate + dir) >= self.upper_constraint).squeeze()

            param_violations = np.logical_or(param_violations_lower, param_violations_upper).squeeze()

            param_not_to_freeze = [i for i, x in enumerate(~(param_violations)) if x]
            param_to_freeze = [i for i, x in enumerate((param_violations)) if x]

            current_cond = np.logical_or(any((self.current_estimate) < self.lower_constraint) , any(
                (self.current_estimate) > self.upper_constraint))
            next_cond = np.logical_or(any((self.current_estimate + dir) < self.lower_constraint) , any(
                (self.current_estimate + dir) > self.upper_constraint))

            self.flag_bisection = np.logical_and(~current_cond,next_cond).squeeze() #transition from permissible to impermissible only
            freeze = np.logical_and(any(param_violations),~self.flag_bisection).squeeze()

            if self.flag_bisection:
                ########################################
                ###### BISECTION
                ########################################
                condition = any((self.current_estimate + dir) < self.lower_constraint) or any(
                    (self.current_estimate + dir) > self.upper_constraint)
                if condition:
                    low = mth.log10(self.gamma)
                    upper = mth.log10(self.gamma) + 5
                    for i in range(1, 1000):
                        mid = (low + upper) / 2
                        damped_hessian = hess + (10 ** mid) * DM_square
                        hessinv = np.linalg.inv(damped_hessian)
                        dir = -np.matmul(hessinv, grad)
                        condition = any((self.current_estimate + dir) < self.lower_constraint) or any(
                            (self.current_estimate + dir) > self.upper_constraint)
                        if condition:
                            low = np.copy(mid)
                        else:
                            upper = np.copy(mid)

                    self.gamma = 10 ** mid


                    for i in range(0, len(param_to_freeze)):
                        if param_violations_lower[param_to_freeze[i]]:
                                dir[param_to_freeze[i],0]= self.lower_constraint-self.current_estimate[param_to_freeze[i],0]
                        else:
                                dir[param_to_freeze[i],0]= self.upper_constraint-self.current_estimate[param_to_freeze[i],0]

                    step=np.copy(dir)


            elif freeze:

                hess = np.delete(hess,param_to_freeze,axis=0)
                hess = np.delete(hess, param_to_freeze, axis=1)

                grad = np.delete(grad,param_to_freeze, axis=0)


                DM_square = np.diag(np.diag(hess) / np.max(np.diag(hess)))

                dir = -np.matmul(np.linalg.inv(hess + DM_square * self.gamma), grad)

                step = np.zeros((np.shape(self.grad)))

                for i in range(0, len(param_not_to_freeze)):
                    step[param_not_to_freeze[i], 0] = dir[i]

            if self.verbose:
                print('     Gamma = ', self.gamma)

            # Compute gain ratio

            m_0 = np.copy(self.f0)  # anyway to avoid having this?
            m_p = m_0 + np.matmul(np.transpose(dir), grad) + 0.5 * np.matmul(np.transpose(dir),
                                                                                  np.matmul(hess, dir))

            f = self.forward_model(self.current_estimate + step)

            print('f = ', f, ' f0 = ', self.f0)
            self.delta_f = np.asscalar(self.f0) - np.asscalar(f)
            self.delta_m = np.asscalar(m_0) - np.asscalar(m_p)  # want this to be >0



            if L == 5 or self.delta_m == 0:  # timout condition

                dir_zeros = np.zeros((len(dir), 1))
                return dir_zeros
            else:
                self.rho = self.delta_f / self.delta_m

                if self.verbose:

                    print('          Rho = ', np.squeeze(self.rho), ' | delta f = ', np.squeeze(self.delta_f),
                          ' | delta m =',
                          np.squeeze(self.delta_m))
                    print('----------------------------------------------------------------------------------------')
                    print('----------------------------------------------------------------------------------------')

                if self.rho < 0.8 or self.rho > 1.2:

                    if self.parameter_update == 'Marquardt':
                        self.gamma = self.gamma * 3
                    elif self.parameter_update == 'Nielson':
                        mult = 2 * mult
                        self.gamma = self.gamma * mult


                elif (self.rho > 0.99999 and self.rho < 1) or (self.rho > 1 and self.rho < 1.00001):

                    if self.parameter_update == 'Marquardt':
                        self.gamma = self.gamma / 2
                    elif self.parameter_update == 'Nielson':
                        mult = 2
                        self.gamma = self.gamma * np.max(np.array([1 / 3, 1 - (2 * self.rho - 1) ** 3]))
                else:
                    self.f0 = f
                    return step
            L = L + 1





    def compute_variables(self):
        self.hess = self.compute_hessian(self.current_estimate)
        self.grad = self.compute_gradient(self.current_estimate)

    def optimisation_main(self):


        for i in range(0, self.num_iterations):

            self.compute_variables()

            dir = self.direction()
            if any(dir==0):
                return self.current_estimate

            self.current_estimate = self.current_estimate + dir


        return self.current_estimate

