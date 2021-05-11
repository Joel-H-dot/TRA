import numpy as np
import math as mth


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

        self.t_elapsed = 0
        self.progress = 0

    def direction(self):

        flag_bisection = True
        if self.verbose:
            print('Finding the damping parameter')
        L = 1

        mult = 2

        while 1:

            if self.damp_type == 'Jacobian':
                DM_square = np.diag(np.diag(self.hess) / np.max(np.diag(self.hess)))
            elif self.damp_type == 'Identity':
                DM_square = np.diag(np.ones((len(self.hess[0, :]), 1)))

            damped_hessian = self.hess + self.gamma * DM_square

            hessinv = np.linalg.inv(damped_hessian)

            dir = -np.matmul(hessinv, self.grad)

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
                        damped_hessian = self.hess + (10 ** mid) * DM_square
                        hessinv = np.linalg.inv(damped_hessian)
                        dir = -np.matmul(hessinv, self.grad)
                        condition = any((self.current_estimate + dir) < self.lower_constraint) or any(
                            (self.current_estimate + dir) > self.upper_constraint)
                        if condition:
                            low = np.copy(mid)
                        else:
                            upper = np.copy(mid)

                    self.gamma = 10 ** mid

            damped_hessian = self.hess + self.gamma * DM_square
            hessinv = np.linalg.inv(damped_hessian)
            dir = -np.matmul(hessinv, self.grad)

            if self.verbose:
                print('     Gamma = ', self.gamma)

            # Compute gain ratio

            m_0 = np.copy(self.f0)  # anyway to avoid having this?
            m_p = m_0 + np.matmul(np.transpose(dir), self.grad) + 0.5 * np.matmul(np.transpose(dir),
                                                                                  np.matmul(self.hess, dir))

            f = self.forward_model(self.current_estimate + dir)


            delta_f = self.f0 - f
            delta_m = m_0 - m_p  # want this to be >0

            self.rho = delta_f / delta_m

            if self.verbose:

                print('          Rho = ', np.squeeze(self.rho), ' | delta f = ', np.squeeze(delta_f), ' | delta m =',
                      np.squeeze(delta_m))
                print('----------------------------------------------------------------------------------------')
                print('----------------------------------------------------------------------------------------')

            if L == 5:  # timout condition

                dir_zeros = np.zeros((len(dir), 1))
                return dir_zeros
            else:
                if self.rho < 0.8 or self.rho > 1.2:

                    if self.parameter_update == 'Marquardt':
                        self.gamma = self.gamma * 3
                    elif self.parameter_update == 'Nielson':
                        mult = 2 * mult
                        self.gamma = self.gamma * mult


                elif (self.rho > 0.99999 and self.rho < 1) or (self.rho > 1 and self.rho < 1.00001):

                    if self.parameter_update == 'levenberg':
                        self.gamma = self.gamma / 2
                    elif self.parameter_update == 'nielson':
                        mult = 2
                        self.gamma = self.gamma * np.max([1 / 3, 1 - (2 * rho - 1) ** 3])
                else:
                    self.f0 = f
                    return dir
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

