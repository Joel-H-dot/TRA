import numpy as np
import math as mth
import time


class powells_dogleg():

    def __init__(self, initial_prediction, f0, compute_hessian, compute_gradient, forward_model,
                 constraint_region=np.inf,lower_constraint = -np.inf, upper_constraint = np.inf, num_iterations=5):


        self.initial_prediction = np.copy(initial_prediction.reshape((len(initial_prediction), 1)))
        self.current_estimate = np.copy(initial_prediction.reshape((len(initial_prediction), 1)))

        self.num_iterations = num_iterations

        self.gamma = constraint_region

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

            DM = DM_square**0.5
            S = np.linalg.inv(DM)

            #### Compute Hessian and gradient in a scaled space
            hessian_scaled = np.matmul(np.transpose(S),np.matmul(hess,S))
            grad_scaled = np.matmul(np.transpose(S),grad)
            hessinv = np.linalg.inv(hessian_scaled)

            #### Compute directions in scaled space

            ideal_step = (np.matmul(np.transpose(grad_scaled), grad_scaled)) / (
                np.matmul(np.transpose(grad_scaled), np.matmul(hessian_scaled, grad_scaled)))

            GN_dir = -np.matmul(hessinv, grad_scaled)
            SD_dir = -(grad_scaled) / np.linalg.norm(grad_scaled)


            # CR will update over time, if the new iterate has a poor model agreement, CR will contract -> it is updated
            # from the point of view of the new space, basically self.gamma_new is actually the constraint region in the transformed system

            if np.linalg.norm(GN_dir) < np.linalg.norm(self.gamma):

                dir= np.matmul(S,GN_dir)

            elif np.linalg.norm(SD_dir) > np.linalg.norm(self.gamma):

                dir= np.matmul(S,SD_dir/ np.linalg.norm(SD_dir)) * self.gamma

            else:
                a = ideal_step * SD_dir
                b = GN_dir
                c = np.matmul(np.transpose(a),(b-a))

                aq = np.linalg.norm(b - a) ** 2
                bq = 2 * c
                cq = np.linalg.norm(a) ** 2 - self.gamma ** 2

                r1 = (-bq + mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq)
                r2 = (-bq - mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq)

                if r1 > 0:
                    psi = r1
                elif r2 > 0:
                    psi = r2

                dir= np.matmul(S, a + psi * (b - a))

            step = np.copy(dir)

            param_violations_lower = ((self.current_estimate + dir) <= self.lower_constraint).squeeze()
            param_violations_upper = ((self.current_estimate + dir) >= self.upper_constraint).squeeze()

            param_violations = np.logical_or(param_violations_lower, param_violations_upper).squeeze()

            param_not_to_freeze = [i for i, x in enumerate(~(param_violations)) if x]
            param_to_freeze = [i for i, x in enumerate((param_violations)) if x]

            current_cond = np.logical_or(any((self.current_estimate) < self.lower_constraint), any(
                (self.current_estimate) > self.upper_constraint))
            next_cond = np.logical_or(any((self.current_estimate + dir) < self.lower_constraint), any(
                (self.current_estimate + dir) > self.upper_constraint))

            self.flag_bisection = np.logical_and(~current_cond,
                                                 next_cond).squeeze()  # transition from permissible to impermissible only
            freeze = np.logical_and(any(param_violations), ~self.flag_bisection).squeeze()

            if self.flag_bisection:
                ########################################
                ###### BISECTION
                ########################################
                condition = any((self.current_estimate + dir) < self.lower_constraint) or any(
                    (self.current_estimate + dir) > self.upper_constraint)
                if condition:
                    low = 0
                    upper = self.gamma
                    for i in range(1, 1000):
                        mid = (low + upper) / 2

                        if np.linalg.norm(GN_dir) < np.linalg.norm(mid):

                            dir = np.matmul(S, GN_dir)

                        elif np.linalg.norm(SD_dir) > np.linalg.norm(mid):

                            dir = np.matmul(S, SD_dir / np.linalg.norm(SD_dir)) * mid

                        else:
                            a = ideal_step * SD_dir
                            b = GN_dir
                            c = np.matmul(np.transpose(a), (b - a))

                            aq = np.linalg.norm(b - a) ** 2
                            bq = 2 * c
                            cq = np.linalg.norm(a) ** 2 - mid ** 2

                            r1 = (-bq + mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq)
                            r2 = (-bq - mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq)

                            if r1 > 0:
                                psi = r1
                            elif r2 > 0:
                                psi = r2

                            dir = np.matmul(S, a + psi * (b - a))

                        step = np.copy(dir)

                        condition = any((self.current_estimate + dir) < self.lower_constraint) or any(
                            (self.current_estimate + dir) > self.upper_constraint)

                        if condition:
                            upper = np.copy(mid)
                        else:
                            low = np.copy(mid)

                    self.gamma = mid

                    for i in range(0, len(param_to_freeze)):
                        if param_violations_lower[param_to_freeze[i]]:
                            dir[param_to_freeze[i], 0] = self.lower_constraint - self.current_estimate[
                                param_to_freeze[i], 0]
                        else:
                            dir[param_to_freeze[i], 0] = self.upper_constraint - self.current_estimate[
                                param_to_freeze[i], 0]

                    step = np.copy(dir)


            elif freeze:

                hess = np.delete(hess, param_to_freeze, axis=0)
                hess = np.delete(hess, param_to_freeze, axis=1)

                grad = np.delete(grad, param_to_freeze, axis=0)

                if self.damp_type == 'Jacobian':
                    DM_square = np.diag(np.diag(hess) / np.max(np.diag(hess)))
                elif self.damp_type == 'Identity':
                    DM_square = np.diag(np.ones((len(hess[0, :]), 1)))

                DM = DM_square ** 0.5
                S = np.linalg.inv(DM)

                #### Compute Hessian and gradient in a scaled space
                hessian_scaled = np.matmul(np.transpose(S), np.matmul(hess, S))
                grad_scaled = np.matmul(np.transpose(S), grad)
                hessinv = np.linalg.inv(hessian_scaled)

                #### Compute Hessian and gradient in a scaled space

                ideal_step = (np.matmul(np.transpose(grad_scaled), grad_scaled)) / (
                    np.matmul(np.transpose(grad_scaled), np.matmul(hessian_scaled, grad_scaled)))

                GN_dir = -np.matmul(hessinv, grad_scaled)
                SD_dir = -(grad_scaled) / np.linalg.norm(grad_scaled)

                #### directions in orininal space

                if np.linalg.norm(GN_dir) < np.linalg.norm(self.gamma):

                    dir = np.matmul(S, GN_dir)

                elif np.linalg.norm(SD_dir) > np.linalg.norm(self.gamma):

                    dir = np.matmul(S, SD_dir / np.linalg.norm(SD_dir)) * self.gamma

                else:
                    a = ideal_step * SD_dir
                    b = GN_dir
                    c = np.matmul(np.transpose(a), (b - a))

                    aq = np.linalg.norm(b - a) ** 2
                    bq = 2 * c
                    cq = np.linalg.norm(a) ** 2 - self.gamma ** 2

                    r1 = (-bq + mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq)
                    r2 = (-bq - mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq)

                    if r1 > 0:
                        psi = r1
                    elif r2 > 0:
                        psi = r2

                    dir = np.matmul(S, a + psi * (b - a))

                step = np.zeros((np.shape(self.grad)))

                for i in range(0, len(param_not_to_freeze)):
                    step[param_not_to_freeze[i], 0] = dir[i]


            if self.verbose:
                print('     Constraint Region = ', self.gamma)

            # Compute gain ratio

            m_0 = np.copy(self.f0)  # anyway to avoid having this?
            m_p = m_0 + np.matmul(np.transpose(dir), self.grad) + 0.5 * np.matmul(np.transpose(dir),
                                                                                  np.matmul(self.hess, dir))

            f = self.forward_model(self.current_estimate + dir)

            print('f = ', f, ' f0 = ', self.f0)
            self.delta_f = np.asscalar(self.f0) - np.asscalar(f)
            self.delta_m = np.asscalar(m_0) - np.asscalar(m_p) # want this to be >0

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
                        self.gamma = self.gamma  / 2
                    elif self.parameter_update == 'Nielson':
                        mult = 2
                        self.gamma  = self.gamma * np.max(np.array([1 / 3, 1 - (2 * self.rho - 1) ** 3]))



                elif (self.rho > 0.99999 and self.rho < 1) or (self.rho > 1 and self.rho < 1.00001):
                    if self.parameter_update == 'Marquardt':
                        self.gamma  = self.gamma  * 3
                    elif self.parameter_update == 'Nielson':
                        mult = 2 * mult
                        self.gamma  = self.gamma  * mult

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


