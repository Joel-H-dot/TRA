import numpy as np
import math as mth



class powells_dogleg():

    def __init__(self, initial_prediction, f0, compute_hessian, compute_gradient, forward_model,
                 constraint_region=np.inf,lower_constraint = -np.inf, upper_constraint = np.inf, num_iterations=5):


        self.initial_prediction = np.copy(initial_prediction.reshape((len(initial_prediction), 1)))
        self.current_estimate = np.copy(initial_prediction.reshape((len(initial_prediction), 1)))

        self.num_iterations = num_iterations

        self.CR = constraint_region

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

            DM = DM_square**0.5
            S = np.linalg.inv(DM)

            #### Compute Hessian and gradient in a scaled space
            hessian_scaled = np.matmul(np.transpose(S),np.matmul(self.hess,S))
            grad_scaled = np.matmul(np.transpose(S),self.grad)
            hessinv = np.linalg.inv(hessian_scaled)

            #### Compute Hessian and gradient in a scaled space

            ideal_step = (np.matmul(np.transpose(grad_scaled), grad_scaled)) / (
                np.matmul(np.transpose(grad_scaled), np.matmul(hessian_scaled, grad_scaled)))

            GN_dir = -np.matmul(hessinv, grad_scaled)
            SD_dir = -(grad_scaled) / np.linalg.norm(grad_scaled)

            #### directions in orininal space

            GN_dir_original_space = np.matmul(S,GN_dir)
            SD_dir_original_space = np.matmul(S,ideal_step*SD_dir)

            if np.linalg.norm(GN_dir_original_space) < np.linalg.norm(self.CR):

                dir = GN_dir_original_space

            elif np.linalg.norm(SD_dir_original_space) > np.linalg.norm(self.CR):

                dir = (SD_dir_original_space/ np.linalg.norm(SD_dir_original_space)) * self.CR

            else:
                a = ideal_step * SD_dir;
                b = GN_dir;
                c = np.matmul(np.transpose(a),(b-a))

                aq = np.linalg.norm(b - a) ** 2;
                bq = 2 * c;
                cq = np.linalg.norm(a) ** 2 - self.CR ** 2;

                r1 = (-bq + mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq);
                r2 = (-bq - mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq);

                if r1 > 0:
                    psi = r1
                elif r2 > 0:
                    psi = r2

                dir = S *(a + psi * (b - a))



            if self.flag_bisection:
                ########################################
                ###### BISECTION
                ########################################
                condition = any((self.current_estimate + dir) < self.lower_constraint) or any(
                    (self.current_estimate + dir) > self.upper_constraint)
                if condition:
                    low = 0
                    upper = self.CR
                    for i in range(1, 1000):
                        mid = (low+upper) / 2

                        if np.linalg.norm(GN_dir_original_space) < np.linalg.norm(mid):

                            dir = GN_dir_original_space

                        elif np.linalg.norm(SD_dir_original_space) > np.linalg.norm(mid):

                            dir = (SD_dir_original_space / np.linalg.norm(SD_dir_original_space)) * mid

                        else:
                            a = ideal_step * SD_dir;
                            b = GN_dir;
                            c = np.matmul(np.transpose(a), (b - a))

                            aq = np.linalg.norm(b - a) ** 2;
                            bq = 2 * c;
                            cq = np.linalg.norm(a) ** 2 - mid ** 2;

                            r1 = (-bq + mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq);
                            r2 = (-bq - mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq);

                            if r1 > 0:
                                psi = r1
                            elif r2 > 0:
                                psi = r2

                            dir = S * (a + psi * (b - a))


                        condition = any((self.current_estimate + dir) < self.lower_constraint) or any(
                            (self.current_estimate + dir) > self.upper_constraint)
                        if condition:
                            upper = np.copy(mid)
                        else:
                            low = np.copy(mid)

                    self.CR = mid

            if np.linalg.norm(GN_dir_original_space) < np.linalg.norm(self.CR):

                dir = GN_dir_original_space

            elif np.linalg.norm(SD_dir_original_space) > np.linalg.norm(self.CR):

                dir = (SD_dir_original_space / np.linalg.norm(SD_dir_original_space)) * self.CR

            else:
                a = ideal_step * SD_dir;
                b = GN_dir;
                c = np.matmul(np.transpose(a), (b - a))

                aq = np.linalg.norm(b - a) ** 2;
                bq = 2 * c;
                cq = np.linalg.norm(a) ** 2 - self.CR ** 2;

                r1 = (-bq + mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq);
                r2 = (-bq - mth.sqrt(bq ** 2 - 4 * aq * cq)) / (2 * aq);

                if r1 > 0:
                    psi = r1
                elif r2 > 0:
                    psi = r2

                dir = S * (a + psi * (b - a))

            if self.verbose:
                print('     Constraint Region = ', self.CR)

            # Compute gain ratio

            m_0 = np.copy(self.f0)  # anyway to avoid having this?
            m_p = m_0 + np.matmul(np.transpose(dir), self.grad) + 0.5 * np.matmul(np.transpose(dir),
                                                                                  np.matmul(self.hess, dir))

            f = self.forward_model(self.current_estimate + dir)

            print('f = ', f, ' f0 = ', self.f0)
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


                    if self.parameter_update == 'levenberg':
                        self.CR  = self.CR / 2
                    elif self.parameter_update == 'nielson':
                        mult = 2
                        self.CR  = self.CR  * np.max([1 / 3, 1 - (2 * rho - 1) ** 3])

                elif (self.rho > 0.99999 and self.rho < 1) or (self.rho > 1 and self.rho < 1.00001):

                    if self.parameter_update == 'Marquardt':
                        self.CR = self.CR  * 3
                    elif self.parameter_update == 'Nielson':
                        mult = 2 * mult
                        self.CR  = self.CR  * mult

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


