import numpy as np
import math as mth


class Levenberg_Marquart():

    def __init__(self, initial_prediction, compute_hessian, compute_gradient,  forward_model, d_param=1e-20,lower_constraint = 0, upper_constraint = 1.4e5, num_iterations=5):

        self.initial_prediction = np.copy(initial_prediction.reshape((len(initial_prediction),1)))
        self.current_estimate = np.copy(initial_prediction.reshape((len(initial_prediction),1)))
        self.num_iterations = num_iterations

        self.gamma = d_param

        self.compute_hessian = compute_hessian
        self.compute_gradient = compute_gradient
        self.forward_model = forward_model

        self.lower_constraint = lower_constraint
        self.upper_constraint = upper_constraint
        self.flag_bisection = True

        self.verbose = True

    def residual(self, input_data):
        predicted_measurement = self.forward_model(input_data)
        residual = predicted_measurement - self.measurement
        return residual


    def direction(self):

        flag_bisection = True
        if self.verbose:
            print('Finding the damping parameter')
        L = 1
        while 1:


            DM_square = np.diag(np.diag(self.hess)/np.max(np.diag(self.hess)))


            damped_hessian = self.hess+self.gamma*DM_square

            hessinv = np.linalg.inv(damped_hessian)

            dir = -np.matmul(hessinv, self.grad)

            if self.flag_bisection:
                ########################################
                ###### BISECTION
                ########################################
                condition = any((self.current_estimate + dir)<self.lower_constraint) or any((self.current_estimate+ dir) > self.upper_constraint)
                if condition:
                    low = mth.log10(self.gamma)
                    upper = mth.log10(self.gamma) + 5
                    for i in range(1,1000):
                        mid = (low + upper) / 2
                        damped_hessian = self.hess + (10**mid)* DM_square
                        hessinv = np.linalg.inv(damped_hessian)
                        dir = -np.matmul(hessinv, self.grad)
                        condition = any((self.current_estimate + dir) < self.lower_constraint) or any(
                            (self.current_estimate + dir) > self.upper_constraint)
                        if condition:
                            low =  np.copy(mid)
                        else:
                            upper = np.copy(mid)

                    self.gamma=10 ** mid

            damped_hessian = self.hess + self.gamma * DM_square
            hessinv = np.linalg.inv(damped_hessian)
            dir = -np.matmul(hessinv, self.grad)

            if self.verbose:
                 print('     Gamma = ', self.gamma)

            # Compute gain ratio

            m_0 = self.forward_model(self.current_estimate)
            m_p = m_0 + np.matmul(np.transpose(dir), self.grad)+0.5*np.matmul(np.transpose(dir), np.matmul(self.hess, dir))

            f_0 = m_0
            f = self.forward_model(self.current_estimate+dir)

            delta_f = f_0 - f
            delta_m = m_0 - m_p # want this to be >0

            rho = delta_f / delta_m


            if self.verbose:
                print('          Rho = ', np.squeeze(rho), ' | delta f = ', np.squeeze(delta_f), ' | delta m =', np.squeeze(delta_m))
                print('----------------------------------------------------------------------------------------')
                print('----------------------------------------------------------------------------------------')

            if L == 5: # timout condition
                return dir
            else:
                if rho<0.8 or rho > 1.2:
                    self.gamma = self.gamma*mth.pi
                elif (rho >0.99999 and rho < 1) or (rho > 1 and rho <1.00001):
                    self.gamma = self.gamma/2
                else:
                    return dir
            L = L + 1


    def optimisation_main(self):


         for i in range(0,self.num_iterations):

             self.gamma=0
             self.hess = self.compute_hessian(self.current_estimate)
             self.grad = self.compute_gradient(self.current_estimate)

             dir = self.direction()
             self.current_estimate = self.current_estimate+dir

         return self.current_estimate

#####
##### Example call
#####
def forward_model(x):
    y = np.array(x[0]**2+x[1]**2)
    y = y.reshape((1,1))
    return y
def compute_gradient(x):
    y = np.array(([2*x[0] ], [2*x[1]]))
    y = y.reshape((2, 1))
    return y
def compute_hessian(x):
    y = np.array(([2 , 0],[0, 2]))
    y = y.reshape((2, 2))
    return y

initial_prediction = np.array([5,2.7])

a=forward_model(initial_prediction)

b=compute_gradient(initial_prediction)

o_class = TRA(initial_prediction, compute_hessian, compute_gradient,  forward_model, lower_constraint = -mth.inf, upper_constraint = mth.inf, num_iterations=5)

minimum = o_class.optimisation_main()

print(minimum)

