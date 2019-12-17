import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import norm

# Some functionality implemented to solve the Langevin equation numerically, but the great difficulty in achieving
# reliable values for q and gamma has made the numerical solution difficult


def trajectory_ex(t_start=0, t_stop=1, dt=2**(-4), x0=np.array([1])):
    """
        Function to do the example from the section in Kloeden&Pladen about the Euler-Maruyama method
    :param t_start: starting time
    :param t_stop: stopping time
    :param dt: timestep value
    :param x0: initial starting condition
    """
    parameters = [1.5, 1]
    number_of_timesteps = int((t_stop - t_start) / dt)
    x = np.zeros((number_of_timesteps + 1, len(x0)))
    x_exact = np.zeros_like(x)
    x[0, :] = x0
    x_exact[0, :] = x0
    time = np.linspace(t_start, t_stop, number_of_timesteps + 1)

    random_force = np.random.normal(scale=np.sqrt(dt), size=(number_of_timesteps, 1))

    for step in range(number_of_timesteps):
        x[step + 1, :] = x[step, :] + parameters[0]*x[step, :]*dt + parameters[1] * x[step, :] * random_force[step]
        x_exact[step+1, :] = x0*np.exp((parameters[0]-parameters[1]**2/2)*time[step]+parameters[1]*np.sum(random_force[:step+1]))

    plt.plot(time, x, label='Numerical solution')
    plt.plot(time, x_exact, label='Exact solution')
    plt.legend()
    plt.show()


def dx_dv(x_n, t, dt, parameters, random_force):
    """
        Function that computes the change in position and velocity at a given position and velocity
    :param x_n: [xn, yn, vx_n, vy_n], position and velocity at a given time
    :param t: time
    :param dt: timestep value
    :param parameters: [m, gamma, q]
    :param random_force: random force in x and y direction as a array
    :return: the change in all quantities
    """
    # x = [xn, yn, vx_n, vy_n]
    # parameters = [m, gamma, q]
    d = np.zeros_like(x_n)
    dx = x_n[2:]
    # divide by dt for random force to use *dt in euler-step
    dv = - (parameters[1]/parameters[0]) * x_n[2:] + (parameters[2]/parameters[0]) * np.array([random_force[0], random_force[1]]) / dt
    d[:2] = dx
    d[2:] = dv
    return d


def euler_step(x_n, t, dt, parameters, random_force):
    """
        Function to take a Euler step as defined by the Euler-Maruyama method.
    :param x_n: [xn, yn, vx_n, vy_n], position and velocity at a given time
    :param t: time
    :param dt: timestep value
    :param parameters: [m, gamma, q]
    :param random_force: random force in x and y direction as a array
    :return: position and velocity after taking a incrementing time
    """
    return x_n+dx_dv(x_n, t, dt, parameters, random_force)*dt


def trajectory(t_start, t_stop, dt, x0, parameters):
    """
        Function to compute a trajectory based on taking euler-steps from t_start to t_stop
    :param t_start: starting time
    :param t_stop: stopping time
    :param dt: timestep value
    :param x0: starting position and velocity
    :param parameters: [mass, gamma, q]
    :return: position and velocity as a function of time, and time array
    """
    number_of_timesteps = int((t_stop-t_start)/dt)
    x = np.zeros((number_of_timesteps+1, len(x0)))
    x[0, :] = x0
    time = np.linspace(t_start, t_stop, number_of_timesteps+1)

    random_force = np.random.normal(scale=np.sqrt(dt), size=(number_of_timesteps, 2))

    for step in range(number_of_timesteps):
        x[step+1, :] = euler_step(x[step, :], time[step], dt, parameters, random_force[step])

    return x, time


def msd_function(t, v0, gamma, q):
    """
        Mean square displacement function given in Granular Gases
    :param t: time
    :param v0: start speed
    :param gamma: friction coefficient
    :param q: strength of random force
    :return: mean square displacement
    """
    return (v0**2-q/gamma)*(1-np.exp(-gamma*t))**2/gamma**2+2*q*t/gamma**2-2*q*(1-np.exp(-gamma*t))/gamma**3


def diffusion_properties(number_of_realizations=10):
    """
        Function that attempts to get the msd_function by solving the sde numericall by the Euler-Maruyama method
    :param number_of_realizations: take average over number_of_realizations runs
    """
    v0 = 0.2
    # v0_estimate = 0.108
    random_angle = np.random.uniform(low=0, high=2*np.pi, size=number_of_realizations)
    # random_angle = 0.5 * np.pi

    # gamma = 1
    # kT = 0.02
    #
    # parameters = [1, gamma, gamma*4*kT]

    parameters = [10, 1.98, 0.0079]
    # parameters = [1, 5, 1.5]

    timestep = 0.1
    t0 = 0
    tend = 5

    info_matrix = np.zeros((int((tend-t0)/timestep)+1, 3))

    # initial_condition = np.array([0.5, 0.5, 0, 0])

    for i in range(number_of_realizations):
        initial_condition = np.array([0.5, 0.5, v0 * np.cos(random_angle[i]), v0 * np.sin(random_angle[i])])
        trajectory_x, trajectory_t = trajectory(t0, tend, timestep, initial_condition, parameters)
        msd = norm(trajectory_x[:, :2] - np.tile(trajectory_x[0, :2], reps=(len(trajectory_x), 1)), axis=1) ** 2
        mss = norm(trajectory_x[:, 2:], axis=1) ** 2
        info_matrix[:, 0] += trajectory_t
        info_matrix[:, 1] += msd
        info_matrix[:, 2] += mss

    info_matrix /= number_of_realizations

    times = info_matrix[:, 0]

    plt.figure()
    plt.plot(times, info_matrix[:, 1], label='Numerical values')
    plt.plot(times, msd_function(times, v0, parameters[1], parameters[2]), label='Exact')
    # plt.plot(times, msd_function(times, v0_estimate, parameters_v0_est[1], parameters_v0_est[2]), label='v0_estimate')
    plt.legend()
    plt.figure()
    plt.plot(times, info_matrix[:, 2])
    plt.show()


if __name__ == "__main__":
    # trajectory_ex()
    diffusion_properties(number_of_realizations=100)
