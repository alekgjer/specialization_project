import numpy as np
import os
import time

from scipy.linalg import norm
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Various utility functions


def compute_scattering_angle(velocity_before, velocity_after):
    """
        Function to compute the scattering angles for a small particle bouncing of a bigger and massive one
    :param velocity_before: velocity vector of the small particle before collision
    :param velocity_after: velocity vector of the small particle after collision
    :return: angle between vectors
    """
    # use definition of dot product
    angle = np.arccos(np.dot(velocity_after, velocity_before)/(norm(velocity_before)*norm(velocity_after)))
    return angle


def random_positions_and_radius(number_particles, min_value=1e-04, y_max=1):
    """
        Function to create random positions in a desired area of a square box with lines at x = 0, x=1, y=0, y=1
    :param number_particles: number of particles in the box
    :param min_value: minimum position away from an axis
    :param y_max: max position away from y-axis
    :return: random positions decided by input parameters and min possible radius for the particles to not overlap each
    other or a wall
    """
    # random positions making sure that the positions do not get closer than min_value to the walls
    x_pos = np.random.uniform(low=min_value, high=(1-min_value), size=number_particles)
    y_pos = np.random.uniform(low=min_value, high=(y_max-min_value), size=number_particles)
    positions = np.zeros((number_particles, 2))  # save positions as a (N, 2) array.
    positions[:, 0] = x_pos
    positions[:, 1] = y_pos

    min_distance_from_zero_x = np.min(positions[:, 0])  # closest particle distance to y_axis
    min_distance_from_zero_y = np.min(positions[:, 1])  # closest particle distance to x_axis
    min_distance_from_x_max = np.min(1 - positions[:, 0])  # closest particle distance to x=1
    min_distance_from_y_max = np.min(y_max - positions[:, 1])  # closest particle distance to y_max

    min_distance = min(min_distance_from_x_max, min_distance_from_y_max,
                       min_distance_from_zero_x, min_distance_from_zero_y)  # closest particle distance to any wall

    for i in range(np.shape(positions)[0]):  # loop through all positions
        random_position = positions[i, :]  # pick out a random position
        diff = positions - np.tile(random_position, reps=(len(positions), 1))  # find distance vectors to other pos
        diff = norm(diff, axis=1)  # find distance from all distance vectors
        number_of_zeros = np.sum(diff == 0)  # compute number of zero distance. Should only be one(itself!)

        if number_of_zeros > 1:  # Can happen(maybe?)
            print('Two particles randomly generated at same point')
            exit()
        diff = diff[diff != 0]  # remove the distance a particle has with itself
        min_distance = min(min_distance, np.min(diff))  # find the closest distance between wall and nearest particle
    print(min_distance)
    min_radius = min_distance/2  # divide by two since the distance should equal at least radius*2
    min_radius *= 0.99  # in order to have no particles connected with either wall or another particle
    return positions, min_radius


def random_positions_for_given_radius(number_of_particles, radius, y_max=1.0):
    """
        Function to create random positions for number_of_particles with a given radius.
    :param number_of_particles: int giving the amount of particles to create positions for
    :param radius: radius of the particles
    :param y_max: max limit on the y-axis. Can be used to create wall on bottom half or particles everywhere.
    :return: uniformly distributed positions as a 2D array with shape (number_of_particles, 2)
    """
    positions = np.zeros((number_of_particles, 2))  # output shape
    # get random positions in the specified region. Need to make the positions not overlapping with the walls
    # do create more points than the output since not all are accepted since they are too close to each other
    x_pos = np.random.uniform(low=radius * 1.001, high=(1 - radius), size=number_of_particles ** 2)
    y_pos = np.random.uniform(low=radius * 1.001, high=(y_max - radius), size=number_of_particles ** 2)
    counter = 0  # help variable to accept positions that are not too close
    for i in range(len(x_pos)):
        if counter == len(positions):  # Check if there is enough accepted positions
            print('Done')
            break

        random_position = np.array([x_pos[i], y_pos[i]])  # pick a random position
        # create a distance vector to all accepted positions
        diff = positions - np.tile(random_position, reps=(len(positions), 1))
        diff = norm(diff, axis=1)  # compute distance as the norm of each distance vector
        boolean = diff <= (2 * radius)  # boolean array to indicate if new position is closer than 2*radius to another
        # check of boolean array. If the sum is higher than zero the random position is closer than 2R and is rejected
        if np.sum(boolean) > 0:
            continue
        else:
            # add position to output array
            positions[counter, :] = random_position
            counter += 1
    # remove all slots that did not get a random position
    positions = positions[positions[:, 0] != 0]
    number_of_positions = len(positions)  # number of accepted points -> number of accepted particles
    if y_max == 1:
        # save file for problem where one use the whole region
        np.save(file=os.path.join(init_folder, f'uniform_pos_N_{number_of_positions}_rad_{radius}'), arr=positions)
    else:
        # save file for problem 5 where these positions in the wall getting hit by a projectile
        np.save(file=os.path.join(init_folder, f'wall_pos_N_{number_of_positions}_rad_{radius}'), arr=positions)


def random_uniformly_distributed_velocities(N, v0):
    """
        Function that creates a random set of velocity vectors for N particles with speed v0. First one create random
        angles uniformly distributed between (0, 2*pi). Then one use that the velocity in the x direction is equal
        to the cosine of the random angles multiplied by the speed. Same for velocity in y direction, but by using sine.
    :param N: number of particles
    :param v0: initial speed of all particles
    :return: uniformly distributed velocities as a 2D array with shape (number_of_particles, 2)
    """
    random_angles = np.random.uniform(low=0, high=2 * np.pi, size=N)  # random angles in range(0, 2pi)
    velocities = np.zeros((N, 2))
    velocities[:, 0], velocities[:, 1] = np.cos(random_angles), np.sin(random_angles)  # take cosine and sine
    velocities *= v0  # multiply with speed
    return velocities


def scattering_angle_func_impact_parameter():
    """
        Function to compute scattering angle as a function of the impact parameter b. Solves problem 1 from exam.
        Involves two particles, one big and massive, and see how the velocity changes for the smaller particle
        by colliding with the bigger particle.
    """
    N = 2
    xi = 1
    initial_position_big_particle = [0.5, 0.5]
    initial_velocity_small_particle = [0.2, 0]
    initial_velocity_big_particle = [0, 0]
    initial_positions = np.zeros((N, 2))
    initial_positions[1, :] = initial_position_big_particle
    initial_velocities = np.zeros_like(initial_positions)
    initial_velocities[0, :] = initial_velocity_small_particle
    initial_velocities[1, :] = initial_velocity_big_particle
    # information given in task
    mass_array = np.array([1, 10 ** 6])
    radius_array = np.array([0.001, 0.1])

    impact_parameters = np.linspace(-radius_array[1]*1.5, radius_array[1]*1.5, 1000)
    scattering_angles = np.zeros_like(impact_parameters)

    for counter, impact_parameter in enumerate(impact_parameters):
        # update initial position of small particle
        initial_position_small_particle = [0.1, 0.5+impact_parameter]
        initial_positions[0, :] = initial_position_small_particle

        box_of_particles = ParticleBox(number_of_particles=N,
                                       restitution_coefficient=xi,
                                       initial_positions=initial_positions,
                                       initial_velocities=initial_velocities,
                                       masses=mass_array,
                                       radii=radius_array)

        velocity_small_particle_after_collision = box_of_particles.simulation(problem_number=1)
        scattering_angles[counter] = compute_scattering_angle(np.array(initial_velocity_small_particle),
                                                              velocity_small_particle_after_collision)
    # save results in a matrix where each row gives impact parameter and scattering angle
    matrix_to_file = np.zeros((len(impact_parameters), 2))
    matrix_to_file[:, 0] = impact_parameters/np.sum(radius_array)
    matrix_to_file[:, 1] = scattering_angles
    # save to file
    np.save(file=os.path.join(results_folder, 'scattering_angle'), arr=matrix_to_file)


def speed_distribution(use_equal_particles=True, number_of_runs=1):
    """
        Function to get speed of every particle after the system has reached equilibrium. Mainly solves problem 2 and
        3 in the exam. The procedure is based on initializing a system of many particles with the same initial speed.
        Then you start the simulation and let them collide until equilibrium have been reached. One then return the
        speed of each particle in order to create speed distribution plots.
    :param use_equal_particles: bool value that separates problem 2 and 3. If not equal: Second halv will have m=4*m0
    :param number_of_runs: In order to achieve better statistics, take averages over multiple runs.
    """
    N = 1000  # number of particles
    xi = 1  # restitution coefficient
    v_0 = 0.2  # initial speed
    radius = 0.01  # radius of all particles
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}.npy'))
    radii = np.ones(N) * radius  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass
    energy_matrix = np.zeros((N, 2))  # array with mass and speed to save to file

    if not use_equal_particles:
        # Problem 3: Second half of particles will have an mass which is bigger than the first half by a factor 4.
        mass[int(N / 2):] *= 4

    energy_matrix[:, 0] = mass  # mass does not change through the simulation

    for run_number in range(number_of_runs):
        # use same initial positions but new initial velocities every run
        # update velocities by getting new random angles, taking cos and sin and multiply with the initial speed
        velocities = random_uniformly_distributed_velocities(N, v_0)

        if run_number == 0:
            # save initial speed as a reference point
            initial_speeds = np.sqrt(np.sum(velocities * velocities, axis=1))
            energy_matrix[:, 1] = initial_speeds
            if use_equal_particles:
                np.save(file=os.path.join(results_folder, f'problem_2_N_{N}_init_energy_matrix'), arr=energy_matrix)
            else:
                np.save(file=os.path.join(results_folder, f'problem_3_N_{N}_init_energy_matrix'), arr=energy_matrix)
        # initialize system
        box_of_particles = ParticleBox(number_of_particles=N,
                                       restitution_coefficient=xi,
                                       initial_positions=positions,
                                       initial_velocities=velocities,
                                       masses=mass,
                                       radii=radii)

        if use_equal_particles:
            # Problem 2
            speeds_after_equilibrium = box_of_particles.simulation(problem_number=2, output_timestep=0.1)
            energy_matrix[:, 1] = speeds_after_equilibrium
            np.save(file=os.path.join(results_folder, f'problem_2_N_{N}_eq_energy_matrix_{run_number}'),
                    arr=energy_matrix)
        else:
            # Problem 3
            speeds_after_equilibrium = box_of_particles.simulation(problem_number=3, output_timestep=0.1)
            energy_matrix[:, 1] = speeds_after_equilibrium
            np.save(file=os.path.join(results_folder, f'problem_3_N_{N}_eq_energy_matrix_{run_number}'),
                    arr=energy_matrix)


def compute_avg_energy_development_after_time():
    """
        Function to mainly solve problem 4 in the exam by saving how the energy develop at a short output step. Will
        save the average kinetic energy of all particles(together and separate for m0 and m particles) at all outputs
        and save to file.
    """
    N = 1000  # number of particles
    v_0 = 0.2  # initial speed of all particles
    radius = 0.01  # radius of all particles
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}.npy'))
    velocities = random_uniformly_distributed_velocities(N, v_0)
    radii = np.ones(N) * radius  # all particles have the same radius
    # first half will be m0 particles and second half is m particles with m=4*m0.
    mass = np.ones(N)
    mass[int(N / 2):] *= 4

    xi_list = [1, 0.9, 0.8]  # will do for multiple values of the restitution coefficient

    for xi in xi_list:
        # initialize a ParticleBox with given parameters
        box_of_particles = ParticleBox(number_of_particles=N,
                                       restitution_coefficient=xi,
                                       initial_positions=positions,
                                       initial_velocities=velocities,
                                       masses=mass,
                                       radii=radii)
        # simulate and solve problem 4
        avg_energy, avg_energy_m0, avg_energy_m, time_array = box_of_particles.simulation(problem_number=4,
                                                                                          output_timestep=0.01)
        # store results in a matrix with given form: time, avg_energg, avg_energy_m0 and avg_energy_m
        matrix_to_file = np.zeros((len(avg_energy), 4))
        matrix_to_file[:, 0] = time_array
        matrix_to_file[:, 1] = avg_energy
        matrix_to_file[:, 2] = avg_energy_m0
        matrix_to_file[:, 3] = avg_energy_m
        # save to file
        np.save(file=os.path.join(results_folder, f'problem_4_N_{N}_energy_development_xi_{xi}'), arr=matrix_to_file)


def get_crater_size(initial_positions_wall, positions_wall_after_simulation, radius):
    """
        Function to compute the size of the crater resulting from a projectile hitting a wall of smaller particles
    :param initial_positions_wall: positions of the particles in the wall before being hit. Is an 2D array with shape
    (number_particles_wall, 2) where first column is the x-positions and the second is the y-positions of particles.
    :param positions_wall_after_simulation: positions of the particles in the wall after being hit. Same shape as above.
    :param radius: radius of the particles in the wall
    :return: size of the crater as a measured quantity of how many particles have changed position
    """
    dx = positions_wall_after_simulation - initial_positions_wall
    dx = norm(dx, axis=1)  # calculate distance as the norm of the distance vector in position
    boolean = dx >= (2*radius)  # if a particle has moved a diameter away is it assumed to be affected
    crater_size = np.sum(boolean)/len(initial_positions_wall)  # number of affected particles / number of particles
    return crater_size


def simulate_and_get_crater_size(number_of_particles, restitution_coefficient, initial_positions, initial_velocities,
                                 masses, radii):
    """
        Help function to initial a ParticleBox and simulate a crater formation by solving problem 5. After simulation
        the function computes the size of the newly made crater.
    :param number_of_particles: number of particles in the box
    :param restitution_coefficient: restitution coefficient of the system. Tells how much energy is kept after colliding
    :param initial_positions: array with shape (numb_particles, 2) to indicate starting point for particles
    :param initial_velocities: array with shape (numb_particles, 2) to indicate starting velocity for particles
    :param masses: array with shape (numb_particles) to indicate the mass of each particle
    :param radii: array with shape (numb_particles) to indicate the radius of each particle
    :return: crater_size
    """
    # initialize system
    box_of_particles = ParticleBox(number_of_particles=number_of_particles,
                                   restitution_coefficient=restitution_coefficient,
                                   initial_positions=initial_positions,
                                   initial_velocities=initial_velocities,
                                   masses=masses,
                                   radii=radii)
    # simulate
    box_of_particles.simulation(problem_number=5, output_timestep=0.01)
    # compute crater size
    crater_size = get_crater_size(initial_positions[1:, :], box_of_particles.positions[1:, :], radii[-1])
    return crater_size


def study_crater_formation(study_parameter, number_of_parameters=10):
    """
        Function to study the formation of a crater by hitting a wall of particles with a projectile. The study is
        performed by doing a parameter sweep over a study_parameter a number of times and then write to file.
    :param study_parameter: string to indicate wh
    :param number_of_parameters:
    """
    N = 5001
    radius = 0.004
    positions_wall = np.load(os.path.join(init_folder, f'wall_pos_N_{N - 1}_rad_{radius}.npy'))
    positions = np.zeros((N, 2))
    positions[0, :] = np.array([0.5, 0.75])  # projectile position
    positions[1:, :] = positions_wall
    velocities = np.zeros((N, 2))
    radii = np.ones(N) * radius
    mass = np.ones(N)

    parameter_study_matrix = np.zeros((number_of_parameters, 2))
    print(f'Parameter study: {study_parameter}')
    if study_parameter == 'test':
        # starting set of parameters
        xi = 0.75
        v0 = 5
        velocities[0, :] = np.array([0, -v0])  # projectile velocity
        radii[0] *= 5  # projectile radius is 5 times bigger than the radius of the other particles
        mass[0] *= 25  # projectile mass is 25 bigger than the mass of the other particles
        crater_size = simulate_and_get_crater_size(N, xi, positions, velocities, mass, radii)
        print(f"Crater size: {crater_size}")

    elif study_parameter == 'mass':
        xi = 0.75
        v0 = 5
        velocities[0, :] = np.array([0, -v0])  # projectile velocity
        radii[0] *= 5  # projectile radius is 5 times bigger than the radius of the other particles
        mass_multiplication_array = np.linspace(3, 30, number_of_parameters)
        for counter, mass_multiplication_factor in enumerate(mass_multiplication_array):
            mass[0] *= mass_multiplication_factor
            crater_size = simulate_and_get_crater_size(N, xi, positions, velocities, mass, radii)
            parameter_study_matrix[counter, :] = [mass_multiplication_factor, crater_size]
            mass[0] /= mass_multiplication_factor
        np.save(file=os.path.join(results_folder, f'problem_5_N_{N}_mass_study'), arr=parameter_study_matrix)
    elif study_parameter == 'radius':
        xi = 0.75
        v0 = 5
        velocities[0, :] = np.array([0, -v0])  # projectile velocity
        mass[0] *= 25  # projectile mass is 25 bigger than the mass of the other particles
        radius_parameter_array = np.linspace(1, 10, number_of_parameters)
        for counter, radius_mult_factor in enumerate(radius_parameter_array):
            radii[0] *= radius_mult_factor

            crater_size = simulate_and_get_crater_size(N, xi, positions, velocities, mass, radii)

            parameter_study_matrix[counter, :] = [radius_mult_factor, crater_size]
            radii[0] /= radius_mult_factor
        np.save(file=os.path.join(results_folder, f'problem_5_N_{N}_radius_study'), arr=parameter_study_matrix)
    elif study_parameter == 'speed':
        xi = 0.75
        radii[0] *= 5  # projectile radius is 5 times bigger than the radius of the other particles
        mass[0] *= 25  # projectile mass is 25 bigger than the mass of the other particles
        v0_array = np.linspace(2, 8, number_of_parameters)
        for counter, v0 in enumerate(v0_array):
            velocities[0, :] = np.array([0, -v0])  # projectile velocity

            crater_size = simulate_and_get_crater_size(N, xi, positions, velocities, mass, radii)

            parameter_study_matrix[counter, :] = [v0, crater_size]
        np.save(file=os.path.join(results_folder, f'problem_5_N_{N}_speed_study'), arr=parameter_study_matrix)
    elif study_parameter == 'xi':
        v0 = 5
        velocities[0, :] = np.array([0, -v0])  # projectile velocity
        radii[0] *= 5  # projectile radius is 5 times bigger than the radius of the other particles
        mass[0] *= 25  # projectile mass is 25 bigger than the mass of the other particles
        xi_array = np.linspace(0.6, 0.8, number_of_parameters)
        for counter, xi in enumerate(xi_array):
            crater_size = simulate_and_get_crater_size(N, xi, positions, velocities, mass, radii)

            parameter_study_matrix[counter, :] = [xi, crater_size]
        np.save(file=os.path.join(results_folder, f'problem_5_N_{N}_xi_study'), arr=parameter_study_matrix)
    else:
        print("Study parameter not understood. Try again!")


def compute_mean_free_path():
    # TODO: seems like the mean free path ends up being half of what it should be in 2D
    N = 2000  # number of particles
    check_particle_index = 42
    v_0 = 0.2  # initial speed of all particles
    xi = 1
    radius = 0.007  # radius of all particles
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}.npy'))
    # velocities = np.load(os.path.join(init_folder, f'eq_vel_N_{N}_rad_{radius}.npy'))
    velocities = random_uniformly_distributed_velocities(N, v_0)
    radii = np.ones(N) * radius  # all particles have the same radius
    mass = np.ones(N)  # all particles have the same mass
    # initialize a ParticleBox with given parameters
    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii)
    # simulate
    computed_mean_free_path = box_of_particles.simulation(problem_number=7, output_timestep=0.1)
    mask = norm(positions - np.tile([0.5, 0.5], reps=(len(positions), 1)), axis=1) < 0.1
    print(f'Free path check: {np.mean(computed_mean_free_path[mask])}')
    print(f'Free path mean all: {np.mean(computed_mean_free_path)}')
    return None


def compute_distance_of_particles_close_to_center():
    # TODO: something wrong with the diffusion properties..
    """
        Functionality to compute the distance of particles starting close to the center of the box as a function of time
        in order to study diffusion properties. Function will save the results to file.
    """
    N = 2000  # number of particles
    v_0 = 0.2  # initial speed of all particles
    xi = 1
    radius = 0.007  # radius of all particles
    # positions = np.load(os.path.join(init_folder, f'eq_pos_N_{N}_rad_{radius}.npy'))
    # velocities = np.load(os.path.join(init_folder, f'eq_vel_N_{N}_rad_{radius}.npy'))
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}.npy'))
    radii = np.ones(N) * radius  # all particles have the same radius
    mass = np.ones(N)  # all particles have the same mass
    info_matrix = np.zeros((2001, 3))
    number_of_runs = 10
    for i in range(number_of_runs):
        print(f'Run number: {i+1}')
        velocities = random_uniformly_distributed_velocities(N, v_0)

        # initialize a ParticleBox with given parameters
        box_of_particles = ParticleBox(number_of_particles=N,
                                       restitution_coefficient=xi,
                                       initial_positions=positions,
                                       initial_velocities=velocities,
                                       masses=mass,
                                       radii=radii)
        mean_q_dist_array, mean_speed_array, time_array = box_of_particles.simulation(problem_number=6,
                                                                                      output_timestep=0.01)
        info_matrix[:, 0] += mean_q_dist_array
        info_matrix[:, 1] += mean_speed_array
        info_matrix[:, 2] += time_array

    info_matrix /= number_of_runs
    # # simulate
    # mean_q_dist, times = box_of_particles.simulation(problem_number=6, output_timestep=0.01)
    # # save to file
    # matrix_to_file = np.zeros((len(mean_q_dist), 2))
    # matrix_to_file[:, 0] = times
    # matrix_to_file[:, 1] = mean_q_dist
    # np.save(file=os.path.join(results_folder, f'problem_6_N_{N}_mean_q_dist'), arr=matrix_to_file)
    # mean_q_distance_x_array, mean_q_distance_y_array, mean_speed_array, time_array = box_of_particles.simulation(problem_number=6, output_timestep=0.01)
    # mean_q_dist_array, mean_speed_array, time_array = box_of_particles.simulation(problem_number=6, output_timestep=0.01)
    np.save(file=os.path.join(results_folder, f'problem_6_N_{N}_mean_q_dist_long_time'), arr=info_matrix)
    # mean_q_dist_array = info_matrix[:, 0]
    # mean_speed_array = info_matrix[:, 1]
    # time_array = info_matrix[:, 2]
    # times = np.linspace(0, time_array[-1], 2)
    # mean_free_path = 6.31*1e-03
    # avg_speed = 0.177
    # slope, intercept, r_value, p_value, std_err = linregress(time_array, mean_q_dist_array)
    # fig, axes = plt.subplots(nrows=2, sharex=True)
    # axes[0].plot(time_array, mean_speed_array)
    # axes[0].plot([time_array[0], time_array[-1]], [avg_speed, avg_speed], 'k--', label='Theoretical average speed')
    # axes[0].set_ylabel('Average speed')
    # axes[0].legend()
    # axes[1].plot(time_array, mean_q_dist_array, 'b', label='Numerical values')
    # #axes[1].plot(times, times*4*mean_free_path*avg_speed/3, 'k--', label='4Kt')
    # D = 7.888*1e-04
    # axes[1].plot(times, times*D*4, 'k--', label='4Dt')
    # axes[1].plot(times, intercept+slope*times, 'r--', label='Linear regression')
    # # axes[1].plot(times, 4*avg_speed*mean_free_path*times/3, 'g--', label='mfp 4Kt')
    #
    # axes[1].set_ylabel('Variance')
    # axes[1].set_xlabel('time')
    # axes[1].legend()
    # plt.show()


def create_fractal_from_sticky_particles():
    """
        Function that creates fractal properties by allowing particles to stick together if a particle collides with
        a particle at rest. Initially the particle closest to the center is at rest and a fractal builds in time
    :return:
    """
    N = 2000  # number of particles
    v_0 = 0.2  # initial speed of all particles
    xi = 1
    radius = 0.007  # radius of all particles
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}.npy'))
    distance_to_center = norm(positions - np.tile([0.5, 0.5], reps=(len(positions), 1)), axis=1)
    closest_particle = np.argmin(distance_to_center)
    velocities = random_uniformly_distributed_velocities(N, v_0)
    velocities[closest_particle, :] = [0, 0]  # particle closest to center is at rest
    radii = np.ones(N) * radius  # all particles have the same radius
    mass = np.ones(N)  # all particles have the same mass initially
    mass[closest_particle] = 10 ** 10  # increase mass of particle at rest in order to identify them in a collision
    # initialize a ParticleBox with given parameters
    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii)
    # simulate
    positions = box_of_particles.simulation(problem_number=8, output_timestep=0.05)
    # save the system at rest with all particle positions to file
    np.save(file=os.path.join(results_folder, f'problem_8_N_{N}_rad_{radius}_fractal'), arr=positions)


if __name__ == "__main__":
    from particle_box import ParticleBox
    from config import results_folder, init_folder
    start_time = time.time()
    # choose what problem one want to solve by simulating particle collisions in 2D
    problem = {1: 'Scattering angle',
               2: 'Speed distribution',
               3: 'Energy development',
               4: 'Initial positions',
               5: 'Crater formation',
               6: 'Mean quadratic distance',
               7: 'Mean free path',
               8: 'Fractal'}[6]
    print(f"Problem: {problem}")
    if problem == 'Scattering angle':
        # scattering angle from one small particles bouncing of one large particle at rest
        scattering_angle_func_impact_parameter()
    elif problem == 'Speed distribution':
        # let system evolve in time until enough collisions has occurred to assume equilibrium has been reached.
        use_same_particle = False  # choice of whether to let all particles have equal mass, or half have 4 times larger
        speed_distribution(use_equal_particles=use_same_particle, number_of_runs=30)
    elif problem == 'Energy development':
        # let the system evolve in time until and see how the energy changes for different restitution coefficients
        compute_avg_energy_development_after_time()
    elif problem == 'Initial positions':
        # create files to be used as initial positions
        N = 2000
        radius_particle = 1 / np.sqrt(4*N*np.pi)
        # radius of particle is chosen in order to give ca. 1/2 packing fraction for wall in crater formation
        radius_particle = np.round(radius_particle+0.0005, decimals=3)
        # save random positions to initial_positions folder
        random_positions_for_given_radius(N, radius_particle, y_max=1.0)
    elif problem == 'Crater formation':
        # study crater formation by hitting a wall of densely packed particles with a projectile
        study_parameters = {0: 'test', 1: 'mass', 2: 'radius', 3: 'xi'}
        choice_of_parameter = study_parameters[0]
        study_crater_formation(study_parameter=choice_of_parameter)
    elif problem == 'Mean quadratic distance':
        # study diffusion properties by looking at the mean quadratic displacement from initial position
        compute_distance_of_particles_close_to_center()
    elif problem == 'Mean free path':
        # study diffusion properties by looking at the mean free path by storing distance travelled between collisions
        compute_mean_free_path()
    elif problem == 'Fractal':
        # create a fractal by letting a particle hitting a particle as rest simulated as sticky particles getting stuck
        create_fractal_from_sticky_particles()

    print(f"Time used: {time.time() - start_time}")
