import numpy as np
import os

from scipy.linalg import norm

from particle_box import ParticleBox
from simulation import Simulation
from config import results_folder, init_folder

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


def random_positions_for_given_radius(number_of_particles, radius, y_max=1.0, brownian_particle=False):
    """
        Function to create random positions for number_of_particles with a given radius.
    :param number_of_particles: int giving the amount of particles to create positions for
    :param radius: radius of the particles
    :param y_max: max limit on the y-axis. Can be used to create wall on bottom half or particles everywhere.
    :param brownian_particle: boolean value used to give if to create a bp in the middle with r=3r0
    :return: uniformly distributed positions as a 2D array with shape (number_of_particles, 2)
    """
    positions = np.zeros((number_of_particles, 2))  # output shape
    # get random positions in the specified region. Need to make the positions not overlapping with the walls
    # do create more points than the output since not all are accepted since they are too close to each other
    x_pos = np.random.uniform(low=radius * 1.001, high=(1 - radius), size=number_of_particles ** 2)
    y_pos = np.random.uniform(low=radius * 1.001, high=(y_max - radius), size=number_of_particles ** 2)
    counter = 0  # help variable to accept positions that are not too close

    if brownian_particle:
        positions[0, :] = [0.5, 0.5]
        counter += 1

    for i in range(len(x_pos)):
        if counter == len(positions):  # Check if there is enough accepted positions
            print('Done')
            break

        random_position = np.array([x_pos[i], y_pos[i]])  # pick a random position
        # create a distance vector to all accepted positions
        diff = positions - np.tile(random_position, reps=(len(positions), 1))
        diff = norm(diff, axis=1)  # compute distance as the norm of each distance vector
        if brownian_particle:
            diff[0] -= 2*radius
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
        if brownian_particle:
            np.save(
                file=os.path.join(init_folder,
                                  f'uniform_pos_around_bp_N_{number_of_positions}_rad_{radius}_bp_rad_{3*radius}'),
                arr=positions)
        else:
            np.save(file=os.path.join(init_folder, f'uniform_pos_N_{number_of_positions}_rad_{radius}'), arr=positions)
    else:
        # save file for problem 5 where these positions in the wall getting hit by a projectile
        np.save(file=os.path.join(init_folder, f'wall_pos_N_{number_of_positions}_rad_{radius}'), arr=positions)


def validate_positions(positions, radius):
    """
        Function to validate the initial positions by checking if the particles are closer than 2r. Not valid for
        positions to a Brownian particle with bigger radius
    :param positions: positions of the particles as a (N, 2) array
    :param radius: radius of the particles
    """
    smallest_distance = np.Inf
    for i in range(len(positions)):
        diff = norm(positions - np.tile(positions[i, :], reps=(len(positions), 1)), axis=1)
        diff = diff[diff != 0]
        smallest_distance = min(smallest_distance, np.min(diff))
    if smallest_distance > (2*radius):
        print('Smallest distance between particles greater than 2r')
    else:
        print('Overlapping positions!!')
    print(f'Smallest dist: {smallest_distance} 2r: {2 * radius}')


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
    # given initial values from exam in numerical physics
    mass_array = np.array([1, 10 ** 6])
    radius_array = np.array([0.001, 0.1])

    # initialize a simulation, where the box_of_particles is set for each parameter
    average_number_of_collisions_stop = 0.5
    simulation = Simulation(box_of_particles=None, stopping_criterion=average_number_of_collisions_stop)

    impact_parameters = np.linspace(-radius_array[1]*1.5, radius_array[1]*1.5, 500)
    scattering_angles = np.zeros_like(impact_parameters)

    for counter, impact_parameter in enumerate(impact_parameters):
        # update initial position of small particle
        initial_position_small_particle = [0.1, 0.5+impact_parameter]
        initial_positions[0, :] = initial_position_small_particle
        # create new system of particles
        box_of_particles = ParticleBox(number_of_particles=N,
                                       restitution_coefficient=xi,
                                       initial_positions=initial_positions,
                                       initial_velocities=initial_velocities,
                                       masses=mass_array,
                                       radii=radius_array)
        # update the system in the simulation
        simulation.box_of_particles = box_of_particles
        simulation.time_at_previous_collision = np.zeros(box_of_particles.N)

        simulation.simulate_until_given_number_of_collisions('scatteringAngle', output_timestep=1.0)
        velocity_small_particle_after_collision = simulation.box_of_particles.velocities[0, :]
        scattering_angles[counter] = compute_scattering_angle(np.array(initial_velocity_small_particle),
                                                              velocity_small_particle_after_collision)
        simulation.reset()  # set time and some parameters equal to zero to make simulation ready for new realization

    # save results in a matrix where each row gives impact parameter and scattering angle
    matrix_to_file = np.zeros((len(impact_parameters), 2))
    matrix_to_file[:, 0] = impact_parameters/np.sum(radius_array)
    matrix_to_file[:, 1] = scattering_angles
    # save to file
    np.save(file=os.path.join(results_folder, 'scattering_angle_func_impact_parameter'), arr=matrix_to_file)


def speed_distribution(use_equal_particles=True, number_of_runs=1):
    """
        Function to get speed of every particle after the system has reached equilibrium. Mainly solves problem 2 and
        3 in the exam. The procedure is based on initializing a system of many particles with the same initial speed.
        Then you start the simulation and let them collide until equilibrium have been reached. One then return the
        speed of each particle in order to create speed distribution plots.
    :param use_equal_particles: bool value that separates problem 2 and 3. If not equal: Second half will have m=4*m0
    :param number_of_runs: In order to achieve better statistics, take averages over multiple runs.
    """
    N = 2000  # number of particles
    xi = 1  # restitution coefficient
    v_0 = 0.2  # initial speed
    radius = 0.007  # radius of all particles
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}.npy'))
    radii = np.ones(N) * radius  # all particles have the same radius
    mass = np.ones(N)  # all particles get initially the same mass
    energy_matrix = np.zeros((N, 2))  # array with mass and speed to save to file

    if not use_equal_particles:
        # Problem 3: Second half of particles will have an mass which is bigger than the first half by a factor 4.
        mass[int(N / 2):] *= 4

    energy_matrix[:, 0] = mass  # mass does not change through the simulation

    # use stopping criterion that the avg_numb_collision should be equal to 2% of numb_particles.
    average_number_of_collisions_stop = N*0.02
    simulation = Simulation(box_of_particles=None, stopping_criterion=average_number_of_collisions_stop)

    for run_number in range(number_of_runs):
        # use same initial positions but new initial velocities every run
        # update velocities by getting new random angles, taking cos and sin and multiply with the initial speed
        velocities = random_uniformly_distributed_velocities(N, v_0)

        if run_number == 0:
            # save initial speed as a reference point
            initial_speeds = norm(velocities, axis=1)
            energy_matrix[:, 1] = initial_speeds
            if use_equal_particles:
                np.save(file=os.path.join(results_folder, f'distributionEqParticles_N_{N}_init_energy_matrix'),
                        arr=energy_matrix)
            else:
                np.save(file=os.path.join(results_folder, f'distributionNEqParticles_N_{N}_init_energy_matrix'),
                        arr=energy_matrix)
        # initialize system
        box_of_particles = ParticleBox(number_of_particles=N,
                                       restitution_coefficient=xi,
                                       initial_positions=positions,
                                       initial_velocities=velocities,
                                       masses=mass,
                                       radii=radii)
        # update simulation object
        simulation.box_of_particles = box_of_particles
        simulation.time_at_previous_collision = np.zeros(box_of_particles.N)

        # simulate system until given stopping criteria, save speed of particles, reset simulation object and repeat
        if use_equal_particles:
            simulation.simulate_until_given_number_of_collisions('distributionEqParticles', output_timestep=0.1)
            energy_matrix[:, 1] = norm(simulation.box_of_particles.velocities, axis=1)
            np.save(file=os.path.join(results_folder, f'distributionEqParticles_N_{N}_eq_energy_matrix_{run_number}'),
                    arr=energy_matrix)
        else:
            simulation.simulate_until_given_number_of_collisions('distributionNEqParticles', output_timestep=0.1)
            energy_matrix[:, 1] = norm(simulation.box_of_particles.velocities, axis=1)
            np.save(file=os.path.join(results_folder, f'distributionNEqParticles_N_{N}_eq_energy_matrix_{run_number}'),
                    arr=energy_matrix)

        simulation.reset()


def compute_avg_energy_development_after_time():
    """
        Function to mainly solve problem 4 in the exam by saving how the energy develop at a short output step. Will
        save the average kinetic energy of all particles(together and separate for m0 and m particles) at all outputs
        and save to file.
    """
    N = 2000  # number of particles
    v_0 = 0.2  # initial speed of all particles
    radius = 0.007  # radius of all particles
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}.npy'))
    # velocities = random_uniformly_distributed_velocities(N, v_0)
    radii = np.ones(N) * radius  # all particles have the same radius
    # first half will be m0 particles and second half is m particles with m=4*m0.
    mass = np.ones(N)
    mass[int(N / 2):] *= 4

    xi_list = [1, 0.9, 0.8]  # will do for multiple values of the restitution coefficient

    number_of_realizations = 5

    # create simulation object
    t_stop = 1
    dt = 0.01  # timestep
    simulation = Simulation(box_of_particles=None, stopping_criterion=t_stop)

    for xi in xi_list:
        matrix_to_file = np.zeros(
            (int(t_stop / dt) + 1, 4))  # matrix used to save information. Size given by stop and output
        for i in range(number_of_realizations):
            print(f"Run number: {i+1}")
            # new velocity vectors for each realization, but same initial positions
            velocities = random_uniformly_distributed_velocities(N, v_0)
            # initialize a ParticleBox with given parameters
            box_of_particles = ParticleBox(number_of_particles=N,
                                           restitution_coefficient=xi,
                                           initial_positions=positions,
                                           initial_velocities=velocities,
                                           masses=mass,
                                           radii=radii)
            # update simulation object with the given system
            simulation.box_of_particles = box_of_particles
            simulation.time_at_previous_collision = np.zeros(box_of_particles.N)

            time_array, energy_array_all, energy_array_m0, energy_array_m, mean_speed_array =\
                simulation.simulate_statistics_until_given_time('energyDevNEqParticles', output_timestep=dt,
                                                                equal_particles=False)

            # add results in a matrix with given form: time, avg_energy, avg_energy_m0 and avg_energy_m
            matrix_to_file[:, 0] += time_array
            matrix_to_file[:, 1] += energy_array_all
            matrix_to_file[:, 2] += energy_array_m0
            matrix_to_file[:, 3] += energy_array_m
            # reset simulation
            simulation.reset()
        matrix_to_file /= number_of_realizations  # get average values for multiple realizations
        # save to file
        np.save(file=os.path.join(results_folder, f'energyDevNEqParticles_N_{N}_xi_{xi}'),
                arr=matrix_to_file)


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
                                 masses, radii, simulation):
    """
        Help function to initial a ParticleBox and simulate a crater formation by solving problem 5. After simulation
        the function computes the size of the newly made crater.
    :param number_of_particles: number of particles in the box
    :param restitution_coefficient: restitution coefficient of the system. Tells how much energy is kept after colliding
    :param initial_positions: array with shape (numb_particles, 2) to indicate starting point for particles
    :param initial_velocities: array with shape (numb_particles, 2) to indicate starting velocity for particles
    :param masses: array with shape (numb_particles) to indicate the mass of each particle
    :param radii: array with shape (numb_particles) to indicate the radius of each particle
    :param simulation: Simulation object to be used in the simulation
    :return: crater_size
    """
    # initialize system
    box_of_particles = ParticleBox(number_of_particles=number_of_particles,
                                   restitution_coefficient=restitution_coefficient,
                                   initial_positions=initial_positions,
                                   initial_velocities=initial_velocities,
                                   masses=masses,
                                   radii=radii)
    # update simulation object with the given system
    simulation.box_of_particles = box_of_particles
    simulation.time_at_previous_collision = np.zeros(box_of_particles.N)
    # simulate
    simulation.simulate_until_given_energy('craterFormation', output_timestep=0.01)
    # compute crater size
    crater_size = get_crater_size(initial_positions[1:, :], simulation.box_of_particles.positions[1:, :], radii[-1])
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

        initial_avg_energy = 0.5*25*v0**2/N
        energy_stop = 0.1*initial_avg_energy
        simulation = Simulation(box_of_particles=None, stopping_criterion=energy_stop)

        crater_size = simulate_and_get_crater_size(N, xi, positions, velocities, mass, radii, simulation)
        print(f"Crater size: {crater_size}")

    elif study_parameter == 'mass':
        xi = 0.75
        v0 = 5
        velocities[0, :] = np.array([0, -v0])  # projectile velocity
        radii[0] *= 5  # projectile radius is 5 times bigger than the radius of the other particles
        mass_multiplication_array = np.linspace(3, 30, number_of_parameters)
        for counter, mass_multiplication_factor in enumerate(mass_multiplication_array):
            mass[0] *= mass_multiplication_factor

            initial_avg_energy = 0.5*mass_multiplication_factor*v0**2/N
            energy_stop = 0.1*initial_avg_energy
            simulation = Simulation(box_of_particles=None, stopping_criterion=energy_stop)

            crater_size = simulate_and_get_crater_size(N, xi, positions, velocities, mass, radii, simulation)
            parameter_study_matrix[counter, :] = [mass_multiplication_factor, crater_size]
            mass[0] /= mass_multiplication_factor
        np.save(file=os.path.join(results_folder, f'craterFormation_N_{N}_mass_study'), arr=parameter_study_matrix)
    elif study_parameter == 'radius':
        xi = 0.75
        v0 = 5
        velocities[0, :] = np.array([0, -v0])  # projectile velocity
        mass[0] *= 25  # projectile mass is 25 bigger than the mass of the other particles

        initial_avg_energy = 0.5 * 25 * v0 ** 2 / N  # energy does not depend on radius
        energy_stop = 0.1*initial_avg_energy
        simulation = Simulation(box_of_particles=None, stopping_criterion=energy_stop)

        radius_parameter_array = np.linspace(1, 10, number_of_parameters)
        for counter, radius_mult_factor in enumerate(radius_parameter_array):
            radii[0] *= radius_mult_factor

            crater_size = simulate_and_get_crater_size(N, xi, positions, velocities, mass, radii, simulation)
            simulation.reset()  # reset object to be ready for next simulation

            parameter_study_matrix[counter, :] = [radius_mult_factor, crater_size]
            radii[0] /= radius_mult_factor
        np.save(file=os.path.join(results_folder, f'craterFormation_N_{N}_radius_study'), arr=parameter_study_matrix)
    elif study_parameter == 'speed':
        xi = 0.75
        radii[0] *= 5  # projectile radius is 5 times bigger than the radius of the other particles
        mass[0] *= 25  # projectile mass is 25 bigger than the mass of the other particles
        v0_array = np.linspace(2, 8, number_of_parameters)
        for counter, v0 in enumerate(v0_array):
            velocities[0, :] = np.array([0, -v0])  # projectile velocity

            initial_avg_energy = 0.5*25*v0**2/N
            energy_stop = 0.1 * initial_avg_energy
            simulation = Simulation(box_of_particles=None, stopping_criterion=energy_stop)
            crater_size = simulate_and_get_crater_size(N, xi, positions, velocities, mass, radii, simulation)

            parameter_study_matrix[counter, :] = [v0, crater_size]
        np.save(file=os.path.join(results_folder, f'craterFormation_N_{N}_speed_study'), arr=parameter_study_matrix)
    elif study_parameter == 'xi':
        v0 = 5
        velocities[0, :] = np.array([0, -v0])  # projectile velocity
        radii[0] *= 5  # projectile radius is 5 times bigger than the radius of the other particles
        mass[0] *= 25  # projectile mass is 25 bigger than the mass of the other particles

        initial_avg_energy = 0.5*25*v0**2/N  # energy does not depend on restitution coefficient
        energy_stop = 0.1*initial_avg_energy
        simulation = Simulation(box_of_particles=None, stopping_criterion=energy_stop)

        xi_array = np.linspace(0.6, 0.8, number_of_parameters)
        for counter, xi in enumerate(xi_array):
            crater_size = simulate_and_get_crater_size(N, xi, positions, velocities, mass, radii, simulation)
            simulation.reset()  # reset object to be ready for new simulation for different xi

            parameter_study_matrix[counter, :] = [xi, crater_size]
        np.save(file=os.path.join(results_folder, f'craterFormation_N_{N}_xi_study'), arr=parameter_study_matrix)
    else:
        print("Study parameter not understood. Try again!")


def create_fractal_from_sticky_particles():
    """
        Function that creates fractal properties by allowing particles to stick together if a particle collides with
        a particle at rest. Initially the particle closest to the center is at rest and a fractal builds in time
    """
    N = 5000  # number of particles
    v_0 = 0.2  # initial speed of all particles
    xi = 1
    radius = 0.004  # radius of all particles
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
    # initialize simulation
    energy_stop = 1e-9
    simulation = Simulation(box_of_particles=box_of_particles, stopping_criterion=energy_stop)

    # simulate until all particles is at rest
    simulation.simulate_until_given_energy('fractalCreation', output_timestep=1, sticky_particles=True)
    # save the system at rest with all particle positions to file
    np.save(file=os.path.join(results_folder, f'fractalPositions_N_{N}_rad_{radius}'),
            arr=simulation.box_of_particles.positions)


def compute_mean_free_path():
    """
        Function that computes the mean free path in a simulation for the particles inside a radius of 0.1 from the
        center of the box. The results is computed for a different set of radius at given radius and below to see how
        the numerical results change as a function of the packing fraction/radius.
    """
    N = 2000  # number of particles
    v_0 = 0.2  # initial speed of all particles
    xi = 1
    initial_radius = 0.007  # radius of all particles
    positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{initial_radius}.npy'))
    velocities = random_uniformly_distributed_velocities(N, v_0)
    mass = np.ones(N)  # all particles have the same mass

    number_of_parameters = 10
    radius_parameter = np.linspace(0.1, 1, number_of_parameters)*initial_radius

    number_of_runs = 10
    t_stop = 5
    simulation = Simulation(box_of_particles=None, stopping_criterion=t_stop)

    matrix_to_file = np.zeros((number_of_parameters, 2))

    for i in range(number_of_runs):
        print(f"Run: {i}")
        for counter, radius in enumerate(radius_parameter):
            # velocities = random_uniformly_distributed_velocities(N, v_0)
            np.random.shuffle(velocities)

            radii = np.ones(N) * radius  # all particles have the same radius

            # initialize a ParticleBox with given parameters
            box_of_particles = ParticleBox(number_of_particles=N,
                                           restitution_coefficient=xi,
                                           initial_positions=positions,
                                           initial_velocities=velocities,
                                           masses=mass,
                                           radii=radii)

            # update simulation object with the given system
            simulation.box_of_particles = box_of_particles
            simulation.time_at_previous_collision = np.zeros(box_of_particles.N)

            # simulate
            simulation.simulate_until_given_time_mask_quantities('mfp', output_timestep=0.1, update_positions=True,
                                                                 save_positions=False)
            computed_mean_free_path =\
                simulation.box_of_particles.distance_to_collisions / simulation.box_of_particles.collision_count_particles
            mean_free_path_mask = np.mean(computed_mean_free_path[simulation.mask])
            matrix_to_file[counter, 0] = radius
            matrix_to_file[counter, 1] = mean_free_path_mask

            simulation.reset()

        # save results to file
        np.save(file=os.path.join(results_folder, f'mean_free_path_N_{N}_func_radius_run_{i}_eq_start'), arr=matrix_to_file)


def compute_diffusion_properties_from_mask_particles(single_bp_particle=False, eq_start=False, bigger_radius=False):
    """
        Functionality to compute the distance of particles starting close to the center of the box as a function of time
        in order to study diffusion properties. Function will save the results to file.
    """
    N = 2000  # number of particles
    v_0 = 0.2  # initial speed of all particles
    xi = 1
    radius = 0.007  # radius of all particles
    if bigger_radius:
        positions = np.load(
            os.path.join(init_folder, f'uniform_pos_around_bp_N_{N}_rad_{radius}_bp_rad_{radius * 3}.npy'))
    else:
        positions = np.load(os.path.join(init_folder, f'uniform_pos_N_{N}_rad_{radius}.npy'))
    if eq_start:
        velocities = np.load(os.path.join(init_folder, f'eq_vel_N_{N}_rad_{radius}.npy'))
    else:
        velocities = random_uniformly_distributed_velocities(N, v_0)

    radii = np.ones(N) * radius  # all particles have the same radius
    mass = np.ones(N)  # all particles have the same mass

    distance_to_center = norm(positions - np.tile([0.5, 0.5], reps=(len(positions), 1)), axis=1)
    closest_particle = np.argmin(distance_to_center)
    if single_bp_particle:
        if not bigger_radius:
            mass[closest_particle] *= 10
        else:
            radii[closest_particle] *= 3

    validate_positions(positions, radius)

    number_of_runs = 10
    t_stop = 5
    timestep = 0.02
    simulation = Simulation(box_of_particles=None, stopping_criterion=t_stop)

    if single_bp_particle:
        simulation.mask = distance_to_center == distance_to_center[closest_particle]

    matrix_to_file = np.zeros((int(t_stop/timestep)+1, 3))
    if single_bp_particle:
        matrix_to_file = np.zeros((int(t_stop / timestep) + 1, 5))

    for i in range(number_of_runs):
        print(f'Run number: {i+1}')
        if eq_start:
            np.random.shuffle(velocities)
        else:
            velocities = random_uniformly_distributed_velocities(N, v_0)
        if single_bp_particle:
            if not bigger_radius:
                velocities[closest_particle] *= 1

        # initialize a ParticleBox with given parameters
        box_of_particles = ParticleBox(number_of_particles=N,
                                       restitution_coefficient=xi,
                                       initial_positions=positions,
                                       initial_velocities=velocities,
                                       masses=mass,
                                       radii=radii)

        # update simulation object with the given system
        simulation.box_of_particles = box_of_particles
        simulation.time_at_previous_collision = np.zeros(box_of_particles.N)
        if single_bp_particle:
            time_array, bp_position_array, bp_velocity_array = \
                simulation.simulate_until_given_time_bp('brownianParticle', output_timestep=timestep,
                                                        save_positions=True)
            matrix_to_file[:, 0] = time_array
            matrix_to_file[:, 1] = bp_position_array[:, 0]
            matrix_to_file[:, 2] = bp_position_array[:, 1]
            matrix_to_file[:, 3] = bp_velocity_array[:, 0]
            matrix_to_file[:, 4] = bp_velocity_array[:, 1]
            if bigger_radius:
                np.save(file=os.path.join(results_folder, f'diffProperties_3r0m0_particle_tmax_{t_stop}_run_{i+200}'),
                        arr=matrix_to_file)
            else:
                np.save(file=os.path.join(results_folder, f'diffProperties_r010m0_particle_tmax_{t_stop}_run_{i}_eq_speed'),
                        arr=matrix_to_file)
        else:
            time_array, mean_quadratic_distance_array, mean_quadratic_speed_array = \
                simulation.simulate_until_given_time_mask_quantities('diffusionProperties', output_timestep=timestep,
                                                                     save_positions=True)

            matrix_to_file[:, 0] = time_array
            matrix_to_file[:, 1] = mean_quadratic_distance_array
            matrix_to_file[:, 2] = mean_quadratic_speed_array
            if eq_start:
                np.save(file=os.path.join(results_folder,
                                          f'diffProperties_r0m0_particle_tmax_{t_stop}_run_{i}_eq_start'),
                        arr=matrix_to_file)
            else:
                np.save(file=os.path.join(results_folder, f'diffProperties_r0m0_particle_tmax_{t_stop}_run_{i+30}'),
                        arr=matrix_to_file)

        simulation.reset()
