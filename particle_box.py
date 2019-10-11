import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import heapq
import os
import shutil

from scipy.linalg import norm

from config import results_folder, plots_folder, init_folder


class ParticleBox:
    """
        Class which implements a square box containing a number of particles with a given radius and mass. The class
        has functionality to implement collisions with vertical and horizontal walls and between two particles. The main
        function is simulate which is a implementation of a event driven simulation of particles colliding in a box.
    """

    def __init__(self, number_of_particles, restitution_coefficient, initial_positions, initial_velocities, masses, radii):
        """
            Initialize the ParticleBox class
        :param number_of_particles: number of particles in the box
        :param restitution_coefficient: number giving how much energy is lost during a collision.
        :param initial_positions: Array with shape = (number_of_particles, 2) giving x and y coordinates
        :param initial_velocities: Array with shape = (number_of_particles, 2) giving x and y velocities
        :param masses: Array with length number_of_particles giving the masses of each particle
        :param radii: Array with length number_of_particles giving the radius of each particle
        """
        self.N = number_of_particles  # amount of particles
        self.restitution_coefficient = restitution_coefficient  # coefficient determining the energy lost in collisions
        # initialize variables used in the class
        self.positions = np.zeros((self.N, 2))  # positions of particles
        self.old_positions = np.zeros((self.N, 2))  # help variable to calculate distance from start and mean free path
        self.velocities = np.zeros((self.N, 2))  # velocities of particles
        self.masses = np.zeros(self.N)  # mass of each particle
        self.radii = np.zeros(self.N)  # radius of each particles
        self.collision_count_particles = np.zeros(self.N)  # array keeping track of the number of collisions
        self.time_at_previous_collision = np.zeros(self.N)  # array keeping track of the time at the previous collision
        self.distance_to_collisions = np.zeros(self.N)  # array keeping track of the distance travelled for collisions
        self.tc = 0  # variable used in the TC model to avoid inelastic collapse
        self.simulation_time = 0
        # set parameters equal to the input to the class. Use .copy() such that the parameters can be used in outer loop
        self.positions = initial_positions.copy()
        self.old_positions = initial_positions.copy()
        self.velocities = initial_velocities.copy()
        self.masses = masses
        self.radii = radii
        # a priority queue / heap queue of tuples of (time_collision, collision_entities, collision_count when
        # computing the collision). The collision count at computation is used to ignore non-valid collisions due to
        # the involved particles being in other collisions between computation and collision.
        self.collision_queue = []  # heap queue needs list structure to work

    def collision_horizontal_wall(self, particle_number, restitution_coefficient):
        """
            Function to solve a collision with a particle and a horizontal wall by updating the velocity vector
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :param restitution_coefficient: number giving how much energy is lost during a collision.
        """
        self.velocities[particle_number, :] *= restitution_coefficient*np.array([1, -1])

    def collision_vertical_wall(self, particle_number, restitution_coefficient):
        """
            Function to solve a collision with a particle and a vertical wall by updating the velocity vector
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :param restitution_coefficient: number giving how much energy is lost during a collision.
        """
        self.velocities[particle_number, :] *= restitution_coefficient * np.array([-1, 1])

    def collision_particles(self, particle_one, particle_two, restitution_coefficient):
        """
            Function to solve a collision between two particles by updating the velocity vector for both particles
        :param particle_one: the index of particle number one
        :param particle_two: the index of particle number two
        :param restitution_coefficient: number giving how much energy is lost during a collision.
        """
        mass_particle_one, mass_particle_two = self.masses[particle_one], self.masses[particle_two]  # get masses
        delta_x = self.positions[particle_two, :] - self.positions[particle_one, :]  # difference in position
        delta_v = self.velocities[particle_two, :] - self.velocities[particle_one, :]  # difference in velocity
        r_squared = (self.radii[particle_one] + self.radii[particle_two]) ** 2  # distance from center to center
        # update velocities of the particles
        self.velocities[particle_one, :] += delta_x*((1+restitution_coefficient)*mass_particle_two*np.dot(delta_v, delta_x)/((mass_particle_one+mass_particle_two)*r_squared))
        self.velocities[particle_two, :] -= delta_x*((1+restitution_coefficient)*mass_particle_one*np.dot(delta_v, delta_x)/((mass_particle_one+mass_particle_two)*r_squared))

    def time_at_collision_vertical_wall(self, particle_number):
        """
            Function that computes at what time a particle will collide with a vertical wall
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :return: the time when particle particle_number will collide with a vertical wall
        """
        velocity_x = self.velocities[particle_number, 0]  # velocity in the x-direction for the particle
        position_x = self.positions[particle_number, 0]  # x-position of the particle
        radius = self.radii[particle_number]  # radius of the particle
        # compute time until collision
        if velocity_x > 0:
            time_until_collision = (1-radius-position_x) / velocity_x
        elif velocity_x < 0:
            time_until_collision = (radius-position_x) / velocity_x
        else:
            time_until_collision = np.inf
        return time_until_collision + self.simulation_time

    def time_at_collision_horizontal_wall(self, particle_number):
        """
            Function that computes at what time a particle will collide with a horizontal wall
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :return: the time when particle particle_number will collide with a horizontal wall
        """
        velocity_y = self.velocities[particle_number, 1]  # velocity in the y-direction of the particle
        position_y = self.positions[particle_number, 1]  # y position of the particle
        radius = self.radii[particle_number]  # radius of the particle
        # compute time until collision
        if velocity_y > 0:
            time_until_collision = (1 - radius - position_y) / velocity_y
        elif velocity_y < 0:
            time_until_collision = (radius - position_y) / velocity_y
        else:
            time_until_collision = np.inf
        return time_until_collision + self.simulation_time

    def time_at_collision_particles(self, particle_number):
        """
            Function that computes the time until a particle collides with all other particles
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        :return: the time when particle particle_number will collide with all of the other particles
        """
        # difference from particle particle_number to all other particles
        delta_x = self.positions - np.tile(self.positions[particle_number, :], reps=(len(self.positions), 1))
        # difference in velocity from particle particle_number to all other particles
        delta_v = self.velocities - np.tile(self.velocities[particle_number, :], reps=(len(self.velocities), 1))
        r_squared = (self.radii[particle_number] + self.radii) ** 2  # array of center to center distances
        dvdx = np.sum(delta_v*delta_x, axis=1)  # dot product between delta_v and delta_x
        dvdv = np.sum(delta_v*delta_v, axis=1)  # dot product between delta_v and delta_v
        d = dvdx ** 2 - dvdv * (norm(delta_x, axis=1) ** 2 - r_squared)  # help array quantity
        time_until_collisions = np.ones(self.N)*np.inf  # assume no particles is going to collide
        boolean = np.logical_and(dvdx < 0, d > 0)  # both these conditions must be valid particle-particle collision
        # check if there exist some valid particle-particle collisions for particle particle_number
        if np.sum(boolean) > 0:
            # compute time until collision
            time_until_collisions[boolean] = -1 * ((dvdx[boolean] + np.sqrt(d[boolean])) / (dvdv[boolean]))
        return time_until_collisions + self.simulation_time

    def add_collision_horizontal_wall_to_queue(self, particle_number):
        """
            Help function to compute time at collision with horizontal wall for a given particle, create collision
            tuple and push the tuple into the heap queue.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        """
        time_hw = self.time_at_collision_horizontal_wall(particle_number)  # time at collision
        # create collision tuple on desired form
        tuple_hw = (time_hw, [particle_number, 'hw'], [self.collision_count_particles[particle_number]])
        # push to heap queue
        heapq.heappush(self.collision_queue, tuple_hw)

    def add_collision_vertical_wall_to_queue(self, particle_number):
        """
            Help function to compute time at collision with vertical wall for a given particle, create collision
            tuple and push the tuple into the heap queue.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        """
        time_vw = self.time_at_collision_vertical_wall(particle_number)  # time at collision
        # create collision tuple on desired form
        tuple_vw = (time_vw, [particle_number, 'vw'], [self.collision_count_particles[particle_number]])
        # push to heap queue
        heapq.heappush(self.collision_queue, tuple_vw)

    def add_collisions_particle_to_queue(self, particle_number):
        """
            Help function to compute time at collision with all particles for a given particle, create collision
            tuples and push valid tuples into the heap queue.
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        """
        time_at_collisions = self.time_at_collision_particles(particle_number)  # get time at all collisions
        collision_particles = np.arange(self.N)  # create a list of possible collision candidates
        # only regard valid collisions by removing all entries which are np.inf
        collision_particles = collision_particles[time_at_collisions != np.inf]
        time_at_collisions = time_at_collisions[time_at_collisions != np.inf]
        # check if there are any valid collisions
        if len(time_at_collisions) > 0:
            # iterate through all valid collisions
            for i in range(len(time_at_collisions)):
                # create collision tuple of valid form
                tuple_particle_collision = (time_at_collisions[i], [particle_number, collision_particles[i]],
                                            [self.collision_count_particles[particle_number],
                                             self.collision_count_particles[collision_particles[i]]])
                # push tuple to heap queue
                heapq.heappush(self.collision_queue, tuple_particle_collision)

    def create_initial_priority_queue(self):
        """
            Help function that initialize the heap queue by iterating though all particles and push all possible
            collisions to the heap queue.
        """
        for particle_number in range(self.N):  # iterate through each particle
            self.add_collision_horizontal_wall_to_queue(particle_number)  # add collision with horizontal wall
            self.add_collision_vertical_wall_to_queue(particle_number)  # add collision with vertical wall
            self.add_collisions_particle_to_queue(particle_number)  # add collisions with other particles

    def valid_collision(self, collision_tuple):
        """
            Function that validates a proposed new collision by looking at the collision tuple information
        :param collision_tuple: tuple with information: (coll_time, coll entities, coll_count_comp_coll)
        """
        # check if first entity has not been in collision since computation of collision
        if collision_tuple[2][0] == self.collision_count_particles[collision_tuple[1][0]]:
            # if there is a second particle and it has been in another collision -> False
            if len(collision_tuple[2]) == 2 and \
                    collision_tuple[2][1] != self.collision_count_particles[collision_tuple[1][1]]:
                return False
            # Accept if there is only one particle, or the second particle has not been in another collision
            else:
                return True
        else:
            return False

    def update_queue_new_collisions_particle(self, particle_number):
        """
            Help function that add all new possible collisions for a particle after being part of a collision
        :param particle_number: the index of a particle in order to retrieve and/or update the particle data
        """
        self.add_collision_horizontal_wall_to_queue(particle_number)
        self.add_collision_vertical_wall_to_queue(particle_number)
        self.add_collisions_particle_to_queue(particle_number)

    def print_output(self, average_number_of_collisions, avg_energy):
        """
            Function to print desired output from the simulation
        """
        print('--- Output ---')
        print(f'Simulation time: {self.simulation_time}')
        print("Priority queue elements: ", len(self.collision_queue))
        print(f"Avg energy: {avg_energy}")
        print(f"Average number of collisions: {average_number_of_collisions}")
        # print(f"Number of collisions per time: {np.sum(self.collision_count_particles)/(self.simulation_time+0.001)}")

    def save_particle_positions(self, simulation_folder, picture_number, mask=None):
        """
            Function to save particle positions as a png image at a output time
        :param simulation_folder: folder to save png images
        :param picture_number: int parameters stating what picture is saved in order to keep order easily
        :param mask: boolean array to indicate the particles starting close to center to plot in different color. If
        None the function will plot all particles in the same color.
        """
        if mask is None:
            fig, ax = plt.subplots()
            coll = matplotlib.collections.EllipseCollection(self.radii * 2, self.radii * 2,
                                                            np.zeros_like(self.radii),
                                                            offsets=self.positions, units='width',
                                                            transOffset=ax.transData)
            ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k')
            ax.add_collection(coll)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            plt.savefig(os.path.join(simulation_folder, f"{picture_number}.png"))
            plt.close()
        else:
            fig, ax = plt.subplots()
            coll = matplotlib.collections.EllipseCollection(self.radii[~mask] * 2, self.radii[~mask] * 2,
                                                            np.zeros_like(self.radii[~mask]),
                                                            offsets=self.positions[~mask, :], units='width',
                                                            transOffset=ax.transData)

            ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k')
            ax.add_collection(coll)
            coll = matplotlib.collections.EllipseCollection(self.radii[mask] * 2, self.radii[mask] * 2,
                                                            np.zeros_like(self.radii[mask]),
                                                            offsets=self.positions[mask, :], units='width',
                                                            transOffset=ax.transData, facecolors='red')
            ax.add_collection(coll)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            plt.savefig(os.path.join(simulation_folder, f"{picture_number}.png"))
            plt.close()

    def compute_energy(self, equal_particles):
        """
            Function to compute the energy in the system of particles. Based on boolean input can compute average
            kinetic energy of all particles or compute average for all, for m0 and for m.
        :param equal_particles: bool value indicating if there exist similar type of particles. Separates problems
        :return: average energy of all particles or average energy of all particles, m0 particles and m particles
        """
        energy = 0.5 * self.masses * np.sum(self.velocities * self.velocities, axis=1)
        avg_energy = np.mean(energy)  # average kinetic energy of all particles
        if equal_particles:
            return avg_energy
        else:
            avg_energy_m0 = np.mean(energy[:int(self.N/2)])  # average kinetic energy of m0 particles
            avg_energy_m = np.mean(energy[int(self.N/2):])  # average kinetic energy of m(=4m0) particles
            return avg_energy, avg_energy_m0, avg_energy_m

    def simulation(self, problem_number, output_timestep=1.0):
        """
            Implementation of an event driven simulation of particles colliding in a square box. Mainly used the above
            functions systematically to increment time from event to event.
        :param problem_number: int telling what problem one is trying to solve. Is used to change stopping criterion
        and the output of the simulation
        :param output_timestep: parameter used to determine how often do to an output in the simulation
        :return: Depend on the problem_number, given in the list below.
        problem_number=0: None since it is only used for testing,
        problem_number=1: Velocity after one collision,
        problem_number=2: Speed of each particle as a an array after the system has reached equilibrium,
        problem_number=3: same as above. Used when half of the particles have 4 times larger mass than the other half,
        problem_number=4: average energy of all particles, m0 particles, m particles and time at each output,
        problem_number=5: None since in the crater formation one use the positions which is retrieved from object,
        problem_number=6: mean square displacement from initial starting points and time at each output,
        problem_number=7: mean free path as the mean distance travelled between each particle particle collision,
        problem_number=8: None since in the fractal formation one use the positions which is retrieved from object
        problem_number=9: TBA
        """

        # TODO: cleanup. Maybe not possible to be clean due to different computations / actions

        print('Starting simulation..')
        print(f'problem: {problem_number} N: {self.N} and xi: {self.restitution_coefficient}..')
        print('---------------------')
        print('Creating initial queue..')
        # create folder in order to save particle positions as a png files throughout the simulation
        simulation_folder = os.path.join(plots_folder, f'simulation_prob_{problem_number}_N_{self.N}_xi_'
                                                       f'{self.restitution_coefficient}')
        if not os.path.isdir(simulation_folder):
            os.mkdir(simulation_folder)
        else:
            shutil.rmtree(simulation_folder)
            os.mkdir(simulation_folder)

        mask = None  # variable used to indicate whether or not to plot the middle particles in separate color
        next_output_time = 0  # value that keeps track of when the next output will occur
        output_number = 0  # value to keep the order of pictures saved at each output
        self.create_initial_priority_queue()  # Initialize the queue with all possible starting collisions
        average_number_of_collisions_stopping_criterion = 0  # stopping criterion
        # The problem_number is used to determine the stopping criterion since it depend on what you want to do
        if problem_number == 0:
            # Testing
            average_number_of_collisions_stopping_criterion = 5
        elif problem_number == 1:
            # Scattering angle computation
            average_number_of_collisions_stopping_criterion = 0.5
        elif problem_number == 2:
            # Speed distribution for one type of mass. Stopping criterion need to allow system to reach equilibrium
            average_number_of_collisions_stopping_criterion = self.N * 0.02
        elif problem_number == 3:
            # Speed distribution for two different masses. Stopping criterion need to allow system to reach equilibrium
            average_number_of_collisions_stopping_criterion = self.N * 0.02
        elif problem_number == 4:
            # look at system for small intervals
            average_number_of_collisions_stopping_criterion = self.N * 0.02
            avg_energy_array = []
            avg_energy_m0_array = []
            avg_energy_m_array = []
            time_array = []
        elif problem_number == 5:
            # crater formation. When the energy is reduced to 10% of initial energy the while-loop is broken
            # the indicated value here is only for show. The energy is reduced to stopping limit before this criterion.
            average_number_of_collisions_stopping_criterion = self.N
        elif problem_number == 6:
            # investigation of the diffusion properties of the middle particles.
            average_number_of_collisions_stopping_criterion = self.N
            distance_to_middle_position = norm((self.positions -
                                                np.tile([0.5, 0.5], reps=(len(self.positions), 1))), axis=1)
            time_array = np.zeros(int(20/output_timestep)+1)
            mean_quadratic_distance_array = np.zeros_like(time_array)
            mean_speed_array = np.zeros_like(time_array)
            mean_speed_array[0] = np.mean(norm(self.velocities, axis=1))
            mask = distance_to_middle_position < 0.1
        elif problem_number == 7:
            # computation of mean free path
            average_number_of_collisions_stopping_criterion = self.N * 0.01
        elif problem_number == 8:
            # create fractal properties
            average_number_of_collisions_stopping_criterion = self.N
        else:
            print('Not a valid choice of problem number! Try again')
            exit()

        if problem_number == 4:
            # detailed energy information
            avg_energy, avg_energy_m0, avg_energy_m = self.compute_energy(equal_particles=False)  # initial energy setup
            avg_energy_array.append(avg_energy)
            avg_energy_m0_array.append(avg_energy_m0)
            avg_energy_m_array.append(avg_energy_m)
            time_array.append(next_output_time)
        else:
            # energy information
            avg_energy = self.compute_energy(equal_particles=True)  # initial energy for all particles

        # give initial output and save particle positions
        self.print_output(0, avg_energy)
        self.save_particle_positions(simulation_folder, output_number, mask=mask)
        next_output_time += output_timestep
        output_number += 1

        if problem_number == 5:
            # save initial energy
            initial_avg_energy = avg_energy

        print('Event driven simulation in progress..')
        # run until the average number of collisions has reached the stopping criterion
        average_number_of_collisions = 0  # average_number_of_collisions is updated every time a collision is accepted
        while average_number_of_collisions < average_number_of_collisions_stopping_criterion:

            collision_tuple = heapq.heappop(self.collision_queue)  # pop the earliest element from heap queue
            time_at_collision = collision_tuple[0]  # extract time_at_collision
            # do output while waiting for time_at_collision if next output is earlier
            while next_output_time < time_at_collision:
                # update positions and simulation_time by incrementing time until next output
                dt = next_output_time - self.simulation_time
                self.positions += self.velocities * dt
                self.simulation_time += dt

                if problem_number == 4:
                    # retrieve detailed energy information
                    avg_energy, avg_energy_m0, avg_energy_m = self.compute_energy(
                        equal_particles=False)  # computed average energies for all particles, and only m0/m particles
                    avg_energy_array.append(avg_energy)
                    avg_energy_m0_array.append(avg_energy_m0)
                    avg_energy_m_array.append(avg_energy_m)
                    time_array.append(next_output_time)
                else:
                    # get energy for output printing
                    avg_energy = self.compute_energy(equal_particles=True)  # computed average energy for all particles

                if problem_number == 6:
                    # save detailed information about distance from initial position
                    distance_to_initial_position = norm((self.positions[mask, :] - self.old_positions[mask, :]), axis=1)
                    mean_quadratic_distance = np.mean(distance_to_initial_position**2)
                    time_array[output_number] = next_output_time
                    mean_speed_array[output_number] = np.mean(norm(self.velocities, axis=1))
                    mean_quadratic_distance_array[output_number] = mean_quadratic_distance

                # give output and save particle positions
                self.print_output(average_number_of_collisions, avg_energy)
                self.save_particle_positions(simulation_folder, output_number, mask=mask)

                next_output_time += output_timestep
                output_number += 1

            if time_at_collision == np.inf:  # check if there is only np.inf in the queue. Should not happen :)
                print('No more collisions can occur!')
                break

            if self.valid_collision(collision_tuple):  # do collision if it is considered valid
                dt = time_at_collision - self.simulation_time
                # update positions and simulation_time by incrementing time until the collision
                self.positions += self.velocities * dt
                self.simulation_time += dt

                object_one = collision_tuple[1][0]  # particle number of particle one
                object_two = collision_tuple[1][1]  # particle number of particle two, or 'hw'/'vw' to indicate wall
                time_since_previous_collision_part_one = time_at_collision - self.time_at_previous_collision[object_one]

                # update velocities by letting a collision happen
                if object_two == 'hw':
                    # update velocity of particle colliding with hw
                    if time_since_previous_collision_part_one < self.tc:
                        print('TC model')
                        self.collision_horizontal_wall(object_one, 1)  # set xi equal to 1 to avoid inelastic collapse
                    else:
                        self.collision_horizontal_wall(object_one, self.restitution_coefficient)
                elif object_two == 'vw':
                    # update velocity of particle in colliding with vw
                    if time_since_previous_collision_part_one < self.tc:
                        print('TC model')
                        self.collision_vertical_wall(object_one, 1)  # set xi equal to 1 to avoid inelastic collapse
                    else:
                        self.collision_vertical_wall(object_one, self.restitution_coefficient)
                else:
                    time_since_previous_collision_part_two =\
                        time_at_collision - self.time_at_previous_collision[object_two]
                    # update velocity of particles in collision
                    if time_since_previous_collision_part_one < self.tc or \
                            time_since_previous_collision_part_two < self.tc:
                        print('TC model')
                        self.collision_particles(object_one, object_two, 1)  # in order to avoid inelastic collapse
                    else:
                        self.collision_particles(object_one, object_two, self.restitution_coefficient)
                    # if a particle has collided with a particle with huge mass, set velocity of both to zero
                    if problem_number == 8:
                        if self.masses[object_one]+self.masses[object_two] > 10**5:
                            self.masses[object_one] = 10**10
                            self.masses[object_two] = 10**10
                            self.velocities[object_one, :] = [0, 0]
                            self.velocities[object_two, :] = [0, 0]

                self.collision_count_particles[object_one] += 1  # update collision count
                if object_two not in ['hw', 'vw']:  # if there is a second particle involved
                    self.collision_count_particles[object_two] += 1  # update collision count particle two
                    self.update_queue_new_collisions_particle(object_two)  # get new collisions for particle two
                    self.time_at_previous_collision[object_two] = time_at_collision  # add time at collision

                self.update_queue_new_collisions_particle(object_one)  # get new collisions for particle one
                self.time_at_previous_collision[object_one] = time_at_collision  # add time at collision
                # update average number of collisions since one or two particles have been in a collision
                average_number_of_collisions = np.mean(self.collision_count_particles)

                if problem_number == 5:
                    # check if the energy in the system is reduced to below 10% of initial energy
                    avg_energy = self.compute_energy(equal_particles=True)
                    if avg_energy < 0.1*initial_avg_energy:
                        # break loop
                        break
                if problem_number == 7:
                    # add distance travelled for each particle involved
                    distance_moved = norm(self.positions[object_one, :] - self.old_positions[object_one, :])
                    self.distance_to_collisions[object_one] += distance_moved
                    self.old_positions[object_one, :] = self.positions[object_one, :]
                    if object_two not in ['hw', 'vw']:  # if there is a second particle involved
                        distance_moved = norm(self.positions[object_two, :] - self.old_positions[object_two, :])
                        self.distance_to_collisions[object_two] += distance_moved
                        self.old_positions[object_two, :] = self.positions[object_two, :]
            if problem_number == 6:
                # end after a given time
                if self.simulation_time >= 20:
                    average_number_of_collisions = average_number_of_collisions_stopping_criterion
            if problem_number == 8:
                # if all particles have stopped
                if avg_energy < 1e-9:
                    average_number_of_collisions = average_number_of_collisions_stopping_criterion

        print('Simulation done!')
        print('---------------------')
        if problem_number == 1:
            # return velocity vector after collision
            return self.velocities[0, :]
        elif problem_number == 2 or problem_number == 3:
            # return speeds after system has reached equilibrium
            speeds = np.sqrt(np.sum(self.velocities*self.velocities, axis=1))
            return speeds
        elif problem_number == 4:
            # return avg_energy arrays and time_array
            return avg_energy_array, avg_energy_m0_array, avg_energy_m_array, time_array
        elif problem_number == 6:
            # return mean quadratic distance vector and time array
            return mean_quadratic_distance_array, mean_speed_array, time_array
        elif problem_number == 7:
            # compute the mean free path for each particle
            mean_free_path = self.distance_to_collisions/self.collision_count_particles
            return mean_free_path
        elif problem_number == 8:
            # positions of all particles clustered together
            return self.positions


if __name__ == "__main__":
    import time
    start_time = time.time()
    print(f"Time used: {time.time()-start_time}")
