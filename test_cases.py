import numpy as np

# Implementation of various tests from the numerical physics exam


def test_one_particle():
    """
        Function to check that for a system wih one particle, the particle bounces of all walls in a endless loop
    """
    # B.1 One particle
    N = 1
    xi = 1
    position = np.array([0.9, 0.4]).reshape(1, 2)
    velocity = np.array([-0.5, -0.5]).reshape(1, 2)
    mass = np.ones(1).reshape(1, 1)
    radius = np.ones(1).reshape(1, 1)*10**(-2)
    # create system object
    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=position,
                                   initial_velocities=velocity,
                                   masses=mass,
                                   radii=radius)
    # create simulation object
    simulation = Simulation(box_of_particles, stopping_criterion=5)
    # simulate until the average number of collisions is >= 5
    simulation.simulate_until_given_number_of_collisions('testOneParticle', output_timestep=0.2)


def test_two_particles():
    """
        Function to check the system for tests with two particles. Can both check that two particles hitting each
        other head on with opposite speeds change speeds, go back to the wall and then collide again. Can also check
        for a certain impact parameter that the scattering angle will be 90 degrees for similar particles.
    """
    # B.2 Two particles
    # first test of two particles hitting each other, bouncing and then colling again.
    N = 2
    xi = 1
    positions = np.array([[0.15, 0.5], [0.75, 0.5]])
    velocities = np.array([[0.05, 0], [-0.05, 0]])
    mass = np.ones(N)
    radius = np.ones(N) * 0.05
    # create system object
    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radius)
    # create simulation object
    simulation = Simulation(box_of_particles, stopping_criterion=5)
    # simulate until the average number of collisions is equal to 5
    simulation.simulate_until_given_number_of_collisions('testTwoParticles', output_timestep=0.5)

    # check if the scattering angle in 90 degrees for certain choice of impact parameter b.
    # positions = np.array([[0.15, 0.5+(radius[0]+radius[1])/np.sqrt(2)], [0.75, 0.5]])
    #
    # box_of_particles = ParticleBox(number_of_particles=N,
    #                                restitution_coefficient=xi,
    #                                initial_positions=positions,
    #                                initial_velocities=velocities,
    #                                masses=mass,
    #                                radii=radius)
    #
    # simulation = Simulation(box_of_particles, stopping_criterion=0.5)
    # simulation.simulate_until_given_number_of_collisions('testImpactParameter', output_timestep=0.2)
    # velocity_after = simulation.box_of_particles.velocities[0, :]
    # scattering_angle = uf.compute_scattering_angle(velocities[0, :], velocity_after)
    # print(velocities[0, :], velocity_after)
    # print(scattering_angle*180/np.pi)


def test_multiple_particles():
    """
        Function that checks if the energy is conserved for many particles. Is seen from the printed outputs.
    """
    # B.3 Many particles
    N = 100
    xi = 1
    v_0 = 0.15
    positions, min_radius = uf.random_positions_and_radius(N)
    random_angles = np.random.random(N)*(2*np.pi)
    velocities = np.zeros((N, 2))
    velocities[:, 0], velocities[:, 1] = np.cos(random_angles), np.sin(random_angles)
    velocities *= v_0
    radii = np.ones(N) * min_radius
    mass = np.ones(N)

    box_of_particles = ParticleBox(number_of_particles=N,
                                   restitution_coefficient=xi,
                                   initial_positions=positions,
                                   initial_velocities=velocities,
                                   masses=mass,
                                   radii=radii)
    simulation = Simulation(box_of_particles, stopping_criterion=0.5)
    simulation.simulate_until_given_number_of_collisions('testManyParticles', output_timestep=0.1)


if __name__ == "__main__":
    import utility_functions as uf
    from particle_box import ParticleBox
    from simulation import Simulation
    test_type = {0: 'One particle', 1: 'Two particles', 2: 'Multiple particles'}[1]
    if test_type == 'One particle':
        test_one_particle()
    elif test_type == 'Two particles':
        test_two_particles()
    elif test_type == 'Multiple particles':
        test_multiple_particles()
