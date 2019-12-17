import time
import numpy as np

import utility_functions as util_funcs

# File used to solve different problems by using the functions in utility_functions

start_time = time.time()
# choose what problem one want to solve by simulating particle collisions in 2D
problem = {1: 'Scattering angle',
           2: 'Speed distribution',
           3: 'Energy development',
           4: 'Initial positions',
           5: 'Crater formation',
           6: 'Diffusion properties',
           7: 'Mean free path',
           8: 'Fractal'}[5]
print(f"Problem: {problem}")
if problem == 'Scattering angle':
    # scattering angle from one small particles bouncing of one large particle at rest
    util_funcs.scattering_angle_func_impact_parameter()
elif problem == 'Speed distribution':
    # let system evolve in time until enough collisions has occurred to assume equilibrium has been reached.
    use_same_particle = True  # choice of whether to let all particles have equal mass, or half have 4 times larger
    util_funcs.speed_distribution(use_equal_particles=use_same_particle, number_of_runs=30)
elif problem == 'Energy development':
    # let the system evolve in time until and see how the energy changes for different restitution coefficients
    util_funcs.compute_avg_energy_development_after_time()
elif problem == 'Initial positions':
    # create files to be used as initial positions
    N = 2000
    radius_particle = 1 / np.sqrt(4*N*np.pi)
    # radius of particle is chosen in order to give ca. 1/2 packing fraction for wall in crater formation
    radius_particle = np.round(radius_particle+0.0005, decimals=3)
    # save random positions to initial_positions folder
    # if one use y_max = 1 => get about 30% packing fraction for entire region
    util_funcs.random_positions_for_given_radius(N, radius_particle, y_max=1.0, brownian_particle=False)
elif problem == 'Crater formation':
    # study crater formation by hitting a wall of densely packed particles with a projectile
    study_parameters = {0: 'test', 1: 'mass', 2: 'radius', 3: 'speed', 4: 'xi'}
    choice_of_parameter = study_parameters[0]
    util_funcs.study_crater_formation(study_parameter=choice_of_parameter)
elif problem == 'Diffusion properties':
    # study diffusion properties by looking at the mean quadratic displacement from initial position
    single_bp_particle = True
    eq_start = False
    util_funcs.compute_diffusion_properties_from_mask_particles(single_bp_particle=single_bp_particle,
                                                                eq_start=eq_start, bigger_radius=True)
elif problem == 'Mean free path':
    # study diffusion properties by looking at the mean free path by storing distance travelled between collisions
    util_funcs.compute_mean_free_path()
elif problem == 'Fractal':
    # create a fractal by letting a particle hitting a particle as rest simulated as sticky particles getting stuck
    util_funcs.create_fractal_from_sticky_particles()

print(f"Time used: {time.time() - start_time}")
