import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import geopandas as gpd
from geopy.geocoders import Nominatim

pi = np.pi
# density of water
rho = 1000.0  # kg / m^3
# specific heat of water
cp = 4186.0  # J  / kg K
# depth of water
z = 10.0  # m
# Stefan Boltzmann constant
sigma = 5.67 * 10 ** (-8)  # W / m^2 K^4
# Stellar insolation
F_o = 1360.0  # W / m^2
# Albedo
A = 0.05
# Orbital velocity
w_orb = 2.0 * 10 ** (-7)  # rad/s
# Rotation velocity
w_rot = 2.0 * pi / (24 * 3600)  # rad/s
# Obliquity
Theta = 23.0 * pi / 180  # rad
# Ice albedo
A_ice = 0.8
# initialize geolocator
geolocator = Nominatim(user_agent="my_app")


def temp_tic(T, t, dt, phi, theta, burn_in=0.0, const_albedo=None):
    """
    Find the change in temperature at a certain location and a certain time
    :param T: [ndarray]
    The initial temperature
    :param t: [float]
    The initial time
    :param dt: [float]
    The time step
    :param phi: [ndarray]
    The location's longitude
    :param theta: [ndarray]
    The location's latitude
    :param burn_in: [float]
    Amount of time (in seconds) to wait before starting ice albedo simulation
    :param const_albedo: [float]
    A constant albedo to use instead of the ice-albedo feedback
    :return: [ndarray]
    The temperature after the time step
    """

    if const_albedo is None:
        # ice-albedo feedback
        if t < burn_in:
            A_cond = 0
        else:
            A_cond = np.where(T > 273, 0.05, 0.80)
    else:
        A_cond = const_albedo

    # Calculate the change in temperature at a given location using equation from class
    delta_T = 1 / (rho * z * cp) * (np.maximum(
        F_o * (1 - A_cond) * (np.sin(theta) * np.cos(Theta * np.cos(w_orb * t)) * np.cos(w_rot * t + phi)
                              + np.cos(theta) * np.sin(Theta * np.cos(w_orb * t))), 0.0) - sigma * T ** 4) * dt

    return T + delta_T


def plot_location(location_name, total_time=24 * 365, dest=None, dt=3600, init_temp=273.0, const_albedo=None):
    """
    Plot the temperature over time at a certain location
    :param location_name: [String]
    The name of the location or address
    :param total_time: [float]
    Time to simulate in units of dt
    :param dest: [String]
    Path and filename to save figure
    :param dt: [float]
    Time step in seconds (default 1 hour)
    :param init_temp: [float]
    The initial temperature everywhere (in Kelvin)
    :param const_albedo: [float]
    A constant albedo to use instead of the ice-albedo feedback
    :return:
    """

    # Get location
    location = geolocator.geocode(location_name)
    print("Using the location: {}".format(location.address))
    # Get coords
    theta = location.latitude * pi / 180
    phi = location.longitude * pi / 180

    #  initialize variables
    T = init_temp
    t = 0
    times = []
    temps = []

    # iteratively update temperature at location
    for i in range(total_time):
        times.append(t)
        T = temp_tic(T, t, dt, phi, theta)
        temps.append(T)
        t += dt

    # create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, temps, label=r"$\theta = $" + str(round(theta, 3)))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Temperature at the location: {}".format(location_name))

    if not dest == None:
        plt.savefig(dest)

    plt.show()


def temperature_animation(location_name=None, total_time=24 * 365, dt=3600.0, init_temp=273.0, burn_in=0, plot_burn_in=False, hw=10,
             dest=None, fps=24, const_albedo=None):
    """
    # Create an animation of the temperature distribution on a map of the earth. Specify a location to plot the temperature at that location over time.
    :param location_name: [String]
    Name of the location for the plot. If no name is specified, no plot is created.
    :param total_time: [int]
    Total time to run the simulation in units of dt
    :param dt: [float]
    Time steps in seconds.
    :param init_temp: [float]
    The initial temperature of the earth everywhere.
    :param burn_in: [int]
    Amount of time to let the model run before plotting in units of dt.
    :param plot_burn_in: [boolean]
    Show the burn in on the plot if True.
    :param hw: [int]
    The height and width of the square tiles that make up the temperature grid.
    :param dest: [String]
    The path to the destination to save the animation and the filename. Animation will not save if None.
    :param fps: [int]
    Frames per second to save the animation. One frame corresponds to one time step.
    :param const_albedo: [float]
    A constant albedo to use instead of the ice-albedo feedback
    :return:
    """
    fig = plt.figure(constrained_layout=True)

    # If a location is specified, generate a plot of the temperature at that location over time next to the map
    if not location_name is None:
        # Get location
        location = geolocator.geocode(location_name)
        print("Using the location: {}".format(location.address))
        # Get coordinates

        lat = location.latitude
        long = location.longitude
        theta = (90 - location.latitude) * pi / 180
        phi = location.longitude * pi / 180

        #  initialize variables
        T = init_temp
        t = 0
        loc_times = []
        loc_temps = []

        # iteratively update temperature at location
        for i in range(total_time + burn_in):
            loc_times.append(t)
            T = temp_tic(T, t, dt, phi, theta, burn_in * dt, const_albedo=const_albedo)
            loc_temps.append(T)
            t += dt

        # Create plot
        fig.set_size_inches(9, 3)
        gs = fig.add_gridspec(1, 6)
        f1_plot = fig.add_subplot(gs[0, 0:3])
        f1_map = fig.add_subplot(gs[0, 3:])
        if plot_burn_in:
            f1_plot.plot(loc_times, loc_temps,
                         label=r"$\theta = {} \pi$".format(str(round(theta / pi, 2))) + ", $\phi = {} \pi$".format(
                             str(round(phi / pi, 2))))
        else:
            f1_plot.plot(loc_times[burn_in:], loc_temps[burn_in:],
                         label=r"$\theta = {} \pi$".format(str(round(theta / pi, 2))) + ", $\phi = {} \pi$".format(
                             str(round(phi / pi, 2))))
        line_point, = f1_plot.plot([], [], 'ro', label="current temperature")
        f1_plot.set_xlabel("Time [s]")
        f1_plot.set_ylabel("Temperature [K]")
        f1_plot.set_title("Temperature at the location: {}".format(location_name))
        f1_plot.legend()

        f1_map.plot(long, lat, marker='X', markersize=12, c='white', mec='black')

    else:
        # create the appropriate size plot for only the map
        fig.set_size_inches(6, 3)
        gs = fig.add_gridspec(1, 2)
        f1_map = fig.add_subplot(gs[0, :])

    # plot earth land masses
    world = gpd.read_file('./Land_Mass_Shapes/World_Land.shp')
    world.plot(ax=f1_map, facecolor='none', ec='white')

    # Setting up grid
    x = np.arange(-180, 180, hw)
    y = np.arange(-90, 90, hw)
    X, Y = np.meshgrid(x, y)

    # Initializing the grid with the initial temperature
    Z = np.zeros((len(y), len(x))) + init_temp
    im = f1_map.imshow(Z, cmap=plt.cm.jet, interpolation='bilinear', extent=(-180, 180, -90, 90), vmin=0, vmax=300)

    # creating a substellar point
    sub_stellar_point, = f1_map.plot(0, 0, marker='*', markersize=12, c='white', mec='black')

    # Doing all the calculations ahead of time so the animation runs more smooth
    times = []
    map_temps = []
    map_means = []
    map_ranges = []
    sub_stel_pos = []
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    for frame in range(total_time + burn_in):
        # updating time
        t = frame * dt
        times.append(t)
        # updating the temperatures
        Z = temp_tic(Z, t, dt, (X + hw / 2) * pi / 180, (Y + hw / 2 + 90) * pi / 180, burn_in * dt, const_albedo)
        map_temps.append(Z)
        # mean temp on earth
        map_means.append(round(np.mean(Z), 1))
        # range of values for the color bar
        map_ranges.append((np.min(Z), np.max(Z) + abs((300 - np.max(Z)) / 10)))
        # updating the location of the substellar point
        sub_stel_pos.append((((w_rot * (-t)) * 180 / pi) - int(((w_rot * (-t)) * 180 / pi - 180) / 360) * 360,
                             (Theta * np.cos(w_orb * (-t))) * 180 / pi))

    # create animated plot
    def my_animate(frame):
        if not dest is None:
            print("The video is {}% done saving.".format(round((frame) / (total_time) * 100, 1)))

        # start animation after burn in
        t = times[frame]
        frame = frame + burn_in

        # update all items in the image
        im.set_array(map_temps[frame])
        im.set_clim(map_ranges[frame][0], map_ranges[frame][1])
        sub_stellar_point.set_data(sub_stel_pos[frame][0], sub_stel_pos[frame][1])

        # update figure title (171 day offset in time is for northern summer solstice)
        f1_map.set_title('Month: ' + months[int((frame*dt + 3600*24*171) // (365 / 12 * 24 * 3600))%12] +
                         ' - Total time: {} days'.format(t // (24 * 3600)) +
                         ' \n Avegare termperature: {}K '.format(str(map_means[frame])))

        # if a location is specified update the current temperature on the plot
        if not location_name == None:
            xs = loc_times[frame:frame + 1]
            ys = loc_temps[frame:frame + 1]
            line_point.set_data(xs, ys)
            f1_plot.set_title("Temperature at the location: {} \nCurrent temperature: {}K ".format(location_name,
                                                                                                   round(loc_temps[frame])))

    # create animation object
    anim = animation.FuncAnimation(fig, my_animate,
                                   frames=total_time,
                                   interval=1 / fps * 1000)

    # aesthetics such as color bars and axes
    plt.colorbar(im, label="Temperature [K]")
    f1_map.set_xlabel("Longitude")
    f1_map.set_ylabel("Latitude")
    f1_map.xaxis.set_ticks(np.linspace(-180, 180, 5))
    f1_map.yaxis.set_ticks(np.linspace(-90, 90, 5))

    # save animation if destination is specified
    if dest is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(dest)

    dest = None
    plt.show()


if __name__ == '__main__':
    temperature_animation("New York City", total_time=24 * 50, init_temp=273, burn_in=int(24 * 365 * 3),
                          plot_burn_in=False, hw=5, const_albedo=0.3)
