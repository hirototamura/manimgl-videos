from manim_imports_ext import *

import numpy as np
import pandas as pd
from datetime import datetime
import ppigrf
import time

EARTH_TILT_ANGLE = 23.3 * DEG

# All in Kilometers
EARTH_RADIUS = 6_371
MOON_RADIUS = 1_737.4
MOON_ORBIT_RADIUS = 384_400
SUN_RADIUS = 695_700

MERCURY_ORBIT_RADIUS = 6.805e7
VENUS_ORBIT_RADIUS = 1.082e8
EARTH_ORBIT_RADIUS = 1.473e8
MARS_ORBIT_RADIUS = 2.280e8
CERES_ORBIT_RADIUS = 4.130e8
JUPITER_ORBIT_RADIUS = 7.613e8
SATURN_ORBIT_RADIUS = 1.439e9

# In days
MERCURY_ORBIT_PERIOD = 87.97
VENUS_ORBIT_PERIOD = 224.7
EARTH_ORBIT_PERIOD = 365.25
MARS_ORBIT_PERIOD = 686.98
JUPITER_ORBIT_PERIOD = 4332.82
SATURN_ORBIT_PERIOD = 10755.7

# In km / s
SPEED_OF_LIGHT = 299792

def get_earth(radius=1.0, day_texture="EarthTextureMap", night_texture="NightEarthTextureMap"):
    sphere = Sphere(radius=radius)
    earth = TexturedSurface(sphere, day_texture, night_texture)
    return earth


def get_sphere_mesh(radius=1.0):
    sphere = Sphere(radius=radius)
    mesh = SurfaceMesh(sphere)
    mesh.set_stroke(WHITE, 0.5, 0.5)
    return mesh


def get_moon(radius=1.0, resolution=(101, 51)):
    moon = TexturedSurface(Sphere(radius=radius, resolution=resolution), "MoonTexture", "DarkMoonTexture")
    moon.set_shading(0.25, 0.25, 1)
    return moon

def get_vector_field_and_stream_lines(
    func, coordinate_system,
    # Vector config
    vector_stroke_width=5,
    vector_opacity=0.5,
    density=4,
    # Streamline config
    sample_freq=5,
    n_samples_per_line=10,
    solution_time=1,
    arc_len=3,  # Does nothing
    time_width=0.5,
    line_color=WHITE,
    line_width=3,
    line_opacity=1.0,
):
    # Vector field
    vector_field = VectorField(
        func, coordinate_system,
        density=density,
        stroke_width=vector_stroke_width,
        stroke_opacity=vector_opacity,
    )

    # Streamlines
    stream_lines = StreamLines(
        func, coordinate_system,
        density=sample_freq,
        n_samples_per_line=n_samples_per_line,
        solution_time=solution_time,
        magnitude_range=vector_field.magnitude_range,
        color_by_magnitude=False,
        stroke_color=line_color,
        stroke_width=line_width,
        stroke_opacity=line_opacity,
    )
    animated_lines = AnimatedStreamLines(
        stream_lines,
        line_anim_config=dict(time_width=time_width),
        rate_multiple=0.25,
    )

    return vector_field, animated_lines


class VectorFieldSolution(InteractiveScene):
    def construct(self):
        # Add axes
        mat = np.array([[1, 2], [3, 1]])
        # mat = np.array([[2, 0], [0, -1]])
        axes = NumberPlane((-4, 4), (-2, 2), faded_line_ratio=1)
        axes.set_height(FRAME_HEIGHT)
        axes.background_lines.set_stroke(BLUE, 1)
        axes.faded_lines.set_stroke(BLUE, 0.5, 0.5)
        axes.add_coordinate_labels(font_size=36)

        def func(v):
            return 0.5 * np.dot(v, mat.T)

        self.add(axes)

        # Calculate eigen vectors
        eigenvalues, eigenvectors = np.linalg.eig(mat)
        eigenlines = VGroup(
            Line(-v, v).set_length(10)
            for v in eigenvectors.T
        )
        eigenlines.set_stroke(TEAL, 5)

        # Show the flow
        config = dict()
        # config = dict(step_multiple=0.5, vector_stroke_width=8)
        vector_field, animated_lines = get_vector_field_and_stream_lines(func, axes, **config)

        self.add(vector_field, animated_lines)
        vector_field.set_stroke(opacity=1)
        self.play(vector_field.animate.set_stroke(opacity=0.5))
        self.wait(10)
        self.play(ShowCreation(eigenlines))
        self.wait(10)


# A scene class showing a simple rotational vector field: F(x, y) = (-y, x)
class SimpleRotationVectorField(InteractiveScene):
    def construct(self):
        # Create axes for the rotational field
        rot_axes = NumberPlane((-4, 4), (-2, 2), faded_line_ratio=1)
        rot_axes.set_height(FRAME_HEIGHT)
        rot_axes.background_lines.set_stroke(GREEN, 1)
        rot_axes.faded_lines.set_stroke(GREEN, 0.5, 0.5)
        rot_axes.add_coordinate_labels(font_size=36)

        # Define the rotational vector field: F(x, y) = (-y, x)
        # Function that can handle both individual points and arrays of points
        def rotation_field(coords):
            # Handle both individual points and arrays of points
            if coords.ndim == 1:
                # Single point: coords has shape (2,) or (3,)
                x, y = coords[0], coords[1]
                return np.array([-y, x])
            else:
                # Array of points: coords has shape (n_points, 2) or (n_points, 3)
                x = coords[:, 0]
                y = coords[:, 1]
                return np.column_stack([-y, x])

        # Create the vector field and animated stream lines
        rot_vector_field, rot_animated_lines = get_vector_field_and_stream_lines(
            rotation_field, rot_axes, vector_stroke_width=3, sample_freq=3)

        # Add to scene
        self.add(rot_axes, rot_vector_field, rot_animated_lines)
        rot_vector_field.set_stroke(opacity=1)
        self.play(rot_vector_field.animate.set_stroke(opacity=0.5))
        self.wait(10)

class ThreeDVectorField(InteractiveScene):
    def construct(self):
        # Example: 3D vector field F(x, y, z) = (-y, x, 0)
        axes3d = ThreeDAxes(
            x_range=(-4, 4, 1),
            y_range=(-4, 4, 1),
            z_range=(-2, 2, 1),
        )
        axes3d.set_height(FRAME_HEIGHT * 0.8)
        axes3d.add_axis_labels(font_size=2)

        # Define a 3D rotational vector field: F(x, y, z) = (-y, x, 0)
        def rotation_field_3d(coords):
            # coords: (n_points, 3) or (3,)
            arr = np.atleast_2d(coords)
            x = arr[:, 0]
            y = arr[:, 1]
            z = arr[:, 2]
            out = np.column_stack([-y, x, np.zeros_like(z)])
            if coords.ndim == 1:
                return out[0]
            return out

        # Create the vector field and animated stream lines in 3D
        vector_field_3d, animated_lines_3d = get_vector_field_and_stream_lines(
            rotation_field_3d, axes3d,
            vector_stroke_width=2.5,
            sample_freq=2.5,
        )

        # Position the 3D axes and field
        axes3d.move_to(ORIGIN)
        axes3d.set_opacity(0.8)
        vector_field_3d.set_stroke(opacity=1)
        animated_lines_3d.set_opacity(0.8)

        # Add to scene
        self.add(axes3d, vector_field_3d, animated_lines_3d)
        # self.move_camera(phi=70 * DEGREES, theta=30 * DEGREES)
        self.wait(2)
        self.play(vector_field_3d.animate.set_stroke(opacity=0.5))
        self.wait(10)

class ThreeDGradientVectorField(InteractiveScene):
    def construct(self):
        # Create 3D axes for the gradient field
        axes3d = ThreeDAxes(
            x_range=(-4, 4, 1),
            y_range=(-4, 4, 1),
            z_range=(-2, 2, 1),
        )
        axes3d.set_height(FRAME_HEIGHT * 0.8)
        axes3d.add_axis_labels(font_size=2)

        # Define a scalar potential function: V(x, y, z) = -1/sqrt(x^2 + y^2 + z^2 + ε)
        # This represents a gravitational potential with a small ε to avoid singularity at origin
        def potential_function(coords):
            # coords: (n_points, 3) or (3,)
            arr = np.atleast_2d(coords)
            x = arr[:, 0]
            y = arr[:, 1]
            z = arr[:, 2]
            r_squared = x**2 + y**2 + z**2
            epsilon = 0.1  # Small constant to avoid singularity
            return -1 / np.sqrt(r_squared + epsilon)

        # Define the gradient vector field: ∇V = (∂V/∂x, ∂V/∂y, ∂V/∂z)
        def gradient_field(coords):
            # coords: (n_points, 3) or (3,)
            arr = np.atleast_2d(coords)
            x = arr[:, 0]
            y = arr[:, 1]
            z = arr[:, 2]
            r_squared = x**2 + y**2 + z**2
            epsilon = 0.1
            
            # Calculate the gradient components
            # ∂V/∂x = x / (x^2 + y^2 + z^2 + ε)^(3/2)
            # ∂V/∂y = y / (x^2 + y^2 + z^2 + ε)^(3/2)
            # ∂V/∂z = z / (x^2 + y^2 + z^2 + ε)^(3/2)
            denominator = (r_squared + epsilon)**(1.5)
            
            grad_x = x / denominator
            grad_y = y / denominator
            grad_z = z / denominator
            
            out = np.column_stack([grad_x, grad_y, grad_z])
            if coords.ndim == 1:
                return out[0]
            return out

        # Create the vector field and animated stream lines in 3D
        vector_field_3d, animated_lines_3d = get_vector_field_and_stream_lines(
            gradient_field, axes3d,
            vector_stroke_width=2.0,
            sample_freq=2.0,
            density=3.0,
            solution_time=2.0,
        )

        # Position the 3D axes and field
        axes3d.move_to(ORIGIN)
        axes3d.set_opacity(0.8)
        vector_field_3d.set_stroke(opacity=1)
        animated_lines_3d.set_opacity(0.8)

        # Add title
        title = Text("Gradient Vector Field: ∇V", font_size=36)
        title.to_edge(UP)
        title.set_color(YELLOW)

        # Add explanation
        explanation = Text(
            "V(x,y,z) = -1/√(x²+y²+z²+ε)\n∇V points toward the origin",
            font_size=24
        )
        explanation.to_edge(DOWN)
        explanation.set_color(BLUE)

        # Add to scene
        self.add(axes3d, vector_field_3d, animated_lines_3d, title, explanation)
        self.wait(2)
        self.play(vector_field_3d.animate.set_stroke(opacity=0.6))
        self.wait(10)

class ThreeDElectricDipoleField(InteractiveScene):
    def construct(self):
        # Create 3D axes for the electric dipole field
        axes3d = ThreeDAxes(
            x_range=(-2, 2, 1),
            y_range=(-2, 2, 1),
            z_range=(-2, 2, 1),
        )
        axes3d.set_height(FRAME_HEIGHT * 1)
        axes3d.add_axis_labels(font_size=1)


        # Define the gradient vector field: ∇V
        def dipole_gradient_field(coords):
            # coords: (n_points, 3) or (3,)
            arr = np.atleast_2d(coords)
            x = arr[:, 0]
            y = arr[:, 1]
            z = arr[:, 2]
            
            # Dipole moment vector (pointing in z-direction)
            p_x, p_y, p_z = 0, 0, -3.0
            
            r_squared = x**2 + y**2 + z**2
            epsilon = 0.1
            
            # Calculate gradient components
            # ∇V = ∇(p·r / |r|³) = (3(p·r)r - |r|²p) / |r|⁵
            p_dot_r = p_x * x + p_y * y + p_z * z
            
            denominator = (r_squared + epsilon)**(2.5)
            
            grad_x = (3 * p_dot_r * x - (r_squared + epsilon) * p_x) / denominator
            grad_y = (3 * p_dot_r * y - (r_squared + epsilon) * p_y) / denominator
            grad_z = (3 * p_dot_r * z - (r_squared + epsilon) * p_z) / denominator
            
            out = np.column_stack([grad_x, grad_y, grad_z])
            if coords.ndim == 1:
                return out[0]
            return out

        # Create the vector field in 3D
        vector_field_3d = VectorField(
            dipole_gradient_field, axes3d,
            stroke_width=3.0,
            stroke_opacity=0.8,
            density=3.0,
        )
        # Streamlines
        stream_lines = StreamLines(
            dipole_gradient_field, 
            axes3d,
            density=1.5,
            solution_time=3.0,
            noise_factor=1.0,
            magnitude_range=vector_field_3d.magnitude_range,
            color_by_magnitude=False,
            stroke_color=WHITE,
            stroke_width=1.0,
            stroke_opacity=1.0,
        )
        animated_lines_3d = AnimatedStreamLines(
            stream_lines,
            lag_range=3,
            line_anim_config=dict(time_width=3),
            rate_multiple=0.5,
        )
        animated_lines_3d.set_opacity(0.0)

        # Position the 3D axes and field
        axes3d.move_to(ORIGIN)
        axes3d.set_opacity(0.8)

        # Add title
        title = Text("Geomagnetic Field", font_size=36)
        title.set_color(RED)
        title.to_edge(IN*5)
        title.rotate(90 * DEG, RIGHT)

        # Add explanation
        explanation = Text(
            "V(x,y,z) = p·r/|r|³\n∇V shows electric field lines",
            font_size=24
        )
        explanation.set_color(ORANGE)
        explanation.to_edge(OUT*5)
        explanation.rotate(90 * DEG, RIGHT)

        conversion_factor = 1.0 / EARTH_RADIUS * 0.7

        earth = get_earth(radius=EARTH_RADIUS * conversion_factor)
        earth.move_to(axes3d.get_origin())
        earth_axis = rotate_vector(OUT, 0, UP)
        # earth_mesh = get_sphere_mesh(radius=EARTH_RADIUS * conversion_factor)
        # earth_mesh.move_to(axes3d.get_origin())

        earth.add_updater(lambda m, dt: m.rotate(dt * 1 * DEG, axis=earth_axis))
        # earth_mesh.add_updater(lambda m, dt: m.rotate(dt * 30 * DEG, axis=earth_axis))

        # self.add(earth)
        # self.add(earth_mesh)
        # self.add(axes3d)

        self.frame.set_euler_angles(phi=90 * DEG, theta=0)
        # Add to scene
        self.add(axes3d, vector_field_3d, animated_lines_3d, title, earth)
        self.wait(2)
        self.play(vector_field_3d.animate.set_stroke(opacity=0.3), 
        animated_lines_3d.animate.set_stroke(opacity=1.0)) 
        self.wait(2)

        self.play(self.frame.animate.reorient(10, 90, 0, earth.get_center()), run_time=3)
        self.frame.add_updater(lambda m, dt: m.set_theta(m.get_theta() * (1 + 0.2 * dt)))
        self.wait(10)
        



class IGRFCoefficientManager:
    """
    Singleton class to manage IGRF coefficients loading.
    Ensures coefficients are loaded only once and cached for reuse.
    """
    _instance = None
    _coefficients_loaded = False
    _g = None
    _h = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IGRFCoefficientManager, cls).__new__(cls)
        return cls._instance
    
    def load_coefficients(self, coeff_file_path=None):
        """
        Load IGRF coefficients if not already loaded.
        
        Parameters
        ----------
        coeff_file_path : str, optional
            Path to the IGRF coefficient file. If None, uses default path.
        
        Returns
        -------
        tuple
            (g, h) where g and h are pandas DataFrames containing the Gauss coefficients
        """
        if not self._coefficients_loaded:
            if coeff_file_path is None:
                coeff_file_path = "/Users/hiroto/dev/manimglations/.mv/lib/python3.11/site-packages/ppigrf/IGRF14.shc"
            
            print("Loading IGRF coefficients (first time)...")
            self._g, self._h = ppigrf.read_shc(coeff_file_path)
            
            # Make them accessible to the ppigrf module
            ppigrf.g = self._g
            ppigrf.h = self._h
            
            self._coefficients_loaded = True
            print("IGRF coefficients loaded and cached.")
        else:
            print("Using cached IGRF coefficients.")
        
        return self._g, self._h
    
    def get_coefficients(self):
        """
        Get the currently loaded IGRF coefficients.
        
        Returns
        -------
        tuple
            (g, h) where g and h are pandas DataFrames containing the Gauss coefficients
        """
        if not self._coefficients_loaded:
            return self.load_coefficients()
        return self._g, self._h
    
    def is_loaded(self):
        """
        Check if coefficients are loaded.
        
        Returns
        -------
        bool
            True if coefficients are loaded, False otherwise
        """
        return self._coefficients_loaded

# Global instance of the coefficient manager
igrf_manager = IGRFCoefficientManager()

def load_igrf_coefficients(coeff_file_path=None):
    """
    Load IGRF coefficients using the singleton manager.
    
    Parameters
    ----------
    coeff_file_path : str, optional
        Path to the IGRF coefficient file. If None, uses default path.
    
    Returns
    -------
    tuple
        (g, h) where g and h are pandas DataFrames containing the Gauss coefficients
    """
    return igrf_manager.load_coefficients(coeff_file_path)

def get_igrf_coefficients():
    """
    Get the currently loaded IGRF coefficients.
    
    Returns
    -------
    tuple
        (g, h) where g and h are pandas DataFrames containing the Gauss coefficients
    """
    return igrf_manager.get_coefficients()

def igrf_gc(r, theta, phi, date, min_degree=1, max_degree=13):
    """
    Calculate IGRF model components

    Input and output in geocentric coordinates

    Broadcasting rules apply for coordinate arrays, and the
    combined shape will be preserved. The dates are kept out
    of the broadcasting, so that the output will have shape
    (N, ...) where N is the number of dates, and ... represents
    the combined shape of the coordinates. If you pass scalars,
    the output will be arrays of shape (1,)
    
    Parameters
    ----------
    r : array
        radius [km] of IGRF calculation
    theta : array
        colatitude [deg] of IGRF calculation
    phi : array
        longitude [deg], positive east, of IGRF claculation
    date : date(s)
        one or more dates to evaluate IGRF coefficients
    coeff_fn : string, optional
        filename of .shc file. Default is latest IGRF
    min_degree : int, optional
        lowest degree of expansion  min_degree >= 1
    max_degree : int, optional
        highest degree of expansion, 1 <= max_degree <= 13

    Return
    ------
    Br : array
        Magnetic field [nT] in radial direction
    Btheta : array
        Magnetic field [nT] in theta direction (south on an
        Earth-centered sphere with radius r)
    Bphi : array
        Magnetic field [nT] in eastward direction
    """

    # read coefficient file:
    g, h = get_igrf_coefficients()

    if not hasattr(date, '__iter__'):
        date = np.array([date])
    else:
        date = np.array(date)

    if np.any(date > g.index[-1]) or np.any(date < g.index[0]):
        print('Warning: You provided date(s) not covered by coefficient file \n({} to {})'.format(
              g.index[0].date(), g.index[-1].date()))

    if min_degree > max_degree:
        print('Warning: Highest degree of expansion must be larger or equal to lowest degree.')
        print('Reset to original range.')
        min_degree, max_degree = 1, 13

    # get coordinate arrays to same size and shape
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    shape = r.shape
    r, theta, phi = map(lambda x: x.flatten().reshape((-1 ,1)), [r, theta, phi]) # column vectors

    # make row vectors of wave numbers n and m:
    n, m = np.array([k for k in g.columns]).T
    n, m = n.reshape((1, -1)), m.reshape((1, -1))

    # TODO: It would be better to adjust the size of calcualtion arrays according to max_degree.
    # combinations = []
    # for n in range(1, max_degree + 1):
    #     for m in range(0, n + 1):
    #         combinations.append((n, m))

    # temp = pd.MultiIndex.from_tuples(combinations, names=['n', 'm'])
    # n, m = np.array([k for k in temp]).T

    # get maximum N and maximum M:
    N, M = np.max(n), np.max(m)

    # get the legendre functions
    P, dP = ppigrf.get_legendre(theta, g.keys())
    # Append coefficients at desired times (skip if index is already in coefficient data frame):
    index = g.index.union(date)

    g = g.reindex(index).groupby(index).first() # reindex and skip duplicates
    h = h.reindex(index).groupby(index).first() # reindex and skip duplicates

    # interpolate and collect the coefficients at desired times:
    g = g.interpolate(method = 'time').loc[date, :]
    h = h.interpolate(method = 'time').loc[date, :]

    # compute cosmlon and sinmlon:
    phi_rad = np.radians(phi)
    cosmphi = np.cos(phi_rad * m) # shape (n_coords x n_model_params/2)
    sinmphi = np.sin(phi_rad * m)

    # make versions of n and m that are repeated twice
    nn, mm = np.tile(n, 2), np.tile(m, 2)

    N_map = ((nn >= min_degree) & (nn <= max_degree)).astype(int)

    RE = EARTH_RADIUS
    # calculate Br:
    G  = N_map * (RE / r) ** (nn + 2) * (nn + 1) * np.hstack((P * cosmphi, P * sinmphi))
    Br = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # calculate Btheta:
    G  = - N_map * (RE / r) ** (nn + 1) * np.hstack((dP * cosmphi, dP * sinmphi)) \
         * RE / r
    Btheta = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)

    # calculate Bphi:
    G  = - N_map * (RE / r) ** (nn + 1) * mm * np.hstack((-P * sinmphi, P * cosmphi)) \
         * RE / r / np.sin(np.radians(theta))
    if np.any(np.isnan(G)):
        print(G[np.isnan(G) == True].shape)
        print(G.shape)
        print(r.shape)
    Bphi = G.dot(np.hstack((g.values, h.values)).T).T # shape (n_times, n_coords)
    # reshape and return
    outshape = tuple([Bphi.shape[0]] + list(shape))
    return Br.reshape(outshape), Btheta.reshape(outshape), Bphi.reshape(outshape)


class ThreeDIgrfField(InteractiveScene):
    def construct(self):
        # Load IGRF coefficients (will only load once, then use cached values)
        g, h = load_igrf_coefficients()
        
        # Create 3D axes for the electric dipole field
        axes3d = ThreeDAxes(
            x_range=(-2, 2, 1),
            y_range=(-2, 2, 1),
            z_range=(-2, 2, 1),
        )
        axes3d.set_height(FRAME_HEIGHT * 1)
        axes3d.add_axis_labels(font_size=1)

        load_igrf_coefficients("/Users/hiroto/dev/manimglations/.mv/lib/python3.11/site-packages/ppigrf/IGRF14.shc")
        # Define the gradient vector field: ∇V
        def igrf_gradient_field(coords):
            # coords: (n_points, 3) or (3,)
            arr = np.atleast_2d(coords)
            n_points = arr.shape[0]
            
            # Initialize output arrays with zeros
            bx = np.zeros(n_points)
            by = np.zeros(n_points)
            bz = np.zeros(n_points)
            
            # If no valid points, return zeros
            if n_points == 0:
                out = np.column_stack([bx, by, bz])
                if coords.ndim == 1:
                    return out[0]
                return out
            
            x = arr[:, 0]
            y = arr[:, 1]
            z = arr[:, 2]
            
            # Convert from scaled Cartesian coordinates to actual Earth-scale coordinates
            scale_factor = EARTH_RADIUS
            
            # Convert to geocentric coordinates (distance from Earth's center)
            x_earth = x * scale_factor
            y_earth = y * scale_factor
            z_earth = z * scale_factor
            
            # Convert Cartesian to spherical coordinates
            r = np.sqrt(x_earth**2 + y_earth**2 + z_earth**2)  # radius in km
            
            # Handle points at or near the origin to avoid division by zero
            epsilon = 1e-10
            r_safe = np.maximum(r, epsilon)
            
            # Use safe division to avoid NaN values
            z_over_r = np.divide(z_earth, r_safe, out=np.zeros_like(z_earth), where=r_safe > epsilon)
            theta = np.arccos(np.clip(z_over_r, -1.0, 1.0))  # colatitude in radians
            phi = np.arctan2(y_earth, x_earth)  # longitude in radians
            
            # Convert to degrees for ppigrf
            theta_deg = np.degrees(theta) 
            phi_deg = np.degrees(phi)
            
            # Filter out points that are exactly at the poles
            valid_points = (theta_deg != 0) & (theta_deg != 180) 
            
            # Only calculate IGRF for valid points
            if np.any(valid_points):
                try:
                    # Calculate IGRF magnetic field components for valid points only
                    date = datetime(1900, 1, 1)  # You can adjust the date as needed
                    

                    br, btheta, bphi = igrf_gc(
                        r[valid_points], 
                        theta_deg[valid_points], 
                        phi_deg[valid_points], 
                        date,
                        max_degree=2,
                    )
                    # if np.any(np.isnan(br)) or np.any(np.isnan(btheta)) or np.any(np.isnan(bphi)):
                    #     # Replace NaN values with zeros for compatibility
                    #     br = np.nan_to_num(br, nan=0.0, posinf=0.0, neginf=0.0)
                    #     btheta = np.nan_to_num(btheta, nan=0.0, posinf=0.0, neginf=0.0)
                    #     bphi = np.nan_to_num(bphi, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # ppigrf returns arrays with shape (1, n_points) for single date
                    # We need to squeeze out the time dimension to get (n_points,)
                    br = np.squeeze(br)
                    btheta = np.squeeze(btheta)
                    bphi = np.squeeze(bphi)
                    
                    # Convert spherical magnetic field components back to Cartesian
                    # Convert to Cartesian components for valid points
                    sin_theta = np.sin(theta[valid_points])
                    cos_theta = np.cos(theta[valid_points])
                    sin_phi = np.sin(phi[valid_points])
                    cos_phi = np.cos(phi[valid_points])
                    
                    # Br component contributes to all Cartesian directions
                    bx_valid = br * sin_theta * cos_phi
                    by_valid = br * sin_theta * sin_phi
                    bz_valid = br * cos_theta
                    
                    # Btheta component (southward)
                    bx_valid += btheta * cos_theta * cos_phi
                    by_valid += btheta * cos_theta * sin_phi
                    bz_valid += -btheta * sin_theta
                    
                    # Bphi component (eastward)
                    bx_valid += -bphi * sin_phi
                    by_valid += bphi * cos_phi
                    # Bphi has no z component
                    
                    # Scale back to your coordinate system
                    bx_scaled = bx_valid / scale_factor
                    by_scaled = by_valid / scale_factor
                    bz_scaled = bz_valid / scale_factor
                    
                    # Assign valid results back to the full arrays
                    bx[valid_points] = bx_scaled
                    by[valid_points] = by_scaled
                    bz[valid_points] = bz_scaled
                except Exception as e:
                    print(f"IGRF calculation error: {e}")
                    # Return zero vectors if IGRF calculation fails
                    pass
            
            # Ensure output has consistent shape
            out = np.column_stack([bx, by, bz])
            # out = np.column_stack([bx_scaled, by_scaled, bz_scaled])
            
            # Handle single point case
            if coords.ndim == 1:
                return out[0]
            return out

        # Create the vector field in 3D
        vector_field_3d = VectorField(
            igrf_gradient_field, axes3d,
            stroke_width=3.0,
            stroke_opacity=0.8,
            density=3.0,
        )
        # Streamlines
        stream_lines = StreamLines(
            igrf_gradient_field, 
            axes3d,
            density=1.5,
            solution_time=3.0,
            noise_factor=1.0,
            magnitude_range=vector_field_3d.magnitude_range,
            color_by_magnitude=False,
            stroke_color=WHITE,
            stroke_width=1.0,
            stroke_opacity=1.0,
        )
        animated_lines_3d = AnimatedStreamLines(
            stream_lines,
            lag_range=3,
            line_anim_config=dict(time_width=3),
            rate_multiple=0.5,
        )
        animated_lines_3d.set_opacity(0.0)

        # Position the 3D axes and field
        axes3d.move_to(ORIGIN)
        axes3d.set_opacity(0.8)

        # Add title
        title = Text("Electric Dipole Gradient Field", font_size=36)
        title.set_color(RED)
        title.to_edge(IN*5)
        title.rotate(90 * DEG, RIGHT)

        # Add explanation
        explanation = Text(
            "V(x,y,z) = p·r/|r|³\n∇V shows electric field lines",
            font_size=24
        )
        explanation.set_color(ORANGE)
        explanation.to_edge(OUT*5)
        explanation.rotate(90 * DEG, RIGHT)

        conversion_factor = 1.0 / EARTH_RADIUS * 0.7

        earth = get_earth(radius=EARTH_RADIUS * conversion_factor)
        earth.move_to(axes3d.get_origin())
        earth_axis = rotate_vector(OUT, 0, UP)
        # earth_mesh = get_sphere_mesh(radius=EARTH_RADIUS * conversion_factor)
        # earth_mesh.move_to(axes3d.get_origin())

        earth.add_updater(lambda m, dt: m.rotate(dt * 1 * DEG, axis=earth_axis))
        # earth_mesh.add_updater(lambda m, dt: m.rotate(dt * 30 * DEG, axis=earth_axis))

        # self.add(earth)
        # self.add(earth_mesh)
        # self.add(axes3d)

        self.frame.reorient(0, 90, 0, ORIGIN, 3.42)
        # Add to scene
        self.add(axes3d, vector_field_3d, animated_lines_3d, title, explanation, earth)
        self.wait(2)
        self.play(vector_field_3d.animate.set_stroke(opacity=0.3), 
        animated_lines_3d.animate.set_stroke(opacity=1.0)) 
        self.wait(10)

