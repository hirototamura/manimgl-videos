from manim_imports_ext import *

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


def get_sun(
    radius=1.0,
    near_glow_ratio=2.0,
    near_glow_factor=2,
    big_glow_ratio=4,
    big_glow_factor=1,
    big_glow_opacity=0.35,
):
    sun = TexturedSurface(Sphere(radius=radius), "SunTexture")
    sun.set_shading(0, 0, 0)
    sun.to_edge(LEFT)

    # Glows
    near_glow = GlowDot(radius=near_glow_ratio * radius, glow_factor=near_glow_factor)
    near_glow.move_to(sun)

    big_glow = GlowDot(radius=big_glow_ratio * radius, glow_factor=big_glow_factor, opacity=big_glow_opacity)
    big_glow.move_to(sun)

    return Group(sun, near_glow, big_glow)


def get_planet(name, radius=1.0):
    planet = TexturedSurface(Sphere(radius=radius), f"{name}Texture", f"Dark{name}Texture")
    planet.set_shading(0.25, 0.25, 1)
    return planet


def get_celestial_sphere(radius=1000, constellation_opacity=0.1):
    sphere = Group(
        TexturedSurface(Sphere(radius=radius, clockwise=True), "hiptyc_2020_8k"),
        TexturedSurface(Sphere(radius=0.99 * radius, clockwise=True), "constellation_figures_8k"),
    )
    sphere.set_shading(0, 0, 0)
    sphere[1].set_opacity(constellation_opacity)

    sphere.rotate(EARTH_TILT_ANGLE, RIGHT)

    return sphere


def get_planet_symbols(text, font_size=48):
    return Tex(
        Rf"\{text}",
        additional_preamble=R"\usepackage{wasysym}",
        font_size=font_size,
    )


class PerspectivesOnEarth(InteractiveScene):
    def construct(self):
        # Ask about size
        light = self.camera.light_source
        light.move_to(50 * LEFT)
        frame = self.frame
        frame.set_field_of_view(25 * DEG)

        conversion_factor = 1.0 / EARTH_RADIUS

        earth = get_earth(radius=EARTH_RADIUS * conversion_factor)
        earth.rotate(-EARTH_TILT_ANGLE, UP)
        earth_axis = rotate_vector(OUT, -EARTH_TILT_ANGLE, UP)

        earth.add_updater(lambda m, dt: m.rotate(dt * 10 * DEG, axis=earth_axis))

        self.add(earth)

        # Clearly show the size of the earth
        brace = Brace(earth, LEFT)
        brace.stretch(0.5, 1, about_edge=UP)
        label = brace.get_tex(Rf"R_E", font_size=24)
        VGroup(brace, label).rotate(90 * DEG, RIGHT, about_point=brace.get_bottom())

        dashed_lines = VGroup(
            DashedLine(brace.get_corner(OUT + RIGHT), earth.get_zenith(), dash_length=0.02),
            DashedLine(brace.get_corner(IN + RIGHT), earth.get_center(), dash_length=0.02),
        )
        dashed_lines.set_stroke(WHITE, 2)

        frame.reorient(0, 90, 0, ORIGIN, 3.42)

        self.play(
            GrowFromCenter(brace),
            Write(label),
            *map(ShowCreation, dashed_lines),
        )
        self.wait(4)
        self.play(LaggedStartMap(FadeOut, VGroup(label, brace, *dashed_lines), shift=IN))

        # Make it wobble a bit
        earth.suspend_updating()

        def wave(x):
            return math.sin(2 * x) * np.exp(-x * x)

        def homotopy(x, y, z, t):
            t = 3 * (2 * t - 1)
            return (x + 0.1 * wave(t - z), y, z + 0.1 * wave(t - x))

        self.play(Homotopy(homotopy, earth, run_time=5))

        # Turn it into a disk
        earth.save_state()
        flat_earth = TexturedSurface(
            ParametricSurface(
                lambda u, v: ((1 - v) * math.cos(u), (1 - v) * math.sin(u), 0),
                u_range=(0, TAU),
                v_range=(0, 1),
                resolution=earth.resolution
            ),
            "EarthTextureMap",
            "NightEarthTextureMap",
        )

        rot = rotation_matrix_transpose(40 * DEG, (-1, 0, -0.5)).T
        flat_earth = earth.copy().apply_matrix(rot).stretch(1e-2, 2)
        flat_earth.data["d_normal_point"] = flat_earth.get_points() + 1e-3 * OUT

        self.play(
            Transform(earth, flat_earth, run_time=4),
            frame.animate.reorient(0, 74, 0, (-0.03, -0.14, 0.04), 3.42),
            light.animate.shift(10 * OUT + 10 * LEFT),
            run_time=2
        )
        self.play(Rotate(earth, PI / 3, axis=RIGHT, run_time=3, rate_func=wiggle))
        self.wait()

        # Zoom out to the moon
        orbit = Circle(radius=MOON_ORBIT_RADIUS * conversion_factor)
        orbit.set_stroke(GREY_C, width=(0, 3))
        orbit.rotate(-45 * DEG)
        orbit.add_updater(lambda m, dt: m.rotate(5 * dt * DEG))

        moon = get_moon(radius=MOON_RADIUS * conversion_factor)
        moon.to_edge(RIGHT)
        moon.add_updater(lambda m: m.move_to(orbit.get_start()))

        self.add(orbit, moon)
        self.play(
            Restore(earth, time_span=(0, 3)),
            frame.animate.reorient(0, 0, 0, ORIGIN, 1.1 * orbit.get_height()),
            run_time=4,
        )

        interp_factor = ValueTracker(0)
        frame.add_updater(lambda m: m.move_to(interpolate(m.get_center(), moon.get_center(), interp_factor.get_value())))
        self.play(
            frame.animate.reorient(-67, 60, 0,).set_height(3),
            interp_factor.animate.set_value(0.2),
            run_time=8
        )
        self.wait(4)

