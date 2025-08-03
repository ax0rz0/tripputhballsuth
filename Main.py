import moderngl
import numpy as np
import sounddevice as sd
from pyrr import Matrix44
import glfw, math, random, time

# ====================== SETTINGS ======================
SampleRate = 44100
Chunk = 1024
NumStrands = 10000
NumPointsPerStrand = 200
SparkThreshold = 800
SparkLifetime = 0.5

# Audio smoothing
SmoothingFactor = 0.15
BassRange = (20, 250)
MidRange = (250, 2000)
TrebleRange = (2000, 8000)
BassLevel = MidLevel = TrebleLevel = 0

# Smoothed values for visuals
SmoothBass = SmoothMid = SmoothTreble = 0
PulseSmooth = 1.0

# Camera
CameraYaw = 0.0
CameraPitch = 0.0
CameraDistance = 15.0
LastMouseX = 0
LastMouseY = 0
RightMouseDown = False
Fullscreen = True

# Sparks
Sparks = []

# ====================== AUDIO ANALYSIS ======================
def freq_to_index(freq, sample_rate, n_fft):
    return int(freq / (sample_rate / n_fft))

def band_amplitude(fft_data, band, sample_rate, n_fft):
    start_idx = freq_to_index(band[0], sample_rate, n_fft)
    end_idx = freq_to_index(band[1], sample_rate, n_fft)
    return np.mean(np.abs(fft_data[start_idx:end_idx]))

def audio_callback(indata, frames, time_data, status):
    global BassLevel, MidLevel, TrebleLevel
    audio_data = np.mean(indata, axis=1)
    fft_data = np.fft.rfft(audio_data)

    BassLevel = (1 - SmoothingFactor) * BassLevel + SmoothingFactor * band_amplitude(fft_data, BassRange, SampleRate, len(audio_data))
    MidLevel = (1 - SmoothingFactor) * MidLevel + SmoothingFactor * band_amplitude(fft_data, MidRange, SampleRate, len(audio_data))
    TrebleLevel = (1 - SmoothingFactor) * TrebleLevel + SmoothingFactor * band_amplitude(fft_data, TrebleRange, SampleRate, len(audio_data))

# ====================== SPARKS ======================
def spawn_spark():
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(-math.pi/2, math.pi/2)
    dir_x = math.cos(phi) * math.cos(theta)
    dir_y = math.sin(phi)
    dir_z = math.cos(phi) * math.sin(theta)
    color = (random.random(), random.random(), random.random())
    Sparks.append([0, 0, 0, dir_x, dir_y, dir_z, *color, time.time()])

def update_sparks():
    now = time.time()
    alive = []
    for spark in Sparks:
        if now - spark[7] < SparkLifetime:
            speed = 10
            spark[0] += spark[3] * speed * (1/120)
            spark[1] += spark[4] * speed * (1/120)
            spark[2] += spark[5] * speed * (1/120)
            alive.append(spark)
    return alive

# ====================== DEVICE SELECTION ======================
def choose_microphone():
    print("Available input devices:")
    devices = sd.query_devices()
    input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]

    for idx in input_devices:
        print(f"{idx}: {devices[idx]['name']}")

    choice = int(input("\nSelect microphone (index): "))
    device_info = sd.query_devices(choice)
    max_channels = device_info['max_input_channels']

    channels = 1 if max_channels == 1 else 2
    print(f"Using {channels} channel(s) for {device_info['name']}")
    return choice, channels

# ====================== MAIN CLASS ======================
class StarburstApp:
    def __init__(self, device_index, channels):
        global Fullscreen
        if not glfw.init():
            raise Exception("GLFW init failed")

        monitor = glfw.get_primary_monitor() if Fullscreen else None
        mode = glfw.get_video_mode(monitor)
        width, height = (mode.size.width, mode.size.height) if Fullscreen else (1200, 800)
        self.window = glfw.create_window(width, height, "ModernGL Audio Starburst", monitor, None)

        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        # Mouse & key controls
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_position_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_key_callback(self.window, self.key_callback)

        # ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # Strand shader
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_pos;
                in vec3 in_dir;
                in vec3 in_weights;
                in float in_phase;
                in float in_bias;
                in float in_thickness;

                uniform float bass;
                uniform float mid;
                uniform float treble;
                uniform mat4 mvp;

                out vec4 v_color;
                out float v_thickness;

                void main() {
                    float t = in_pos.z;

                    // Weighted contributions
                    float bassInfluence = bass * in_weights.x;
                    float midInfluence = mid * in_weights.y;
                    float trebleInfluence = treble * in_weights.z;

                    // Dynamic extension
                    float dynamic_extension = (0.5 + (bassInfluence + midInfluence + trebleInfluence) * 0.002) * in_bias;
                    float base_radius = t * (2.0 + bassInfluence * 0.3) * dynamic_extension;

                    // Curvature + vibration
                    float curve = sin(t * 4.0 + in_phase) * midInfluence * 0.3;
                    float jitter = sin(t * 20.0 + in_phase) * trebleInfluence * 0.2;

                    vec3 pos = normalize(in_dir) * base_radius + vec3(curve+jitter, curve-jitter, jitter);
                    gl_Position = mvp * vec4(pos, 1.0);

                    // Colour cycling (rainbow hue shift)
                    float cycle = in_phase + t * 0.2;
                    vec3 cycleColor = vec3(
                        0.5 + 0.5 * sin(cycle),
                        0.5 + 0.5 * sin(cycle + 2.094), // +120°
                        0.5 + 0.5 * sin(cycle + 4.188)  // +240°
                    );

                    // Frequency mapping tint
                    vec3 freqColor = vec3(
                        bassInfluence * 0.002 + 0.5,
                        midInfluence * 0.002 + 0.5,
                        trebleInfluence * 0.002 + 0.5
                    );

                    // Combine colours smoothly
                    v_color = vec4(mix(freqColor, cycleColor, 0.5), 1.0);

                    // Fake thickness (used in frag to make glow stronger)
                    v_thickness = 1.0 + in_thickness * (0.5 + (bassInfluence + midInfluence + trebleInfluence) * 0.001);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec4 v_color;
                in float v_thickness;
                out vec4 f_color;
                void main() {
                    // Simulate glow by boosting brightness with thickness
                    vec3 glowColor = v_color.rgb * v_thickness;
                    f_color = vec4(glowColor, 1.0);
                }
            '''
        )

        # Spark shader (same)
        self.spark_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_pos;
                in vec3 in_color;
                uniform mat4 mvp;
                out vec3 v_color;
                void main() {
                    gl_Position = mvp * vec4(in_pos, 1.0);
                    gl_PointSize = 6.0;
                    v_color = in_color;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_color, 1.0);
                }
            '''
        )

        # Buffers
        self.vertices, self.directions, self.weights, self.phases, self.biases, self.thicknesses = self.generate_strands()
        self.vbo_pos = self.ctx.buffer(self.vertices.astype('f4').tobytes())
        self.vbo_dir = self.ctx.buffer(self.directions.astype('f4').tobytes())
        self.vbo_weights = self.ctx.buffer(self.weights.astype('f4').tobytes())
        self.vbo_phases = self.ctx.buffer(self.phases.astype('f4').tobytes())
        self.vbo_biases = self.ctx.buffer(self.biases.astype('f4').tobytes())
        self.vbo_thickness = self.ctx.buffer(self.thicknesses.astype('f4').tobytes())

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo_pos, '3f', 'in_pos'),
                (self.vbo_dir, '3f', 'in_dir'),
                (self.vbo_weights, '3f', 'in_weights'),
                (self.vbo_phases, '1f', 'in_phase'),
                (self.vbo_biases, '1f', 'in_bias'),
                (self.vbo_thickness, '1f', 'in_thickness'),
            ]
        )

        # Start audio
        self.stream = sd.InputStream(device=device_index, callback=audio_callback,
                                     channels=channels, samplerate=SampleRate, blocksize=Chunk)
        self.stream.start()

    def generate_strands(self):
        verts, dirs, weights, phases, biases, thicknesses = [], [], [], [], [], []
        for _ in range(NumStrands):
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(-math.pi/2, math.pi/2)
            dir_x = math.cos(phi) * math.cos(theta)
            dir_y = math.sin(phi)
            dir_z = math.cos(phi) * math.sin(theta)
            direction = (dir_x, dir_y, dir_z)

            weight_bass = random.uniform(0.5, 1.0)
            weight_mid = random.uniform(0.5, 1.0)
            weight_treble = random.uniform(0.5, 1.0)
            phase_offset = random.uniform(0, 2 * math.pi)
            bias = random.uniform(0.8, 1.2)
            thickness = random.uniform(0.5, 2.0)

            for i in range(NumPointsPerStrand):
                t = i / 40.0
                verts.append((0, 0, t))
                dirs.append(direction)
                weights.append((weight_bass, weight_mid, weight_treble))
                phases.append((phase_offset,))
                biases.append((bias,))
                thicknesses.append((thickness,))
        return np.array(verts), np.array(dirs), np.array(weights), np.array(phases), np.array(biases), np.array(thicknesses)

    def render(self):
        global Sparks, CameraYaw, CameraPitch, CameraDistance
        global SmoothBass, SmoothMid, SmoothTreble, PulseSmooth

        while not glfw.window_should_close(self.window):
            if BassLevel > SparkThreshold:
                for _ in range(5):
                    spawn_spark()
            Sparks = update_sparks()

            self.ctx.clear(0.0, 0.0, 0.0)

            # Smooth audio
            smoothing = 0.1
            SmoothBass += (BassLevel - SmoothBass) * smoothing
            SmoothMid += (MidLevel - SmoothMid) * smoothing
            SmoothTreble += (TrebleLevel - SmoothTreble) * smoothing

            # Smooth bass pulse
            target_pulse = 1.0 - (BassLevel * 0.05 * 0.001)
            PulseSmooth += (target_pulse - PulseSmooth) * 0.1
            current_distance = CameraDistance * PulseSmooth

            # Camera matrix
            camX = current_distance * math.cos(CameraPitch) * math.sin(CameraYaw)
            camY = current_distance * math.sin(CameraPitch)
            camZ = current_distance * math.cos(CameraPitch) * math.cos(CameraYaw)

            proj = Matrix44.perspective_projection(60, 1200/800, 0.1, 1000.0)
            view = Matrix44.look_at((camX, camY, camZ), (0, 0, 0), (0, 1, 0))
            mvp = proj * view

            # Draw strands
            self.prog['bass'].value = SmoothBass
            self.prog['mid'].value = SmoothMid
            self.prog['treble'].value = SmoothTreble
            self.prog['mvp'].write(mvp.astype('f4').tobytes())
            self.vao.render(moderngl.LINE_STRIP)

            # Draw sparks
            if Sparks:
                spark_data = []
                for s in Sparks:
                    spark_data.append((s[0], s[1], s[2], s[6], s[7], s[8]))
                spark_array = np.array(spark_data, dtype='f4')
                vbo_sparks = self.ctx.buffer(spark_array.tobytes())
                vao_sparks = self.ctx.vertex_array(
                    self.spark_prog,
                    [(vbo_sparks, '3f 3f', 'in_pos', 'in_color')]
                )
                self.spark_prog['mvp'].write(mvp.astype('f4').tobytes())
                vao_sparks.render(moderngl.POINTS)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        self.stream.stop()
        self.stream.close()
        glfw.terminate()

    # ====================== CAMERA & KEYS ======================
    def mouse_button_callback(self, window, button, action, mods):
        global RightMouseDown, LastMouseX, LastMouseY
        if button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                RightMouseDown = True
                LastMouseX, LastMouseY = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                RightMouseDown = False

    def cursor_position_callback(self, window, xpos, ypos):
        global CameraYaw, CameraPitch, LastMouseX, LastMouseY
        if RightMouseDown:
            dx = xpos - LastMouseX
            dy = ypos - LastMouseY
            LastMouseX, LastMouseY = xpos, ypos
            CameraYaw += dx * 0.005
            CameraPitch -= dy * 0.005
            CameraPitch = max(-math.pi/2, min(math.pi/2, CameraPitch))

    def scroll_callback(self, window, xoffset, yoffset):
        global CameraDistance
        CameraDistance -= yoffset * 0.5  # infinite zoom

    def key_callback(self, window, key, scancode, action, mods):
        global Fullscreen
        if key == glfw.KEY_F11 and action == glfw.PRESS:
            Fullscreen = not Fullscreen
            monitor = glfw.get_primary_monitor() if Fullscreen else None
            mode = glfw.get_video_mode(glfw.get_primary_monitor())
            width, height = (mode.size.width, mode.size.height) if Fullscreen else (1200, 800)
            glfw.set_window_monitor(self.window, monitor, 0, 0, width, height, mode.refresh_rate if Fullscreen else 0)

# ====================== RUN ======================
if __name__ == "__main__":
    device_index, channels = choose_microphone()
    app = StarburstApp(device_index, channels)
    app.render()
