"""Main interactive viewer application.

Controls:
    Mouse: Look around
    W/A/S/D: Move forward/left/backward/right
    +/-: Zoom in/out
    R: Toggle recording
    ESC: Quit

Headless mode (--headless):
    Simulates a camera path without opening a window, useful for testing.
"""

import argparse
import math
import sys
import time

import numpy as np

from viewer.camera import FPSCamera
from viewer.recorder import PathRecorder


def run_headless(num_frames: int = 60, output_path: str = None) -> list[np.ndarray]:
    """Run a simulated camera path without a display.

    Orbits around the flower, capturing camera poses. Useful for testing
    the recording pipeline without requiring a GUI.

    Args:
        num_frames: Number of frames to simulate
        output_path: If set, save recorded path to this JSON file

    Returns:
        List of (4, 4) camera-to-world matrices
    """
    camera = FPSCamera(position=np.array([0.0, 0.5, 3.0]))
    recorder = PathRecorder()
    recorder.start()

    dt = 1.0 / 60.0
    poses = []

    for i in range(num_frames):
        # Orbit around the scene
        angle = 2 * math.pi * i / num_frames
        camera.yaw = -90.0 + angle * 180.0 / math.pi
        camera.pitch = -10.0
        camera._update_vectors()

        radius = 3.0
        camera.position = np.array([
            radius * math.cos(math.radians(camera.yaw + 90)),
            0.5,
            radius * math.sin(math.radians(camera.yaw + 90)),
        ])

        c2w = camera.get_c2w_matrix()
        recorder.capture(c2w, camera.fov)
        poses.append(c2w.copy())

    recorder.stop()

    if output_path:
        recorder.save(output_path)

    return poses


def run_viewer():
    """Launch the interactive OpenGL viewer."""
    try:
        import pygame
        from pygame.locals import (
            DOUBLEBUF,
            HWSURFACE,
            KEYDOWN,
            KEYUP,
            MOUSEMOTION,
            OPENGL,
            QUIT,
            K_ESCAPE,
            K_PLUS,
            K_MINUS,
            K_EQUALS,
            K_KP_PLUS,
            K_KP_MINUS,
            K_a,
            K_d,
            K_r,
            K_s,
            K_w,
        )
        import OpenGL.GL as GL
        import OpenGL.GLU as GLU
    except ImportError as e:
        print(f"Error: {e}")
        print("Install pygame and PyOpenGL: pip install pygame PyOpenGL")
        sys.exit(1)

    from viewer.flower_mesh import FlowerRenderer

    # Initialize pygame + OpenGL
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | HWSURFACE)
    pygame.display.set_caption("NeRF Flower Viewer - WASD/Mouse/+/- | R=Record | ESC=Quit")

    # Set up projection
    GL.glViewport(0, 0, width, height)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()

    camera = FPSCamera(position=np.array([0.0, 0.5, 3.0]))
    GLU.gluPerspective(camera.fov, width / height, 0.1, 100.0)

    GL.glMatrixMode(GL.GL_MODELVIEW)

    # Initialize renderer
    flower = FlowerRenderer()
    flower.init_gl()

    recorder = PathRecorder()

    # Capture mouse
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.set_pos(width // 2, height // 2)

    clock = pygame.time.Clock()
    keys_pressed = {"w": False, "a": False, "s": False, "d": False}
    running = True

    # HUD font
    try:
        pygame.font.init()
        font = pygame.font.SysFont("monospace", 16)
    except Exception:
        font = None

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_w:
                    keys_pressed["w"] = True
                elif event.key == K_a:
                    keys_pressed["a"] = True
                elif event.key == K_s:
                    keys_pressed["s"] = True
                elif event.key == K_d:
                    keys_pressed["d"] = True
                elif event.key in (K_PLUS, K_EQUALS, K_KP_PLUS):
                    camera.zoom(2.0)
                elif event.key in (K_MINUS, K_KP_MINUS):
                    camera.zoom(-2.0)
                elif event.key == K_r:
                    is_recording = recorder.toggle()
                    state = "STARTED" if is_recording else f"STOPPED ({recorder.frame_count} frames)"
                    print(f"Recording {state}")
                    if not is_recording and recorder.frame_count > 0:
                        recorder.save("camera_path.json")
                        print("Saved camera_path.json")
            elif event.type == KEYUP:
                if event.key == K_w:
                    keys_pressed["w"] = False
                elif event.key == K_a:
                    keys_pressed["a"] = False
                elif event.key == K_s:
                    keys_pressed["s"] = False
                elif event.key == K_d:
                    keys_pressed["d"] = False
            elif event.type == MOUSEMOTION:
                dx, dy = event.rel
                camera.process_mouse(dx, dy)

        # Update camera
        camera.process_keyboard(keys_pressed, dt)

        # Record frame if recording
        if recorder.recording:
            recorder.capture(camera.get_c2w_matrix(), camera.fov)

        # Update projection for zoom
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(camera.fov, width / height, 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

        # Render
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()

        # Apply view matrix
        view = camera.get_view_matrix()
        GL.glMultMatrixf(view.T.astype(np.float32).flatten())

        flower.render()

        pygame.display.flip()

    # Cleanup
    flower.cleanup()
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="NeRF Flower Viewer")
    parser.add_argument("--headless", action="store_true", help="Run without display")
    parser.add_argument("--num_frames", type=int, default=60, help="Frames for headless mode")
    parser.add_argument("--output", default="camera_path.json", help="Output path file")
    args = parser.parse_args()

    if args.headless:
        poses = run_headless(args.num_frames, args.output)
        print(f"Headless: captured {len(poses)} frames -> {args.output}")
    else:
        run_viewer()


if __name__ == "__main__":
    main()
