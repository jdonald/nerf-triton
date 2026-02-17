"""OpenGL flower mesh renderer with directional lighting."""

import ctypes

import numpy as np

from data.generate_flower import generate_flower

# Lazy imports — OpenGL may not be available in headless/test environments
_gl = None
_glu = None


def _import_gl():
    global _gl, _glu
    if _gl is None:
        import OpenGL.GL as GL
        import OpenGL.GLU as GLU
        _gl = GL
        _glu = GLU
    return _gl, _glu


class FlowerRenderer:
    """Renders the procedural flower mesh using OpenGL with directional lighting."""

    def __init__(self):
        self.vertices, self.normals, self.colors, self.faces = generate_flower()
        self._vbo_initialized = False
        self._display_list = None

    def init_gl(self):
        """Initialize OpenGL state for rendering (call after context creation)."""
        GL, GLU = _import_gl()

        # Enable features
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glEnable(GL.GL_NORMALIZE)

        # Directional light from upper-right
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [0.5, 0.8, 0.3, 0.0])  # w=0 -> directional
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])

        # Material
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)

        GL.glClearColor(0.15, 0.15, 0.2, 1.0)

        # Build display list for fast rendering
        self._build_display_list()

    def _build_display_list(self):
        """Pre-compile the mesh into an OpenGL display list."""
        GL, _ = _import_gl()

        self._display_list = GL.glGenLists(1)
        GL.glNewList(self._display_list, GL.GL_COMPILE)

        GL.glBegin(GL.GL_TRIANGLES)
        for face in self.faces:
            for idx in face:
                GL.glNormal3fv(self.normals[idx].tolist())
                GL.glColor3fv(self.colors[idx].tolist())
                GL.glVertex3fv(self.vertices[idx].tolist())
        GL.glEnd()

        GL.glEndList()

    def render(self):
        """Draw the flower mesh."""
        GL, _ = _import_gl()
        if self._display_list is not None:
            GL.glCallList(self._display_list)

    def cleanup(self):
        """Free OpenGL resources."""
        GL, _ = _import_gl()
        if self._display_list is not None:
            GL.glDeleteLists(self._display_list, 1)
            self._display_list = None
