"""Tests end-to-end del C++ generado.

Para cada (ejemplo × estrategia × modo) genera el C++, le añade un driver de
prueba, lo compila (g++ o MSVC) y comprueba que la salida coincide con el
resultado esperado. Se omiten automáticamente si no hay compilador disponible.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casos import (  # noqa: E402
    EJEMPLOS, cpp_completo, compilador_disponible, nombre_compilador,
    compilar_y_ejecutar,
)

# (algoritmo, space_opt)
ESTRATEGIAS = [
    ("sin-memo", False),
    ("top-down", False),
    ("bottom-up", False),
    ("bottom-up", True),   # con optimización de espacio
]
MODOS = ["funcion", "clase"]


@unittest.skipUnless(compilador_disponible(),
                     "no se encontró compilador C++ (g++ o MSVC)")
class TestCppEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n[tests C++] compilador: {nombre_compilador()}")

    def test_resultados(self):
        for nombre, meta in EJEMPLOS.items():
            for algoritmo, sopt in ESTRATEGIAS:
                for modo in MODOS:
                    with self.subTest(ejemplo=nombre, algoritmo=algoritmo,
                                      space_opt=sopt, modo=modo):
                        cpp = cpp_completo(nombre, algoritmo, modo, sopt)
                        salida = compilar_y_ejecutar(cpp).strip()
                        self.assertEqual(
                            salida, str(meta["expected"]),
                            f"{nombre}/{algoritmo}/space_opt={sopt}/{modo}: "
                            f"obtenido {salida!r}, esperado {meta['expected']!r}",
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
