"""Tests del compilador (sin C++): validan el front-end y el análisis.

Comprueban que:
  - cada ejemplo válido en ejemplos/ pasa sintaxis + semántica + terminación y
    produce código;
  - cada caso de ejemplos/fail/ es rechazado con un error de terminación.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casos import EJEMPLOS, NEGATIVOS, EJEMPLOS_DIR  # noqa: E402

sys.path.insert(0, os.path.dirname(EJEMPLOS_DIR))
from codigoPD import validar_entrada, parser, IRBuilder, VerificadorIndices  # noqa: E402


def _leer(*partes):
    with open(os.path.join(EJEMPLOS_DIR, *partes), encoding="utf-8") as f:
        return f.read()


class TestEjemplosValidos(unittest.TestCase):
    def test_generan_codigo(self):
        for nombre in EJEMPLOS:
            with self.subTest(ejemplo=nombre):
                ok, salida = validar_entrada(_leer(nombre + ".dp"))
                self.assertTrue(ok, f"{nombre} debería ser válido, pero falló:\n{salida}")
                self.assertIn("int", salida)  # algo de C++ se generó

    def test_todas_las_estrategias_generan(self):
        for nombre in EJEMPLOS:
            for algoritmo in ("sin-memo", "top-down", "bottom-up"):
                for modo in ("funcion", "clase"):
                    sopt = algoritmo == "bottom-up"
                    with self.subTest(ejemplo=nombre, algoritmo=algoritmo, modo=modo, space_opt=sopt):
                        ok, salida = validar_entrada(
                            _leer(nombre + ".dp"),
                            modo=modo, algoritmo=algoritmo, space_opt=sopt,
                        )
                        self.assertTrue(ok, f"{nombre}/{algoritmo}/{modo}:\n{salida}")


class TestCasosNegativos(unittest.TestCase):
    def test_terminacion_rechazada(self):
        for nombre in NEGATIVOS:
            with self.subTest(caso=nombre):
                ok, salida = validar_entrada(_leer("fail", nombre + ".dp"))
                self.assertFalse(ok, f"{nombre} debería ser rechazado")
                self.assertIn("Terminación", salida)

    def test_variable_no_declarada(self):
        fuente = "nat N; f(0) = 0; f(n) = f(n-1) + xx; return f(N);"
        ok, salida = validar_entrada(fuente)
        self.assertFalse(ok)
        self.assertIn("no declarada", salida)

    def test_error_sintactico(self):
        ok, salida = validar_entrada("nat N; f(0) = ; return f(N);")
        self.assertFalse(ok)
        self.assertIn("Sintaxis", salida)


class TestIndices(unittest.TestCase):
    """Verificación de índices fuera de rango con el SOLVER PROPIO (sin SMT).

    Gracias a la inferencia de cotas inferiores, NO hay falsos positivos: todos
    los ejemplos válidos quedan limpios y las recurrencias mal definidas se
    detectan (y, en validar_entrada, detienen la generación)."""

    def _avisos(self, fuente):
        tree = IRBuilder().transform(parser.parse(fuente))
        return VerificadorIndices(tree).analizar()

    def test_ejemplos_sin_falsos_positivos(self):
        # TODOS los ejemplos válidos deben quedar limpios (sin falsos positivos).
        for nombre in EJEMPLOS:
            with self.subTest(ejemplo=nombre):
                self.assertEqual(self._avisos(_leer(nombre + ".dp")), [],
                                 f"{nombre} no debería dar avisos de índice")

    def test_detecta_indice_negativo(self):
        # 'paths' sin guardar lee tabla[-1]; debe detectarse.
        paths_malo = (
            "nat F, C;\n"
            "paths(1, j) = 1;\n"
            "paths(i, 1) = 1;\n"
            "paths(i, j) = paths(i-1, j) + paths(i, j-1);\n"
            "return paths(F, C);\n"
        )
        self.assertTrue(self._avisos(paths_malo))

    def test_detecta_caso_base_faltante(self):
        # Fibonacci con un solo caso base lee fib(-1); debe detectarse.
        fib_malo = "nat N;\nfib(0)=0;\nfib(n)=fib(n-1)+fib(n-2);\nreturn fib(N);\n"
        self.assertTrue(self._avisos(fib_malo))

    def test_indices_detienen_generacion(self):
        # En validar_entrada, una violación de índices es un error bloqueante.
        fib_malo = "nat N;\nfib(0)=0;\nfib(n)=fib(n-1)+fib(n-2);\nreturn fib(N);\n"
        ok, salida = validar_entrada(fib_malo)
        self.assertFalse(ok)
        self.assertIn("Índices fuera de rango", salida)


if __name__ == "__main__":
    unittest.main(verbosity=2)
