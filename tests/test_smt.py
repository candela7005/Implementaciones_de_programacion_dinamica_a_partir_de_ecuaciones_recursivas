"""Tests de la verificación alternativa con SMT (Z3).

Se omiten si z3 no está instalado. Comprueban que:
  - la verificación de terminación vía SMT acepta los ejemplos válidos y
    rechaza los negativos (mismo veredicto que el solver propio);
  - el código C++ generado es idéntico con y sin --smt;
  - el análisis de índices marca como seguro lo que debe, y avisa cuando una
    recurrencia lee fuera de rango (versión sin guardar de 'paths').
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import casos  # noqa: E402
from casos import EJEMPLOS, NEGATIVOS, EJEMPLOS_DIR  # noqa: E402

sys.path.insert(0, os.path.dirname(EJEMPLOS_DIR))
import codigoPD  # noqa: E402
from codigoPD import (  # noqa: E402
    validar_entrada, parser, IRBuilder, VerificadorIndices, VerificadoraSMT,
)


def _leer(*partes):
    with open(os.path.join(EJEMPLOS_DIR, *partes), encoding="utf-8") as f:
        return f.read()


@unittest.skipUnless(codigoPD.Z3_DISPONIBLE, "z3 no está instalado")
class TestTerminacionSMT(unittest.TestCase):
    def test_acepta_validos(self):
        for nombre in EJEMPLOS:
            with self.subTest(ejemplo=nombre):
                ok, _ = validar_entrada(_leer(nombre + ".dp"), usar_smt=True)
                self.assertTrue(ok, f"{nombre} debería aceptarse vía SMT")

    def test_rechaza_negativos(self):
        for nombre in NEGATIVOS:
            with self.subTest(caso=nombre):
                ok, salida = validar_entrada(_leer("fail", nombre + ".dp"), usar_smt=True)
                self.assertFalse(ok)
                self.assertIn("Terminación", salida)

    def test_codigo_identico_con_y_sin_smt(self):
        for nombre in EJEMPLOS:
            with self.subTest(ejemplo=nombre):
                _, sin = validar_entrada(_leer(nombre + ".dp"), usar_smt=False)
                _, con = validar_entrada(_leer(nombre + ".dp"), usar_smt=True)
                self.assertEqual(sin, con)


@unittest.skipUnless(codigoPD.Z3_DISPONIBLE, "z3 no está instalado")
class TestIndicesSMT(unittest.TestCase):
    """Mismo análisis de índices, pero descargando con Z3 (debe coincidir con
    el solver propio de test_compilador.TestIndices)."""

    def _avisos(self, fuente: str):
        tree = IRBuilder().transform(parser.parse(fuente))
        return VerificadorIndices(tree, VerificadoraSMT()).analizar()

    def test_ejemplos_sin_falsos_positivos(self):
        # Con Z3, igual que con el solver propio: ningún ejemplo válido avisa.
        for nombre in EJEMPLOS:
            with self.subTest(ejemplo=nombre):
                self.assertEqual(self._avisos(_leer(nombre + ".dp")), [],
                                 f"{nombre}: {self._avisos(_leer(nombre + '.dp'))}")

    def test_detecta_fuera_de_rango(self):
        # 'paths' sin guardar lee tabla[-1]; el análisis debe avisar.
        fuente = (
            "nat F, C;\n"
            "paths(1, j) = 1;\n"
            "paths(i, 1) = 1;\n"
            "paths(i, j) = paths(i-1, j) + paths(i, j-1);\n"
            "return paths(F, C);\n"
        )
        self.assertTrue(self._avisos(fuente))


@unittest.skipUnless(codigoPD.Z3_DISPONIBLE, "z3 no está instalado")
class TestPrecondiciones(unittest.TestCase):
    """Las precondiciones (`requires`) con arrays/cuantificadores solo se
    aprovechan con --smt. El cambio de monedas necesita `forall k: moneda[k]>=1`
    para que su terminación sea demostrable; además, con --smt el verificador de
    índices comprueba también los índices NO lineales (dependientes de los
    datos), lo que distingue la versión con guarda de la que no la tiene."""

    def test_monedas_necesita_smt(self):
        # Sin --smt, el solver propio no puede usar la precondición → rechaza
        # ambas por terminación (μ = v / i+v requiere moneda[k] >= 1).
        for nombre in ("coins", "ways"):
            with self.subTest(ejemplo=nombre):
                ok, salida = validar_entrada(_leer(nombre + ".dp"))
                self.assertFalse(ok, f"{nombre} debería rechazarse sin --smt")
                self.assertIn("Terminación", salida)

    def test_ways_con_smt(self):
        # La versión 2D con guarda `if moneda[i] <= v` garantiza v-moneda[i] >= 0:
        # con --smt se prueba terminación e índices, y se genera.
        ok, salida = validar_entrada(_leer("ways.dp"), usar_smt=True)
        self.assertTrue(ok, f"ways debería generarse con --smt:\n{salida}")

    def test_coins_1d_rechazada_por_indices(self):
        # La versión 1D `coins(v) = min{1<=k<=N}(coins(v-moneda[k])+1)` evalúa el
        # cuerpo para TODAS las k, sin poder filtrar las que cumplen moneda[k]<=v:
        # puede leer coins(v-moneda[k]) con índice negativo. Con --smt, el
        # verificador de índices lo refuta y la rechaza (no genera código que pete).
        ok, salida = validar_entrada(_leer("coins.dp"), usar_smt=True)
        self.assertFalse(ok, "coins 1D debería rechazarse por índices fuera de rango")
        self.assertIn("Índices fuera de rango", salida)

    def test_monedas_min_2d(self):
        # Mínimo de monedas 2D con guarda `if d[i] <= c`: el índice no lineal
        # c - d[i] es seguro y, con --smt, demostrable (terminación + índices).
        ok, salida = validar_entrada(_leer("monedas.dp"), usar_smt=True)
        self.assertTrue(ok, f"monedas debería generarse con --smt:\n{salida}")
        # Sin --smt no se prueba su terminación (μ = i+c necesita d[k] >= 1).
        ok2, _ = validar_entrada(_leer("monedas.dp"))
        self.assertFalse(ok2, "monedas debería rechazarse sin --smt (terminación)")


@unittest.skipUnless(codigoPD.Z3_DISPONIBLE, "z3 no está instalado")
@unittest.skipUnless(casos.compilador_disponible(), "no hay compilador C++")
class TestMonedasEndToEnd(unittest.TestCase):
    """El mínimo de monedas 2D, generado con --smt, compila y da el resultado
    correcto (incluido el centinela ∞ para el caso imposible)."""

    def _ejecuta(self, dvec, N, C):
        ok, cpp = validar_entrada(_leer("monedas.dp"), algoritmo="bottom-up", usar_smt=True)
        self.assertTrue(ok, cpp)
        init = "{" + ",".join(str(x) for x in dvec) + "}"
        driver = (
            "\n#include <iostream>\n"
            f"int main(){{ std::vector<int> d = {init};\n"
            f"  std::cout << monedas(d, {N}, {C}) << std::endl; return 0; }}\n"
        )
        return casos.compilar_y_ejecutar(cpp + driver).strip()

    def test_resultados(self):
        # d = {1,3,4}: 6 = 3+3 (2 monedas); 11 = 4+4+3 (3).
        self.assertEqual(self._ejecuta([0, 1, 3, 4], 3, 6), "2")
        self.assertEqual(self._ejecuta([0, 1, 3, 4], 3, 11), "3")
        # d = {2,5}, C=3: imposible → centinela.
        self.assertEqual(self._ejecuta([0, 2, 5], 2, 3), "1000000")


if __name__ == "__main__":
    unittest.main(verbosity=2)
