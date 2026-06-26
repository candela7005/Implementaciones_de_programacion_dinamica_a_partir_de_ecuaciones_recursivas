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
    """Las precondiciones (`requires`) con arrays/cuantificadores las aprovecha
    Z3. Como el solver de cota afín NO puede gestionarlas, una precondición no
    lineal (cuantificada o con arrays, p. ej. `forall k: moneda[k]>=1`) se
    descarga en Z3 AUTOMÁTICAMENTE, sin necesidad de --smt. El flag --smt sigue
    forzando la vía Z3 para todo."""

    def test_precondicion_no_lineal_va_a_smt_sin_flag(self):
        # forall k: d[k] >= 1 es no lineal -> su terminación se descarga en Z3
        # automáticamente. ways y monedas (2D, con guarda) se aceptan SIN --smt.
        for nombre in ("ways", "monedas"):
            with self.subTest(ejemplo=nombre):
                ok, salida = validar_entrada(_leer(nombre + ".dp"))
                self.assertTrue(ok, f"{nombre} debería aceptarse sin --smt:\n{salida}")

    def test_ways_con_smt(self):
        # La versión 2D con guarda `if moneda[i] <= v` garantiza v-moneda[i] >= 0:
        # con --smt se prueba terminación e índices, y se genera.
        ok, salida = validar_entrada(_leer("ways.dp"), usar_smt=True)
        self.assertTrue(ok, f"ways debería generarse con --smt:\n{salida}")

    def test_coins_1d_sin_filtro_rechazada_por_indices(self):
        # La versión 1D SIN filtro `coins(v) = min{1<=k<=N}(coins(v-moneda[k])+1)`
        # evalúa el cuerpo para TODAS las k, sin descartar las que incumplen
        # moneda[k]<=v: puede leer coins(v-moneda[k]) con índice negativo. Su
        # terminación se prueba sola (Z3 con la precondición no lineal), pero el
        # verificador de índices la refuta y la rechaza. Fuente embebida para no
        # depender de si ejemplos/coins.dp lleva o no el filtro.
        fuente = (
            "nat N, V;\n"
            "array<nat> moneda;\n"
            "requires forall k: moneda[k] >= 1;\n"
            "coins(0) = 0;\n"
            "coins(v) = min{1 <= k <= N}( coins(v - moneda[k]) + 1 ) if v > 0;\n"
            "return coins(V);\n"
        )
        ok, salida = validar_entrada(fuente)  # sin --smt: termina vía Z3, falla índices
        self.assertFalse(ok, "coins 1D sin filtro debería rechazarse por índices")
        self.assertIn("Índices fuera de rango", salida)

    def test_monedas_min_2d(self):
        # Mínimo de monedas 2D con guarda `if d[i] <= c`: el índice no lineal
        # c - d[i] es seguro y demostrable (terminación + índices).
        ok, salida = validar_entrada(_leer("monedas.dp"), usar_smt=True)
        self.assertTrue(ok, f"monedas debería generarse con --smt:\n{salida}")
        # Y también SIN --smt: la precondición no lineal forall d[k]>=1 se
        # descarga en Z3 automáticamente, así que ya no hace falta el flag.
        ok2, salida2 = validar_entrada(_leer("monedas.dp"))
        self.assertTrue(ok2, f"monedas debería aceptarse sin --smt:\n{salida2}")

    def test_filtro_acota_indice_no_lineal(self):
        # El filtro de una reducción se pasa EN CRUDO a Z3 (como la guarda y las
        # precondiciones), de modo que su parte no afín sirve para acotar índices.
        # Aquí g termina con μ = i (lineal, sin el filtro), pero el 2.º índice de
        # la llamada recursiva es a[k] (no lineal): su cota a[k] <= M se sigue
        # SOLO del filtro a[k] <= j con el invariante j <= M. Lo único que cambia
        # entre las dos fuentes es el filtro, así que aislamos su efecto.
        con_filtro = (
            "nat N, M;\n"
            "array<nat> a;\n"
            "g(0, j) = 0;\n"
            "g(i, j) = max{1 <= k <= M : a[k] <= j}( g(i - 1, a[k]) ) if i > 0;\n"
            "return g(N, M);\n"
        )
        sin_filtro = con_filtro.replace(" : a[k] <= j", "")
        ok_con, salida = validar_entrada(con_filtro, usar_smt=True)
        self.assertTrue(ok_con, f"con filtro debería aceptarse:\n{salida}")
        ok_sin, salida2 = validar_entrada(sin_filtro, usar_smt=True)
        self.assertFalse(ok_sin, "sin filtro a[k] no está acotado: debe rechazarse")
        self.assertIn("Índices fuera de rango", salida2)


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
