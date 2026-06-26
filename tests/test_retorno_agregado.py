"""Tests del RETORNO AGREGADO: `return max/min{...}( f(...) )`.

A diferencia de un retorno en una celda concreta `f(args)`, un retorno agregado
recorre celdas y devuelve su máximo/mínimo. El caso de referencia es la LIS
GLOBAL (`ejemplos/lis_global.dp`): el máximo de LIS(i) sobre todo i, frente a la
LIS que termina en N (`lis.dp`).

Se comprueba que:
  - top-down y bottom-up (función y clase) generan, compilan y dan la LIS global
    (3 para a = [1,2,3,1], mientras que LIS(N) sería 1: el agregado SÍ se nota);
  - --space-opt cae al bottom-up con tabla completa (la agregación necesita
    todas las celdas);
  - sin-memo y --reconstruir se rechazan con un mensaje claro (aún no soportados).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import casos  # noqa: E402
from casos import EJEMPLOS_DIR  # noqa: E402

sys.path.insert(0, os.path.dirname(EJEMPLOS_DIR))
from codigoPD import validar_entrada  # noqa: E402


def _leer(*partes):
    with open(os.path.join(EJEMPLOS_DIR, *partes), encoding="utf-8") as f:
        return f.read()


class TestRetornoAgregadoGeneracion(unittest.TestCase):
    def test_acepta_y_agrega_en_la_entrada(self):
        # El punto de entrada agrega sobre las celdas (no devuelve una sola).
        td = validar_entrada(_leer("lis_global.dp"), algoritmo="top-down")[1]
        self.assertIn("res_local = max(res_local", td)
        bu = validar_entrada(_leer("lis_global.dp"), algoritmo="bottom-up")[1]
        self.assertIn("res_local = max(res_local", bu)

    def test_sin_memo_rechazado(self):
        ok, salida = validar_entrada(_leer("lis_global.dp"), algoritmo="sin-memo")
        self.assertFalse(ok)
        self.assertIn("No soportado", salida)
        self.assertIn("sin-memo", salida)

    def test_reconstruir_rechazado(self):
        ok, salida = validar_entrada(_leer("lis_global.dp"), reconstruir=True)
        self.assertFalse(ok)
        self.assertIn("No soportado", salida)

    def test_space_opt_cae_a_tabla_completa(self):
        ok, salida = validar_entrada(_leer("lis_global.dp"),
                                     algoritmo="bottom-up", space_opt=True)
        self.assertTrue(ok, salida)
        self.assertIn("tabla completa", salida)


@unittest.skipUnless(casos.compilador_disponible(), "no hay compilador C++")
class TestRetornoAgregadoEndToEnd(unittest.TestCase):
    """La LIS global de a = [1,2,3,1] es 3 (termina en la posición 3), mientras
    que LIS(N) = LIS(4) = 1: el retorno agregado da el valor correcto."""

    DATOS = "vector<int> a = {0,1,2,3,1};"

    def _funcion(self, algoritmo):
        ok, cpp = validar_entrada(_leer("lis_global.dp"), algoritmo=algoritmo)
        self.assertTrue(ok, cpp)
        driver = ("\n#include <iostream>\n"
                  f"int main(){{ {self.DATOS}\n"
                  "  std::cout << LIS(a, 4) << std::endl; return 0; }\n")
        return casos.compilar_y_ejecutar(cpp + driver).strip()

    def _clase(self, algoritmo):
        ok, cpp = validar_entrada(_leer("lis_global.dp"), algoritmo=algoritmo, modo="clase")
        self.assertTrue(ok, cpp)
        driver = ("\n#include <iostream>\n"
                  "int main(){ LIS obj(4, {0,1,2,3,1});\n"
                  "  std::cout << obj() << std::endl; return 0; }\n")
        return casos.compilar_y_ejecutar(cpp + driver).strip()

    def test_top_down(self):
        self.assertEqual(self._funcion("top-down"), "3")
        self.assertEqual(self._clase("top-down"), "3")

    def test_bottom_up(self):
        self.assertEqual(self._funcion("bottom-up"), "3")
        self.assertEqual(self._clase("bottom-up"), "3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
