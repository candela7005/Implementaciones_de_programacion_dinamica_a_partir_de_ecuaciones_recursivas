"""Tests de la reconstrucción de la solución óptima (--reconstruir).

Comprueban que:
  - las recurrencias de optimización con un subproblema por término generan la
    función/método de reconstrucción (secuencia de estados);
  - el producto de matrices (dos subproblemas dentro de una reducción con
    rango) se reconstruye como parentización (lista de cortes {i, j, k});
  - una recurrencia que no encaja en ninguna forma (p. ej. Fibonacci) se
    rechaza con un mensaje claro, en vez de producir un resultado incorrecto;
  - --reconstruir prevalece sobre --space-opt (necesita la tabla completa);
  - lo reconstruido reproduce el valor óptimo en los ejemplos clásicos (mochila,
    varilla, LCS, caminos y el coste del producto de matrices), en función y
    clase.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import casos  # noqa: E402
from casos import EJEMPLOS_DIR  # noqa: E402

sys.path.insert(0, os.path.dirname(EJEMPLOS_DIR))
import codigoPD  # noqa: E402
from codigoPD import validar_entrada  # noqa: E402


def _leer(*partes):
    with open(os.path.join(EJEMPLOS_DIR, *partes), encoding="utf-8") as f:
        return f.read()


class TestReconstruccionGeneracion(unittest.TestCase):
    """Generación y elegibilidad (sin compilar)."""

    def test_elegibles_generan_funcion_de_reconstruccion(self):
        for nombre in ("mochila", "lcs", "edit", "rod", "camino", "lis"):
            with self.subTest(ejemplo=nombre):
                ok, salida = validar_entrada(_leer(nombre + ".dp"), reconstruir=True)
                self.assertTrue(ok, f"{nombre}: {salida.splitlines()[0]}")
                self.assertIn("_reconstruir", salida)

    def test_secmatrices_arbol_genera_cortes(self):
        # min{i<=k<j}(sec(i,k)+sec(k+1,j)+...): dos subproblemas por término.
        # Se reconstruye como parentización: lista de cortes {i, j, k}.
        ok, salida = validar_entrada(_leer("secmatrices.dp"), reconstruir=True)
        self.assertTrue(ok, salida)
        self.assertIn("secMatrices_reconstruir", salida)
        self.assertIn("cortes", salida)

    def test_secmatrices_arbol_modo_clase(self):
        ok, salida = validar_entrada(_leer("secmatrices.dp"), reconstruir=True, modo="clase")
        self.assertTrue(ok, salida)
        self.assertIn("reconstruir()", salida)
        self.assertIn("cortes", salida)

    def test_rechaza_recurrencia_no_reconstruible(self):
        # Fibonacci suma dos subproblemas SIN reducción con rango: no es un DP
        # de optimización ni de intervalos, no hay solución que reconstruir.
        ok, salida = validar_entrada(_leer("fibonacci.dp"), reconstruir=True)
        self.assertFalse(ok, "fibonacci no debería ser reconstruible")
        self.assertIn("Reconstrucción", salida)

    def test_reconstruir_ignora_space_opt(self):
        # La reconstrucción necesita la tabla completa: prevalece sobre space-opt.
        ok, salida = validar_entrada(_leer("rod.dp"), reconstruir=True, space_opt=True)
        self.assertTrue(ok, salida)
        self.assertIn("_reconstruir", salida)

    def test_modo_clase_emite_metodo(self):
        ok, salida = validar_entrada(_leer("mochila.dp"), reconstruir=True, modo="clase")
        self.assertTrue(ok, salida)
        self.assertIn("reconstruir()", salida)

    @unittest.skipUnless(codigoPD.Z3_DISPONIBLE, "z3 no está instalado")
    def test_monedas_con_smt(self):
        ok, salida = validar_entrada(_leer("monedas.dp"), reconstruir=True, usar_smt=True)
        self.assertTrue(ok, salida)
        self.assertIn("_reconstruir", salida)


@unittest.skipUnless(casos.compilador_disponible(), "no hay compilador C++")
class TestReconstruccionEndToEnd(unittest.TestCase):
    """El camino reconstruido reproduce el valor óptimo (compila y ejecuta)."""

    def _ejecutar(self, nombre, driver, modo="funcion"):
        ok, cpp = validar_entrada(_leer(nombre + ".dp"), reconstruir=True, modo=modo)
        self.assertTrue(ok, cpp)
        return casos.compilar_y_ejecutar(cpp + driver).strip()

    def test_mochila_funcion(self):
        # Suma de v[i] en los pasos donde la capacidad baja (objeto tomado) = valor.
        driver = (
            "\n#include <iostream>\n"
            "int main(){ vector<int> v={0,1,4,5,7}, w={0,1,3,4,5};\n"
            "  auto p = mochila_reconstruir(v, w, 4, 7); int r = 0;\n"
            "  for (size_t k=0;k+1<p.size();k++) if (p[k+1][1]<p[k][1]) r += v[p[k][0]];\n"
            "  std::cout << r << std::endl; return 0; }\n"
        )
        self.assertEqual(self._ejecutar("mochila", driver), "9")

    def test_mochila_clase(self):
        driver = (
            "\n#include <iostream>\n"
            "int main(){ mochila obj(4, 7, {0,1,4,5,7}, {0,1,3,4,5});\n"
            "  auto p = obj.reconstruir(); vector<int> v={0,1,4,5,7}; int r=0;\n"
            "  for (size_t k=0;k+1<p.size();k++) if (p[k+1][1]<p[k][1]) r += v[p[k][0]];\n"
            "  std::cout << r << std::endl; return 0; }\n"
        )
        self.assertEqual(self._ejecutar("mochila", driver, modo="clase"), "9")

    def test_rod(self):
        # Suma de price[corte] (corte = caída de n) = beneficio óptimo.
        driver = (
            "\n#include <iostream>\n"
            "int main(){ vector<int> price={0,1,5,8,9};\n"
            "  auto p = rod_reconstruir(price, 4); int r = 0;\n"
            "  for (size_t k=0;k+1<p.size();k++) r += price[p[k][0]-p[k+1][0]];\n"
            "  std::cout << r << std::endl; return 0; }\n"
        )
        self.assertEqual(self._ejecutar("rod", driver), "10")

    def test_lcs(self):
        # Pasos diagonales (i y j bajan) = caracteres emparejados = longitud LCS.
        driver = (
            "\n#include <iostream>\n"
            "int main(){ vector<char> A={'#','A','G','G','T','A','B'},\n"
            "                          B={'#','G','X','T','X','A','Y','B'};\n"
            "  auto p = LCS_reconstruir(A, B, 6, 7); int d = 0;\n"
            "  for (size_t k=0;k+1<p.size();k++)\n"
            "    if (p[k+1][0]==p[k][0]-1 && p[k+1][1]==p[k][1]-1) d++;\n"
            "  std::cout << d << std::endl; return 0; }\n"
        )
        self.assertEqual(self._ejecutar("lcs", driver), "4")

    def test_camino(self):
        # camino(i,j) = coste[i][j] + min(...): el valor es la suma de coste
        # a lo largo del camino reconstruido.
        driver = (
            "\n#include <iostream>\n"
            "int main(){ vector<vector<int>> coste={{0,0,0,0},{0,1,3,1},{0,1,5,1},{0,4,2,1}};\n"
            "  auto p = camino_reconstruir(coste, 3, 3); int s = 0;\n"
            "  for (auto& c : p) s += coste[c[0]][c[1]];\n"
            "  std::cout << s << std::endl; return 0; }\n"
        )
        self.assertEqual(self._ejecutar("camino", driver), "7")

    def test_secmatrices_parentizacion(self):
        # La parentización óptima (cortes {i, j, k}) reproduce el coste mínimo
        # del producto de matrices: recomputa el coste recorriendo el árbol.
        driver = (
            "\n#include <iostream>\n"
            "int main(){ vector<int> d = {0,10,20,30,40};\n"
            "  auto cortes = secMatrices_reconstruir(d, 3);\n"
            "  function<int(int,int)> coste = [&](int i, int j) -> int {\n"
            "    if (i == j) return 0;\n"
            "    for (auto& c : cortes) if (c[0]==i && c[1]==j)\n"
            "      return coste(i,c[2]) + coste(c[2]+1,j) + d[i]*d[c[2]+1]*d[j+1];\n"
            "    return -1; };\n"
            "  std::cout << coste(1, 3) << std::endl; return 0; }\n"
        )
        self.assertEqual(self._ejecutar("secmatrices", driver), "18000")

    def test_lis(self):
        # El camino reconstruido es la subsecuencia; su longitud = LIS hasta N.
        # Ejercita la reconstrucción de una reducción CON filtro (a[k] < a[i]).
        driver = (
            "\n#include <iostream>\n"
            "int main(){ vector<int> a = {0,1,3,2,4};\n"
            "  auto p = LIS_reconstruir(a, 4);\n"
            "  std::cout << p.size() << std::endl; return 0; }\n"
        )
        self.assertEqual(self._ejecutar("lis", driver), "3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
