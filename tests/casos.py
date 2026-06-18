"""Metadatos y utilidades compartidas por la suite de tests.

Contiene:
  - EJEMPLOS: por cada problema, los datos de prueba en C++, la llamada inicial
    y el resultado esperado, para los tests end-to-end del código generado.
  - construir_driver(): arma un main() de prueba adaptado a (estrategia, modo).
  - utilidades para localizar un compilador C++ (g++ o MSVC) y compilar+ejecutar.

Diseñado para depender solo de la biblioteca estándar (sin pytest): se ejecuta
con `python -m unittest discover -s tests` o con `python run_tests.py`.
"""

import os
import sys
import glob
import shutil
import tempfile
import subprocess

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
EJEMPLOS_DIR = os.path.join(RAIZ, "ejemplos")

# Para poder importar codigoPD desde los tests.
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)


# ---------------------------------------------------------------------------
# Casos de prueba
# ---------------------------------------------------------------------------
# Convención de índices 1-based: los arrays se proveen con la posición 0 como
# centinela (salvo `d` del producto de matrices, donde d[0] es significativo).
#
# Claves por ejemplo:
#   func      : nombre de la función/clase generada
#   decls     : código C++ que declara los datos de entrada
#   data      : nombres de los arrays que recibe la función (orden de firma)
#   scalars   : valores de los escalares declarados (para la firma pública)
#   initial   : argumentos de la llamada inicial canónica de la recurrencia
#   ctor      : argumentos del constructor en modo clase (todas las decls)
#   reducible : True si admite optimización de espacio a dos filas
#   expected  : salida entera esperada

EJEMPLOS = {
    "lcs": {
        "func": "LCS",
        "decls": "vector<char> A = {'#','A','G','G','T','A','B'};\n"
                 "    vector<char> B = {'#','G','X','T','X','A','Y','B'};",
        "data": ["A", "B"],
        "scalars": ["6", "7"],
        "initial": ["6", "7"],
        "ctor": ["6", "7", "A", "B"],
        "reducible": True,
        "expected": 4,
    },
    "lis": {
        "func": "LIS",
        # LIS que termina en la posición N; reducción con filtro a[k] < a[i].
        "decls": "vector<int> a = {0,1,3,2,4};",
        "data": ["a"],
        "scalars": ["4"],
        "initial": ["4"],
        "ctor": ["4", "a"],
        "reducible": False,
        "expected": 3,
    },
    "mochila": {
        "func": "mochila",
        "decls": "vector<int> v = {0,1,4,5,7};\n"
                 "    vector<int> w = {0,1,3,4,5};",
        "data": ["v", "w"],
        "scalars": ["4", "7"],
        "initial": ["4", "7"],
        "ctor": ["4", "7", "v", "w"],
        "reducible": True,
        "expected": 9,
    },
    "camino": {
        "func": "camino",
        "decls": "vector<vector<int>> coste = {{0,0,0,0},{0,1,3,1},{0,1,5,1},{0,4,2,1}};",
        "data": ["coste"],
        "scalars": ["3", "3"],
        "initial": ["3", "3"],
        "ctor": ["3", "3", "coste"],
        "reducible": True,
        "expected": 7,
    },
    "secmatrices": {
        "func": "secMatrices",
        # Base-1: d[1..N+1] = dimensiones (A_i es d[i] x d[i+1]); d[0] centinela.
        "decls": "vector<int> d = {0,10,20,30,40};",
        "data": ["d"],
        "scalars": ["3"],
        "initial": ["1", "3"],
        "ctor": ["3", "d"],
        "reducible": False,   # DP de intervalos: no reducible a dos filas
        "expected": 18000,
    },
    "binom": {
        "func": "binom",
        "decls": "",
        "data": [],
        "scalars": ["5", "2"],
        "initial": ["5", "2"],
        "ctor": ["5", "2"],
        "reducible": True,
        "expected": 10,
    },
    "fact": {
        "func": "fact",
        "decls": "",
        "data": [],
        "scalars": ["5"],
        "initial": ["5"],
        "ctor": ["5"],
        "reducible": True,    # 1 parámetro, ventana 1 → buffer circular O(1)
        "expected": 120,
    },
    "fibonacci": {
        "func": "fib",
        "decls": "",
        "data": [],
        "scalars": ["10"],
        "initial": ["10"],
        "ctor": ["10"],
        "reducible": True,    # ventana 2 → buffer circular O(1)
        "expected": 55,
    },
    "edit": {
        "func": "edit",
        "decls": "vector<char> A = {'#','a','b','c'};\n"
                 "    vector<char> B = {'#','a','b','d'};",
        "data": ["A", "B"],
        "scalars": ["3", "3"],
        "initial": ["3", "3"],
        "ctor": ["3", "3", "A", "B"],
        "reducible": True,
        "expected": 1,
    },
    "paths": {
        "func": "paths",
        "decls": "",
        "data": [],
        "scalars": ["3", "3"],
        "initial": ["3", "3"],
        "ctor": ["3", "3"],
        "reducible": True,
        "expected": 6,
    },
    "rod": {
        "func": "rod",
        "decls": "vector<int> price = {0,1,5,8,9};",
        "data": ["price"],
        "scalars": ["4"],
        "initial": ["4"],
        "ctor": ["4", "price"],
        "reducible": False,   # 1 parámetro
        "expected": 10,
    },
    "trib": {
        "func": "trib",
        "decls": "",
        "data": [],
        "scalars": ["10"],
        "initial": ["10"],
        "ctor": ["10"],
        "reducible": True,    # ventana 3 → buffer circular O(1)
        "expected": 149,
    },
}

# Casos que deben ser RECHAZADOS por el análisis semántico/terminación.
NEGATIVOS = ["creciente", "sin_progreso"]


# ---------------------------------------------------------------------------
# Construcción del driver de prueba (main) según estrategia y modo
# ---------------------------------------------------------------------------

def construir_driver(meta: dict, estrategia: str, modo: str, space_opt: bool) -> str:
    """Genera un `main()` que invoca el código generado con los datos de prueba.

    La forma de la llamada depende de:
      - modo función vs clase,
      - sin-memo (expone parámetros crudos → llamada inicial canónica),
      - space-opt en clase reducible (operator() sin argumentos).
    """
    func = meta["func"]
    lineas = ["", "#include <iostream>", "int main() {"]
    if meta["decls"]:
        lineas.append("    " + meta["decls"])

    if modo == "funcion":
        args = list(meta["data"])
        args += meta["initial"] if estrategia == "sin-memo" else meta["scalars"]
        lineas.append(f"    std::cout << {func}({', '.join(args)}) << std::endl;")
    else:  # clase
        lineas.append(f"    {func} obj({', '.join(meta['ctor'])});")
        if estrategia == "bottom-up" and space_opt and meta["reducible"]:
            lineas.append("    std::cout << obj() << std::endl;")
        else:
            lineas.append(f"    std::cout << obj({', '.join(meta['initial'])}) << std::endl;")

    lineas.append("    return 0;")
    lineas.append("}")
    return "\n".join(lineas)


def cpp_completo(nombre: str, estrategia: str, modo: str, space_opt: bool) -> str:
    """Genera el C++ del ejemplo y le añade el driver de prueba."""
    from codigoPD import validar_entrada
    with open(os.path.join(EJEMPLOS_DIR, nombre + ".dp"), encoding="utf-8") as f:
        fuente = f.read()
    ok, salida = validar_entrada(fuente, modo=modo, algoritmo=estrategia, space_opt=space_opt)
    if not ok:
        raise AssertionError(f"generación falló para {nombre}/{estrategia}/{modo}:\n{salida}")
    return salida + "\n" + construir_driver(EJEMPLOS[nombre], estrategia, modo, space_opt)


# ---------------------------------------------------------------------------
# Localización del compilador y compilación
# ---------------------------------------------------------------------------

_compilador = None  # cache: ("gpp", None) | ("msvc", env) | ("none", None)


def _cargar_env_msvc():
    """Localiza vcvars64.bat, lo ejecuta una vez y captura el entorno resultante
    (para invocar cl.exe sin re-ejecutar el batch en cada compilación)."""
    patrones = [
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
    ]
    vcvars = None
    for patron in patrones:
        encontrados = sorted(glob.glob(patron))
        if encontrados:
            vcvars = encontrados[-1]  # la versión más reciente
            break
    if vcvars is None:
        return None

    res = subprocess.run(
        f'"{vcvars}" >nul 2>&1 && set',
        shell=True, capture_output=True, text=True,
    )
    if res.returncode != 0:
        return None

    env = {}
    for linea in res.stdout.splitlines():
        if "=" in linea:
            clave, valor = linea.split("=", 1)
            env[clave] = valor
    return env if env else None


def _detectar_compilador():
    global _compilador
    if _compilador is not None:
        return _compilador
    if shutil.which("g++"):
        _compilador = ("gpp", None)
    else:
        env = _cargar_env_msvc()
        _compilador = ("msvc", env) if env else ("none", None)
    return _compilador


def compilador_disponible() -> bool:
    return _detectar_compilador()[0] != "none"


def nombre_compilador() -> str:
    return _detectar_compilador()[0]


def _buscar_exe(nombre: str, env: dict):
    rutas = env.get("PATH") or env.get("Path") or ""
    for d in rutas.split(os.pathsep):
        cand = os.path.join(d, nombre)
        if os.path.isfile(cand):
            return cand
    return None


def compilar_y_ejecutar(cpp_source: str) -> str:
    """Compila el C++ dado y devuelve su stdout. Lanza RuntimeError si falla
    la compilación o no hay compilador."""
    kind, env = _detectar_compilador()
    if kind == "none":
        raise RuntimeError("no hay compilador C++ disponible")

    tmp = tempfile.mkdtemp(prefix="dptest_")
    try:
        src = os.path.join(tmp, "prog.cpp")
        with open(src, "w", encoding="utf-8") as f:
            f.write(cpp_source)
        exe = os.path.join(tmp, "prog.exe" if os.name == "nt" else "prog")

        if kind == "gpp":
            comp = subprocess.run(
                ["g++", "-std=c++17", "-O2", src, "-o", exe],
                capture_output=True, text=True, errors="replace",
            )
            if comp.returncode != 0:
                raise RuntimeError("g++ falló:\n" + comp.stderr)
        else:  # msvc
            cl = _buscar_exe("cl.exe", env) or "cl.exe"
            comp = subprocess.run(
                [cl, "/nologo", "/EHsc", "/std:c++17", f"/Fe:{exe}", src],
                capture_output=True, text=True, errors="replace", env=env, cwd=tmp,
            )
            if comp.returncode != 0:
                raise RuntimeError("cl falló:\n" + comp.stdout + comp.stderr)

        run = subprocess.run([exe], capture_output=True, text=True, errors="replace")
        if run.returncode != 0:
            raise RuntimeError(f"ejecución terminó con código {run.returncode}:\n{run.stderr}")
        return run.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
