"""Punto de entrada del compilador. Orquesta las fases del compilador
(sintaxis, semántica, terminación, índices) y la generación de código, y
ofrece la interfaz de línea de comandos.

Organización en módulos:
    modelo.py       tipos y AST
    sintaxis.py     gramática, parser e IRBuilder
    terminacion.py  función de cota, lema de Farkas, SMT y verificación de índices
    semantica.py    comprobaciones de tipos
    generacion.py   generadores de C++
"""
import argparse
import sys
from lark import UnexpectedInput
from modelo import *
from sintaxis import *
from terminacion import *
from semantica import *
from generacion import *


def validar_entrada(codigo: str, modo: str = "funcion", algoritmo: str = "top-down",
                    space_opt: bool = False, usar_smt: bool = False,
                    reconstruir: bool = False):
    # sintactica
    try:
        tree_lark = parser.parse(codigo)
    except UnexpectedInput as e:
        context = e.get_context(codigo)
        return False, f"[Sintaxis] L{e.line}:{e.column} cerca de -> {context.strip()}"

    tree = IRBuilder().transform(tree_lark)

    if usar_smt and not Z3_DISPONIBLE:
        return False, ("[SMT] La opción --smt requiere el paquete 'z3-solver'. "
                       "Instálalo con:  pip install z3-solver")

    # semantica (con terminación vía SMT si se pidió)
    checker = SemanticChecks()
    try:
        checker.validar_programa(tree, usar_smt=usar_smt)
    except ValueError as semerr:
        return False, str(semerr)

    # Verificación de índices fuera de rango (BLOQUEANTE). Gracias a la
    # inferencia de cotas inferiores no produce falsos positivos en las
    # familias estándar, así que una violación se trata como error y detiene
    # la generación. Con el solver propio por defecto, o con Z3 si --smt.
    verif = VerificadoraSMT() if usar_smt else TerminacionVerificadora()
    avisos_idx = VerificadorIndices(tree, verif).analizar()
    if avisos_idx:
        motor = "Z3" if usar_smt else "solver propio"
        detalle = "\n".join(f"  - {a}" for a in avisos_idx)
        return False, (f"[Índices fuera de rango] ({motor}) la recurrencia puede "
                       f"acceder fuera de rango:\n{detalle}")

    try:
        if reconstruir:
            # La reconstrucción necesita la tabla completa: usa su propio
            # generador (bottom-up + trace-back), ignorando --space-opt.
            generador = ReconstruccionGenerator(modo=modo)
        elif algoritmo == "sin-memo":
            generador = SinMemoGenerator(modo=modo)
        elif algoritmo == "bottom-up" and space_opt:
            generador = SpaceOptGenerator(modo=modo)
        elif algoritmo == "bottom-up":
            generador = BottomUpGenerator(modo=modo)
        else:
            generador = CodeGenerator(modo=modo)
        codigo_cpp = generador.generar(tree)
        return True, codigo_cpp
    except ValueError as elegib_err:
        # Caso esperado: la recurrencia no es reconstruible (mensaje claro).
        return False, str(elegib_err)
    except Exception as codegen_err:
        import traceback
        trace = traceback.format_exc()
        return False, f"[Generación de Código C++] {str(codegen_err)}\n{trace}"


def construir_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codigoPD",
        description="Compilador de un DSL para problemas de programación dinámica."
    )
    p.add_argument("archivo", help="Ruta al fichero .dp con la recurrencia.")
    p.add_argument(
        "--algoritmo", choices=["sin-memo", "top-down", "bottom-up"],
        default="top-down",
        help="Estrategia para el código C++: 'sin-memo' (recursión directa, "
             "referencia), 'top-down' (recursivo memoizado) o 'bottom-up' (iterativo)."
    )
    p.add_argument(
        "--space-opt", action="store_true",
        help="Optimización de espacio: reduce la tabla a dos filas cuando la "
             "recurrencia lo permite (requiere --algoritmo bottom-up)."
    )
    p.add_argument(
        "--gen", choices=["funcion", "clase"], default="funcion",
        help="Estilo del esqueleto C++: pareja de funciones o clase con operator()."
    )
    p.add_argument(
        "--out", "-o", default=None,
        help="Fichero de salida para el C++ generado. Por defecto, stdout."
    )
    p.add_argument(
        "--smt", action="store_true",
        help="Descargar las verificaciones (terminación e índices) con Z3 en "
             "lugar del solver propio. La comprobación de índices se hace SIEMPRE; "
             "esta bandera solo cambia el motor (requiere 'pip install z3-solver')."
    )
    p.add_argument(
        "--reconstruir", action="store_true",
        help="Genera además una reconstrucción de la solución óptima: la "
             "secuencia de estados (celdas) del camino óptimo, o la "
             "parentización (lista de cortes) en los DP de intervalos como el "
             "producto de matrices. Usa la tabla completa (bottom-up); "
             "incompatible con --space-opt."
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = construir_cli().parse_args(argv)

    # Avisos para banderas inconsistentes.
    if args.reconstruir and args.space_opt:
        print(
            "[aviso] --reconstruir necesita la tabla completa; es incompatible "
            "con --space-opt, que se ignora.",
            file=sys.stderr,
        )
        args.space_opt = False
    if args.space_opt and args.algoritmo != "bottom-up":
        print(
            "[aviso] --space-opt requiere --algoritmo bottom-up; se ignora.",
            file=sys.stderr,
        )
        args.space_opt = False

    try:
        with open(args.archivo, "r", encoding="utf-8") as f:
            codigo_fuente = f.read()
    except OSError as e:
        print(f"[error] no se pudo leer '{args.archivo}': {e}", file=sys.stderr)
        return 2

    ok, salida = validar_entrada(
        codigo_fuente, modo=args.gen, algoritmo=args.algoritmo,
        space_opt=args.space_opt, usar_smt=args.smt, reconstruir=args.reconstruir,
    )
    if not ok:
        print(salida, file=sys.stderr)
        return 1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(salida)
    else:
        print(salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
